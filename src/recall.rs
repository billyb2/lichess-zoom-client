use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use tokio::time::sleep;

pub const DEFAULT_WEBHOOK_URL: &str = "https://billy-recall.ngrok.app";
pub const RECALL_RATE_LIMIT_MAX_ATTEMPTS: usize = 8;
pub const RECALL_RATE_LIMIT_INITIAL_DELAY_MS: u64 = 300;
pub const RECALL_RATE_LIMIT_MAX_DELAY_MS: u64 = 5_000;

pub const ALL_WEBHOOK_EVENTS: [&str; 13] = [
    "participant_events.join",
    "participant_events.leave",
    "participant_events.update",
    "participant_events.speech_on",
    "participant_events.speech_off",
    "participant_events.webcam_on",
    "participant_events.webcam_off",
    "participant_events.screenshare_on",
    "participant_events.screenshare_off",
    "participant_events.chat_message",
    "transcript.data",
    "transcript.partial_data",
    "transcript.provider_data",
];
pub const VOICE_COMMAND_WEBHOOK_EVENTS: [&str; 6] = [
    "participant_events.join",
    "participant_events.leave",
    "participant_events.update",
    "transcript.data",
    "transcript.partial_data",
    "transcript.provider_data",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceCommandTranscriptMode {
    PrioritizeLowLatency,
    PrioritizeAccuracy,
}

impl VoiceCommandTranscriptMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PrioritizeLowLatency => "prioritize_low_latency",
            Self::PrioritizeAccuracy => "prioritize_accuracy",
        }
    }

    fn provider_config(self) -> JsonValue {
        match self {
            Self::PrioritizeLowLatency => json!({
                "mode": "prioritize_low_latency",
                "language_code": "en"
            }),
            Self::PrioritizeAccuracy => json!({
                "mode": "prioritize_accuracy"
            }),
        }
    }
}

pub fn voice_command_recallai_provider(mode: VoiceCommandTranscriptMode) -> JsonValue {
    json!({
        "recallai_streaming": mode.provider_config()
    })
}

pub fn voice_command_deepgram_provider(
    model: impl AsRef<str>,
    language: impl AsRef<str>,
) -> JsonValue {
    json!({
        "deepgram_streaming": {
            "model": model.as_ref(),
            "language": language.as_ref()
        }
    })
}

pub fn voice_command_assembly_ai_provider(
    speech_model: impl AsRef<str>,
    format_turns: bool,
    keyterms_prompt: &[String],
) -> JsonValue {
    let mut provider = json!({
        "speech_model": speech_model.as_ref(),
        "format_turns": format_turns
    });

    if !keyterms_prompt.is_empty() {
        provider["keyterms_prompt"] = JsonValue::Array(
            keyterms_prompt
                .iter()
                .cloned()
                .map(JsonValue::String)
                .collect(),
        );
    }

    json!({
        "assembly_ai_v3_streaming": provider
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bot {
    pub id: String,
    pub name: String,
    pub meeting_url: String,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Clone)]
pub struct RecallClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl RecallClient {
    pub fn new(api_key: impl Into<String>, region: impl AsRef<str>) -> Self {
        let region = region.as_ref().trim().trim_matches('/');
        let base_url = if region.starts_with("http://") || region.starts_with("https://") {
            region.to_string()
        } else {
            format!("https://{region}.recall.ai")
        };

        Self {
            api_key: api_key.into(),
            base_url,
            http: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(api_key: impl Into<String>, base_url: impl AsRef<str>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.as_ref().trim().trim_matches('/').to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub async fn create_bot(&self, request: &CreateBotRequest) -> Result<Bot, RecallApiError> {
        let response = self.post_json_with_retry("/api/v1/bot/", request).await?;
        let response = Self::ensure_success(response).await?;
        let response_body = response.text().await.map_err(RecallApiError::Http)?;
        let response = parse_create_bot_response(&response_body)?;
        Ok(Bot {
            id: response.id,
            name: response
                .bot_name
                .unwrap_or_else(|| request.bot_name.clone()),
            meeting_url: response
                .meeting_url
                .unwrap_or_else(|| request.meeting_url.clone()),
            extra: response.extra,
        })
    }

    pub async fn list_bots(&self) -> Result<Vec<Bot>, RecallApiError> {
        self.list_bots_from_url(format!("{}/api/v1/bot/", self.base_url))
            .await
    }

    pub async fn list_bots_filtered(
        &self,
        filters: &[(&str, &str)],
    ) -> Result<Vec<Bot>, RecallApiError> {
        let mut url =
            reqwest::Url::parse(&format!("{}/api/v1/bot/", self.base_url)).map_err(|err| {
                RecallApiError::InvalidResponse {
                    message: format!("failed to build filtered bot list URL: {err}"),
                }
            })?;
        {
            let mut query = url.query_pairs_mut();
            for (key, value) in filters {
                query.append_pair(key, value);
            }
        }
        self.list_bots_from_url(url.to_string()).await
    }

    async fn list_bots_from_url(&self, initial_url: String) -> Result<Vec<Bot>, RecallApiError> {
        let mut bots = Vec::new();
        let mut next_url = Some(initial_url);

        while let Some(url) = next_url.take() {
            let response = self.get_absolute_with_retry(&url).await?;
            let response = Self::ensure_success(response).await?;
            let response_body = response.text().await.map_err(RecallApiError::Http)?;
            let (page_bots, page_next) = parse_bot_list_response(&response_body)?;
            bots.extend(page_bots);
            next_url = page_next;
        }

        Ok(bots)
    }

    pub async fn leave_call(&self, bot: Bot) -> Result<(), RecallApiError> {
        let path = format!("/api/v1/bot/{}/leave_call/", bot.id);
        let response = self.post_with_retry(&path).await?;
        let _ = Self::ensure_success(response).await?;
        Ok(())
    }

    pub async fn send_chat_message(
        &self,
        bot_id: &str,
        to: &str,
        message: &str,
    ) -> Result<(), RecallApiError> {
        let path = format!("/api/v1/bot/{bot_id}/send_chat_message/");
        let payload = json!({
            "to": to,
            "message": message,
        });
        let response = self.post_json_with_retry(&path, &payload).await?;
        let _ = Self::ensure_success(response).await?;
        Ok(())
    }

    pub async fn output_video_jpeg(
        &self,
        bot_id: &str,
        jpeg_bytes: &[u8],
    ) -> Result<(), RecallApiError> {
        let path = format!("/api/v1/bot/{bot_id}/output_video/");
        let payload = json!({
            "kind": "jpeg",
            "b64_data": BASE64_STANDARD.encode(jpeg_bytes),
        });
        let response = self.post_json_with_retry(&path, &payload).await?;
        let _ = Self::ensure_success(response).await?;
        Ok(())
    }

    async fn get_with_retry(&self, path: &str) -> Result<reqwest::Response, RecallApiError> {
        let url = format!("{}{}", self.base_url, path);
        self.get_absolute_with_retry(&url).await
    }

    async fn get_absolute_with_retry(
        &self,
        url: &str,
    ) -> Result<reqwest::Response, RecallApiError> {
        self.send_with_retry(|| {
            self.http
                .get(url)
                .header("Authorization", self.auth_header_value())
        })
        .await
    }

    async fn post_with_retry(&self, path: &str) -> Result<reqwest::Response, RecallApiError> {
        let url = format!("{}{}", self.base_url, path);
        self.send_with_retry(|| {
            self.http
                .post(&url)
                .header("Authorization", self.auth_header_value())
        })
        .await
    }

    async fn post_json_with_retry<T: Serialize + ?Sized>(
        &self,
        path: &str,
        payload: &T,
    ) -> Result<reqwest::Response, RecallApiError> {
        let url = format!("{}{}", self.base_url, path);
        self.send_with_retry(|| {
            self.http
                .post(&url)
                .header("Authorization", self.auth_header_value())
                .json(payload)
        })
        .await
    }

    async fn send_with_retry<F>(&self, make_request: F) -> Result<reqwest::Response, RecallApiError>
    where
        F: Fn() -> reqwest::RequestBuilder,
    {
        let mut fallback_delay = Duration::from_millis(RECALL_RATE_LIMIT_INITIAL_DELAY_MS);

        for attempt in 1..=RECALL_RATE_LIMIT_MAX_ATTEMPTS {
            let response = make_request().send().await.map_err(RecallApiError::Http)?;
            if response.status() != StatusCode::TOO_MANY_REQUESTS {
                return Ok(response);
            }

            let retry_delay =
                retry_after_from_headers(response.headers()).unwrap_or(fallback_delay);
            let body = response.text().await.unwrap_or_default();
            if attempt == RECALL_RATE_LIMIT_MAX_ATTEMPTS {
                return Err(RecallApiError::Api {
                    status: StatusCode::TOO_MANY_REQUESTS,
                    body,
                });
            }

            eprintln!(
                "Recall API returned 429 (attempt {attempt}/{}). Retrying in {:?}",
                RECALL_RATE_LIMIT_MAX_ATTEMPTS, retry_delay
            );
            sleep(retry_delay).await;

            let next_delay_ms = (fallback_delay.as_millis() as u64)
                .saturating_mul(2)
                .min(RECALL_RATE_LIMIT_MAX_DELAY_MS);
            fallback_delay = Duration::from_millis(next_delay_ms);
        }

        unreachable!("retry loop must return response or error")
    }

    async fn ensure_success(
        response: reqwest::Response,
    ) -> Result<reqwest::Response, RecallApiError> {
        let status = response.status();
        if status.is_success() {
            Ok(response)
        } else {
            let body = response.text().await.unwrap_or_default();
            Err(RecallApiError::Api { status, body })
        }
    }

    fn auth_header_value(&self) -> String {
        format!("Token {}", self.api_key)
    }
}

fn retry_after_from_headers(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let retry_after = headers.get(reqwest::header::RETRY_AFTER)?;
    let retry_after = retry_after.to_str().ok()?.trim();
    let seconds = retry_after.parse::<u64>().ok()?;
    Some(Duration::from_secs(seconds))
}

#[derive(Debug)]
pub enum RecallApiError {
    Http(reqwest::Error),
    Api { status: StatusCode, body: String },
    InvalidResponse { message: String },
}

impl fmt::Display for RecallApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "{err}"),
            Self::Api { status, body } => write!(f, "Recall API error ({status}): {body}"),
            Self::InvalidResponse { message } => {
                write!(f, "Recall API returned an unexpected response: {message}")
            }
        }
    }
}

impl Error for RecallApiError {}

fn parse_create_bot_response(body: &str) -> Result<CreateBotResponse, RecallApiError> {
    let payload: JsonValue =
        serde_json::from_str(body).map_err(|err| RecallApiError::InvalidResponse {
            message: format!(
                "failed to parse JSON: {err}; body={}",
                truncate_for_error(body, 1_500)
            ),
        })?;

    let id = value_at_any(&payload, &["/id", "/bot/id", "/bot_id"])
        .and_then(scalar_to_string)
        .ok_or_else(|| RecallApiError::InvalidResponse {
            message: format!(
                "missing required bot id in create response; body={}",
                truncate_for_error(body, 1_500)
            ),
        })?;

    let bot_name = value_at_any(
        &payload,
        &["/bot_name", "/name", "/bot/bot_name", "/bot/name"],
    )
    .and_then(scalar_to_string);
    let meeting_url = value_at_any(
        &payload,
        &[
            "/meeting_url",
            "/meeting_url_normalized",
            "/bot/meeting_url",
            "/bot/meeting_url_normalized",
        ],
    )
    .and_then(scalar_to_string);

    let mut extra = BTreeMap::new();
    if let JsonValue::Object(map) = &payload {
        for (key, value) in map {
            if key != "id"
                && key != "bot_id"
                && key != "bot_name"
                && key != "name"
                && key != "meeting_url"
                && key != "meeting_url_normalized"
            {
                extra.insert(key.clone(), value.clone());
            }
        }
    }

    Ok(CreateBotResponse {
        id,
        bot_name,
        meeting_url,
        extra,
    })
}

fn parse_bot_list_response(body: &str) -> Result<(Vec<Bot>, Option<String>), RecallApiError> {
    let payload: JsonValue =
        serde_json::from_str(body).map_err(|err| RecallApiError::InvalidResponse {
            message: format!(
                "failed to parse bot list JSON: {err}; body={}",
                truncate_for_error(body, 1_500)
            ),
        })?;

    let mut next_url = None;
    let bot_items = match &payload {
        JsonValue::Array(items) => items.clone(),
        JsonValue::Object(obj) => {
            next_url = obj
                .get("next")
                .and_then(|value| value.as_str())
                .filter(|url| !url.trim().is_empty())
                .map(ToOwned::to_owned);
            match obj.get("results") {
                Some(JsonValue::Array(items)) => items.clone(),
                _ if obj.contains_key("id") || obj.contains_key("bot_id") => vec![payload.clone()],
                _ => Vec::new(),
            }
        }
        _ => Vec::new(),
    };

    let mut bots = Vec::new();
    for item in bot_items {
        if let Some(bot) = parse_bot_from_value(&item) {
            bots.push(bot);
        }
    }

    Ok((bots, next_url))
}

fn parse_bot_from_value(value: &JsonValue) -> Option<Bot> {
    let id = value_at_any(value, &["/id", "/bot_id", "/bot/id"]).and_then(scalar_to_string)?;
    let name = value_at_any(value, &["/bot_name", "/name", "/bot/bot_name", "/bot/name"])
        .and_then(scalar_to_string)
        .unwrap_or_else(|| id.clone());
    let meeting_url = value_at_any(
        value,
        &[
            "/meeting_url",
            "/meeting_url_normalized",
            "/bot/meeting_url",
            "/bot/meeting_url_normalized",
        ],
    )
    .and_then(scalar_to_string)
    .unwrap_or_default();

    let mut extra = BTreeMap::new();
    if let JsonValue::Object(obj) = value {
        for (key, item) in obj {
            if key != "id"
                && key != "bot_id"
                && key != "bot_name"
                && key != "name"
                && key != "meeting_url"
                && key != "meeting_url_normalized"
            {
                extra.insert(key.clone(), item.clone());
            }
        }
    }

    Some(Bot {
        id,
        name,
        meeting_url,
        extra,
    })
}

fn value_at_any<'a>(payload: &'a JsonValue, pointers: &[&str]) -> Option<&'a JsonValue> {
    pointers.iter().find_map(|pointer| payload.pointer(pointer))
}

fn scalar_to_string(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(value) if !value.trim().is_empty() => Some(value.clone()),
        JsonValue::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn truncate_for_error(value: &str, max_len: usize) -> String {
    if value.len() <= max_len {
        value.to_string()
    } else {
        format!("{}...", &value[..max_len])
    }
}

pub fn webhook_realtime_endpoint(url: impl AsRef<str>) -> JsonValue {
    webhook_realtime_endpoint_with_events(url, &ALL_WEBHOOK_EVENTS)
}

pub fn webhook_realtime_endpoint_with_events(url: impl AsRef<str>, events: &[&str]) -> JsonValue {
    json!({
        "type": "webhook",
        "url": url.as_ref(),
        "events": events,
    })
}

pub fn recording_config_with_all_webhook_events(url: impl AsRef<str>) -> JsonValue {
    json!({
        "participant_events": {},
        "transcript": {
            "provider": {
                "recallai_streaming": {}
            }
        },
        "realtime_endpoints": [webhook_realtime_endpoint(url)]
    })
}

pub fn recording_config_for_voice_command_transcripts(url: impl AsRef<str>) -> JsonValue {
    recording_config_for_voice_command_transcripts_with_mode(
        url,
        VoiceCommandTranscriptMode::PrioritizeLowLatency,
    )
}

pub fn recording_config_for_voice_command_transcripts_with_mode(
    url: impl AsRef<str>,
    mode: VoiceCommandTranscriptMode,
) -> JsonValue {
    recording_config_for_voice_command_transcripts_with_provider(
        url,
        voice_command_recallai_provider(mode),
    )
}

pub fn recording_config_for_voice_command_transcripts_with_provider(
    url: impl AsRef<str>,
    provider: JsonValue,
) -> JsonValue {
    json!({
        "participant_events": {},
        "transcript": {
            "provider": provider,
            "diarization": {
                "use_separate_streams_when_available": true
            }
        },
        "realtime_endpoints": [webhook_realtime_endpoint_with_events(url, &VOICE_COMMAND_WEBHOOK_EVENTS)]
    })
}

pub fn parse_realtime_webhooks(
    payload: &JsonValue,
) -> Vec<Result<RecallRealtimeWebhook, serde_json::Error>> {
    match payload {
        JsonValue::Array(items) => items
            .iter()
            .map(|item| serde_json::from_value(item.clone()))
            .collect(),
        _ => vec![serde_json::from_value(payload.clone())],
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "event", content = "data")]
pub enum RecallRealtimeWebhook {
    #[serde(rename = "participant_events.join")]
    ParticipantJoin(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.leave")]
    ParticipantLeave(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.update")]
    ParticipantUpdate(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.speech_on")]
    ParticipantSpeechOn(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.speech_off")]
    ParticipantSpeechOff(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.webcam_on")]
    ParticipantWebcamOn(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.webcam_off")]
    ParticipantWebcamOff(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.screenshare_on")]
    ParticipantScreenshareOn(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.screenshare_off")]
    ParticipantScreenshareOff(ParticipantEventWebhookPayload),
    #[serde(rename = "participant_events.chat_message")]
    ParticipantChatMessage(ParticipantEventWebhookPayload),
    #[serde(rename = "transcript.data")]
    TranscriptData(TranscriptWebhookPayload),
    #[serde(rename = "transcript.partial_data")]
    TranscriptPartialData(TranscriptWebhookPayload),
    #[serde(rename = "transcript.provider_data")]
    TranscriptProviderData(JsonValue),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParticipantEventWebhookPayload {
    pub data: ParticipantEventData,
    pub realtime_endpoint: RecallResourceRef,
    #[serde(default)]
    pub participant_events: Option<RecallResourceRef>,
    pub recording: RecallResourceRef,
    #[serde(default)]
    pub bot: Option<RecallResourceRef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParticipantEventData {
    pub participant: ParticipantIdentity,
    pub timestamp: EventTimestamp,
    #[serde(default)]
    pub data: Option<ParticipantChatData>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParticipantIdentity {
    pub id: i64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub is_host: Option<bool>,
    #[serde(default)]
    pub platform: Option<String>,
    #[serde(default)]
    pub extra_data: JsonValue,
    #[serde(default)]
    pub email: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EventTimestamp {
    #[serde(default)]
    pub absolute: Option<String>,
    pub relative: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParticipantChatData {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptWebhookPayload {
    pub data: TranscriptEventData,
    pub realtime_endpoint: RecallResourceRef,
    pub transcript: RecallResourceRef,
    pub recording: RecallResourceRef,
    #[serde(default)]
    pub bot: Option<RecallResourceRef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptEventData {
    #[serde(default)]
    pub words: Vec<TranscriptWord>,
    #[serde(default)]
    pub participant: Option<ParticipantIdentity>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptWord {
    pub text: String,
    #[serde(default)]
    pub start_timestamp: Option<RelativeTimestamp>,
    #[serde(default)]
    pub end_timestamp: Option<RelativeTimestamp>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RelativeTimestamp {
    pub relative: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecallResourceRef {
    pub id: String,
    #[serde(default)]
    pub metadata: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateBotRequest {
    pub meeting_url: String,
    pub bot_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub join_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recording_config: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_media: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub automatic_video_output: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub automatic_audio_output: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub automatic_leave: Option<AutomaticLeaveConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<BotVariantConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zoom: Option<ZoomConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_meet: Option<GoogleMeetConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breakout_room: Option<BreakoutRoomConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Option<String>>>,
    #[serde(flatten, skip_serializing_if = "BTreeMap::is_empty", default)]
    pub extra: BTreeMap<String, JsonValue>,
}

impl Default for CreateBotRequest {
    fn default() -> Self {
        Self::new(String::new())
    }
}

impl CreateBotRequest {
    pub fn new(meeting_url: impl Into<String>) -> Self {
        Self {
            meeting_url: meeting_url.into(),
            bot_name: "Meeting Notetaker".to_string(),
            join_at: None,
            recording_config: None,
            output_media: None,
            automatic_video_output: None,
            automatic_audio_output: None,
            chat: None,
            automatic_leave: None,
            variant: None,
            zoom: None,
            google_meet: None,
            breakout_room: None,
            metadata: None,
            extra: BTreeMap::new(),
        }
    }

    pub fn with_all_webhook_events(mut self, webhook_url: impl AsRef<str>) -> Self {
        self.configure_all_webhook_events(webhook_url);
        self
    }

    pub fn configure_all_webhook_events(&mut self, webhook_url: impl AsRef<str>) {
        let webhook_url = webhook_url.as_ref().to_string();
        let mut recording_config = match self.recording_config.take() {
            Some(JsonValue::Object(obj)) => obj,
            _ => JsonMap::new(),
        };

        recording_config
            .entry("participant_events".to_string())
            .or_insert_with(|| json!({}));
        recording_config
            .entry("transcript".to_string())
            .or_insert_with(|| {
                json!({
                    "provider": {
                        "recallai_streaming": {}
                    }
                })
            });

        let mut realtime_endpoints = match recording_config.remove("realtime_endpoints") {
            Some(JsonValue::Array(existing)) => existing
                .into_iter()
                .filter(|endpoint| {
                    endpoint.get("type") != Some(&JsonValue::String("webhook".to_string()))
                        || endpoint.get("url") != Some(&JsonValue::String(webhook_url.clone()))
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };

        realtime_endpoints.push(webhook_realtime_endpoint(&webhook_url));
        recording_config.insert(
            "realtime_endpoints".to_string(),
            JsonValue::Array(realtime_endpoints),
        );
        self.recording_config = Some(JsonValue::Object(recording_config));
    }

    pub fn configure_voice_command_transcript_webhooks(&mut self, webhook_url: impl AsRef<str>) {
        self.configure_voice_command_transcript_webhooks_with_mode(
            webhook_url,
            VoiceCommandTranscriptMode::PrioritizeLowLatency,
        );
    }

    pub fn configure_voice_command_transcript_webhooks_with_mode(
        &mut self,
        webhook_url: impl AsRef<str>,
        mode: VoiceCommandTranscriptMode,
    ) {
        self.configure_voice_command_transcript_webhooks_with_provider(
            webhook_url,
            voice_command_recallai_provider(mode),
        );
    }

    pub fn configure_voice_command_transcript_webhooks_with_provider(
        &mut self,
        webhook_url: impl AsRef<str>,
        provider: JsonValue,
    ) {
        let webhook_url = webhook_url.as_ref().to_string();
        let mut recording_config = match self.recording_config.take() {
            Some(JsonValue::Object(obj)) => obj,
            _ => JsonMap::new(),
        };

        recording_config
            .entry("participant_events".to_string())
            .or_insert_with(|| json!({}));
        recording_config.insert(
            "transcript".to_string(),
            json!({
                "provider": provider,
                "diarization": {
                    "use_separate_streams_when_available": true
                }
            }),
        );

        let mut realtime_endpoints = match recording_config.remove("realtime_endpoints") {
            Some(JsonValue::Array(existing)) => existing
                .into_iter()
                .filter(|endpoint| {
                    endpoint.get("type") != Some(&JsonValue::String("webhook".to_string()))
                        || endpoint.get("url") != Some(&JsonValue::String(webhook_url.clone()))
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };

        realtime_endpoints.push(webhook_realtime_endpoint_with_events(
            &webhook_url,
            &VOICE_COMMAND_WEBHOOK_EVENTS,
        ));
        recording_config.insert(
            "realtime_endpoints".to_string(),
            JsonValue::Array(realtime_endpoints),
        );

        self.recording_config = Some(JsonValue::Object(recording_config));
    }

    pub fn configure_transcript_without_video_recording(&mut self) {
        let mut recording_config = match self.recording_config.take() {
            Some(JsonValue::Object(obj)) => obj,
            _ => JsonMap::new(),
        };

        // Keep transcript + realtime webhook behavior, but explicitly disable
        // mixed video recording artifacts.
        recording_config.insert("video_mixed_mp4".to_string(), JsonValue::Null);
        self.recording_config = Some(JsonValue::Object(recording_config));
    }

    pub fn configure_automatic_video_output_jpeg(&mut self, jpeg_bytes: &[u8]) {
        let b64_data = BASE64_STANDARD.encode(jpeg_bytes);
        self.automatic_video_output = Some(json!({
            "in_call_not_recording": {
                "kind": "jpeg",
                "b64_data": b64_data
            },
            "in_call_recording": {
                "kind": "jpeg",
                "b64_data": b64_data
            }
        }));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutomaticLeaveConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waiting_room_timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noone_joined_timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub everyone_left_timeout: Option<AutomaticLeaveEveryoneLeft>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_call_not_recording_timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_call_recording_timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recording_permission_denied_timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub silence_detection: Option<AutomaticLeaveSilenceDetection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bot_detection: Option<AutomaticLeaveBotDetection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutomaticLeaveEveryoneLeft {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activate_after: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutomaticLeaveSilenceDetection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activate_after: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutomaticLeaveBotDetection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub using_participant_events: Option<AutomaticLeaveBotDetectionUsingParticipantEvents>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub using_participant_names: Option<AutomaticLeaveBotDetectionUsingParticipantNames>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutomaticLeaveBotDetectionUsingParticipantEvents {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activate_after: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutomaticLeaveBotDetectionUsingParticipantNames {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activate_after: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matches: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BotVariantConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zoom: Option<BotRuntimeVariant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_meet: Option<BotRuntimeVariant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub microsoft_teams: Option<BotRuntimeVariant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webex: Option<BotRuntimeVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BotRuntimeVariant {
    #[serde(rename = "web")]
    Web,
    #[serde(rename = "web_4_core")]
    Web4Core,
    #[serde(rename = "web_gpu")]
    WebGpu,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ZoomConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub join_token_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zak_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub obf_token_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_email: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoogleMeetConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_login_group_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum BreakoutRoomConfig {
    JoinMainRoom,
    JoinSpecificRoom { room_id: String },
    AutoAcceptAllInvites,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateBotResponse {
    pub id: String,
    #[serde(default)]
    pub bot_name: Option<String>,
    #[serde(default)]
    pub meeting_url: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, JsonValue>,
}
