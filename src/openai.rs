use std::error::Error;
use std::fmt;

use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com";
const DEFAULT_OPENAI_MODEL: &str = "gpt-5-mini";
const DEFAULT_OPENAI_REASONING_EFFORT: &str = "minimal";

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    api_key: String,
    base_url: String,
    model: String,
    reasoning_effort: String,
    http: reqwest::Client,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceCommandKind {
    Command,
    Chat,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoiceCommandClassification {
    pub kind: VoiceCommandKind,
    pub content: String,
}

impl OpenAiClient {
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").ok()?;
        if api_key.trim().is_empty() {
            return None;
        }

        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_OPENAI_BASE_URL.to_string())
            .trim()
            .trim_matches('/')
            .to_string();
        let model = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.to_string())
            .trim()
            .to_string();
        let reasoning_effort = std::env::var("OPENAI_REASONING_EFFORT")
            .unwrap_or_else(|_| DEFAULT_OPENAI_REASONING_EFFORT.to_string())
            .trim()
            .to_string();

        Some(Self {
            api_key,
            base_url,
            model,
            reasoning_effort,
            http: reqwest::Client::new(),
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn reasoning_effort(&self) -> &str {
        &self.reasoning_effort
    }

    pub async fn generate_text(&self, input: &str) -> Result<String, OpenAiApiError> {
        let payload = json!({
            "model": self.model,
            "input": input,
            "store": false,
            "reasoning": {
                "effort": self.reasoning_effort
            }
        });
        let payload = self.post_responses_payload(&payload).await?;
        extract_output_text(&payload).ok_or_else(|| {
            OpenAiApiError::InvalidResponse(format!(
                "missing output text in response; body={}",
                truncate_for_error(&payload.to_string(), 2_000)
            ))
        })
    }

    pub async fn classify_voice_command(
        &self,
        transcript_text: &str,
        legal_moves_uci: &[String],
    ) -> Result<VoiceCommandClassification, OpenAiApiError> {
        let legal_moves_line = if legal_moves_uci.is_empty() {
            "none".to_string()
        } else {
            legal_moves_uci.join(", ")
        };

        let payload = json!({
            "model": self.model,
            "store": false,
            "reasoning": {
                "effort": self.reasoning_effort
            },
            "input": [
                {
                    "role": "system",
                    "content": "You classify transcript text for a voice-to-text chess game.\nReturn JSON that matches the response schema.\nCategories:\n- command: a chess move instruction. Output exactly one UCI move string that is legal now.\n- chat: any non-move speech.\nUse the provided legal move list as ground truth for current position legality.\nStrongly prefer `command` when text sounds like a chess move, including spoken forms such as \"pawn to a four\", \"knight f3\", \"castle kingside\", or \"takes on e5\".\nOnly choose `chat` when the text is clearly conversational and not a move request.\nDo not include any additional keys or text."
                },
                {
                    "role": "user",
                    "content": format!(
                        "Transcript: {}\nLegal UCI moves now: {}\nIf kind=command, content must be one move from the legal list.",
                        transcript_text,
                        legal_moves_line
                    )
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "voice_command_classification",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["command", "chat"]
                            },
                            "content": {
                                "type": "string",
                                "minLength": 1
                            }
                        },
                        "required": ["kind", "content"]
                    }
                }
            }
        });

        let payload = self.post_responses_payload(&payload).await?;
        let output_text = extract_output_text(&payload).ok_or_else(|| {
            OpenAiApiError::InvalidResponse(format!(
                "missing output text for command classification; body={}",
                truncate_for_error(&payload.to_string(), 2_000)
            ))
        })?;
        serde_json::from_str::<VoiceCommandClassification>(output_text.trim()).map_err(|err| {
            OpenAiApiError::InvalidResponse(format!(
                "invalid command classification JSON: {err}; output={}",
                output_text.trim()
            ))
        })
    }

    async fn post_responses_payload(
        &self,
        payload: &JsonValue,
    ) -> Result<JsonValue, OpenAiApiError> {
        let url = format!("{}/v1/responses", self.base_url);
        let response = self
            .http
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(payload)
            .send()
            .await
            .map_err(OpenAiApiError::Http)?;
        let status = response.status();
        let body = response.text().await.map_err(OpenAiApiError::Http)?;
        if !status.is_success() {
            return Err(OpenAiApiError::Api { status, body });
        }

        serde_json::from_str(&body).map_err(|err| {
            OpenAiApiError::InvalidResponse(format!("failed to parse JSON: {err}; body={body}"))
        })
    }
}

fn extract_output_text(payload: &JsonValue) -> Option<String> {
    if let Some(text) = payload.get("output_text").and_then(JsonValue::as_str) {
        let text = text.trim();
        if !text.is_empty() {
            return Some(text.to_string());
        }
    }

    let mut texts = Vec::new();
    if let Some(items) = payload.get("output").and_then(JsonValue::as_array) {
        for item in items {
            if let Some(contents) = item.get("content").and_then(JsonValue::as_array) {
                for content in contents {
                    if let Some(text) = content.get("text").and_then(JsonValue::as_str) {
                        let text = text.trim();
                        if !text.is_empty() {
                            texts.push(text.to_string());
                        }
                    }
                }
            }
        }
    }
    if !texts.is_empty() {
        return Some(texts.join("\n"));
    }

    if let Some(choice_text) = payload
        .pointer("/choices/0/message/content")
        .and_then(JsonValue::as_str)
    {
        let choice_text = choice_text.trim();
        if !choice_text.is_empty() {
            return Some(choice_text.to_string());
        }
    }

    None
}

fn truncate_for_error(value: &str, max_len: usize) -> String {
    if value.len() <= max_len {
        value.to_string()
    } else {
        format!("{}...", &value[..max_len])
    }
}

#[derive(Debug)]
pub enum OpenAiApiError {
    Http(reqwest::Error),
    Api { status: StatusCode, body: String },
    InvalidResponse(String),
}

impl fmt::Display for OpenAiApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "{err}"),
            Self::Api { status, body } => write!(f, "OpenAI API error ({status}): {body}"),
            Self::InvalidResponse(message) => write!(f, "OpenAI API invalid response: {message}"),
        }
    }
}

impl Error for OpenAiApiError {}
