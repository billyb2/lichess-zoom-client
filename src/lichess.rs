use std::error::Error;
use std::fmt;
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};

const DEFAULT_LICHESS_BASE_URL: &str = "https://lichess.org";
const DEFAULT_RECONNECT_DELAY_MS: u64 = 1_500;
const DEFAULT_BROADCAST_VISIBILITY: &str = "public";
const DEFAULT_BROADCAST_NAME: &str = "Hackathon Zoom Relay";
const DEFAULT_BROADCAST_ROUND_NAME: &str = "Live Game";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LichessStreamMode {
    Board,
    Public,
}

impl LichessStreamMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Board => "board",
            Self::Public => "public",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LichessGameUpdate {
    pub mode: LichessStreamMode,
    pub moves: Option<String>,
    pub fen: Option<String>,
    pub status: Option<String>,
}

impl LichessGameUpdate {
    pub fn signature(&self) -> Option<String> {
        self.fen
            .as_ref()
            .map(|value| format!("fen:{value}"))
            .or_else(|| self.moves.as_ref().map(|value| format!("moves:{value}")))
    }
}

#[derive(Debug, Clone)]
pub struct LichessCreatedGame {
    pub game_id: String,
    pub share_url: String,
}

#[derive(Debug, Clone)]
pub struct LichessClient {
    base_url: String,
    game_id: String,
    api_token: Option<String>,
    mode: LichessStreamMode,
    reconnect_delay: Duration,
    http: reqwest::Client,
}

impl LichessClient {
    pub fn from_env() -> Option<Self> {
        let game_id_raw = std::env::var("LICHESS_BROADCAST_ROUND_ID")
            .or_else(|_| std::env::var("LICHESS_GAME_ID"))
            .or_else(|_| std::env::var("LICHESS_GAME_URL"))
            .ok()?;
        Self::from_env_for_game_id(game_id_raw)
    }

    pub fn from_env_for_game_id(game_id: impl AsRef<str>) -> Option<Self> {
        let game_id = parse_game_id(game_id.as_ref())?;
        let base_url = base_url_from_env();
        let api_token = api_token_from_env();

        let mode = match stream_mode_from_env(api_token.is_some()) {
            LichessStreamMode::Board => LichessStreamMode::Board,
            LichessStreamMode::Public => {
                eprintln!(
                    "Lichess public game stream endpoint is unavailable; using board stream mode"
                );
                LichessStreamMode::Board
            }
        };

        if api_token.is_none() {
            eprintln!(
                "Lichess board stream requires LICHESS_API_KEY (or LICHESS_API_TOKEN); disabling Lichess"
            );
            return None;
        }

        Some(Self {
            base_url,
            game_id,
            api_token,
            mode,
            reconnect_delay: reconnect_delay_from_env(),
            http: reqwest::Client::new(),
        })
    }

    pub async fn create_broadcast_round_from_env() -> Result<LichessCreatedGame, LichessApiError> {
        let api_key = api_token_from_env().ok_or_else(|| {
            LichessApiError::Unsupported(
                "LICHESS_API_KEY (or LICHESS_API_TOKEN) is required to create a broadcast on boot"
                    .to_string(),
            )
        })?;

        let base_url = base_url_from_env();
        let http = reqwest::Client::new();
        let tournament_response = http
            .post(format!("{}/broadcast/new", base_url))
            .header("Authorization", format!("Bearer {api_key}"))
            .form(&[
                ("name", broadcast_name_from_env()),
                ("visibility", broadcast_visibility_from_env()),
            ])
            .send()
            .await
            .map_err(LichessApiError::Http)?;
        let status = tournament_response.status();
        let body = tournament_response
            .text()
            .await
            .map_err(LichessApiError::Http)?;
        if !status.is_success() {
            return Err(scope_or_api_error(status, body, "study:write"));
        }
        let tournament_payload: LichessBroadcastTournamentCreateResponse =
            serde_json::from_str(&body).map_err(|err| {
                LichessApiError::InvalidResponse(format!(
                    "failed to decode broadcast tournament response: {err}; body={body}"
                ))
            })?;

        let tournament_id = parse_game_id(&tournament_payload.tour.id).ok_or_else(|| {
            LichessApiError::InvalidResponse(format!(
                "broadcast tournament response did not include a valid id: {}",
                tournament_payload.tour.id
            ))
        })?;

        let round_response = http
            .post(format!("{}/broadcast/{}/new", base_url, tournament_id))
            .header("Authorization", format!("Bearer {api_key}"))
            .form(&[
                ("name", broadcast_round_name_from_env()),
                ("syncSource", "push".to_string()),
            ])
            .send()
            .await
            .map_err(LichessApiError::Http)?;
        let status = round_response.status();
        let body = round_response.text().await.map_err(LichessApiError::Http)?;
        if !status.is_success() {
            return Err(scope_or_api_error(status, body, "study:write"));
        }
        let round_payload: LichessBroadcastRoundCreateResponse = serde_json::from_str(&body)
            .map_err(|err| {
                LichessApiError::InvalidResponse(format!(
                    "failed to decode broadcast round response: {err}; body={body}"
                ))
            })?;

        let round_id = parse_game_id(&round_payload.round.id).ok_or_else(|| {
            LichessApiError::InvalidResponse(format!(
                "broadcast round response did not include a valid id: {}",
                round_payload.round.id
            ))
        })?;

        Ok(LichessCreatedGame {
            game_id: round_id,
            share_url: round_payload.round.url,
        })
    }

    pub async fn push_broadcast_pgn(&self, pgn: &str) -> Result<(), LichessApiError> {
        let response = self
            .http
            .post(format!(
                "{}/api/broadcast/round/{}/push",
                self.base_url, self.game_id
            ))
            .header("Authorization", self.auth_header_value())
            .header("Content-Type", "text/plain")
            .body(pgn.to_string())
            .send()
            .await
            .map_err(LichessApiError::Http)?;
        let status = response.status();
        let body = response.text().await.map_err(LichessApiError::Http)?;
        if status.is_success() {
            Ok(())
        } else {
            Err(scope_or_api_error(status, body, "study:write"))
        }
    }

    pub async fn finish_broadcast_round(&self) -> Result<(), LichessApiError> {
        let round = self.fetch_broadcast_round().await?;

        let response = self
            .http
            .post(format!(
                "{}/broadcast/round/{}/edit",
                self.base_url, self.game_id
            ))
            .header("Authorization", self.auth_header_value())
            .form(&[
                ("name", round.round.name),
                ("status", "finished".to_string()),
            ])
            .send()
            .await
            .map_err(LichessApiError::Http)?;
        let status = response.status();
        let body = response.text().await.map_err(LichessApiError::Http)?;
        if status.is_success() {
            Ok(())
        } else {
            Err(scope_or_api_error(status, body, "study:write"))
        }
    }

    pub async fn send_broadcast_chat_message(&self, message: &str) -> Result<(), LichessApiError> {
        let response = self
            .http
            .post(format!(
                "{}/api/broadcast/round/{}/chat",
                self.base_url, self.game_id
            ))
            .header("Authorization", self.auth_header_value())
            .form(&[("text", message.to_string())])
            .send()
            .await
            .map_err(LichessApiError::Http)?;
        let status = response.status();
        let body = response.text().await.map_err(LichessApiError::Http)?;
        if status.is_success() {
            Ok(())
        } else {
            Err(scope_or_api_error(status, body, "study:write"))
        }
    }

    pub fn game_id(&self) -> &str {
        &self.game_id
    }

    pub fn game_url(&self) -> String {
        format!("{}/broadcast/-/-/{}", self.base_url, self.game_id)
    }

    pub fn mode(&self) -> LichessStreamMode {
        self.mode
    }

    pub fn reconnect_delay(&self) -> Duration {
        self.reconnect_delay
    }

    pub async fn stream_updates<F, Fut>(&self, mut on_update: F) -> Result<(), LichessApiError>
    where
        F: FnMut(LichessGameUpdate) -> Fut,
        Fut: Future<Output = ()>,
    {
        let url = self.stream_url();
        let response = self.get_stream_response(&url).await?;
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(LichessApiError::Http)?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_index) = buffer.find('\n') {
                let line = buffer[..newline_index].trim().to_string();
                buffer = buffer[newline_index + 1..].to_string();
                if line.is_empty() {
                    continue;
                }
                let value: JsonValue = serde_json::from_str(&line).map_err(|err| {
                    LichessApiError::InvalidResponse(format!(
                        "failed to parse Lichess NDJSON line: {err}; line={line}"
                    ))
                })?;
                if let Some(update) = parse_game_update(&value, self.mode) {
                    on_update(update).await;
                }
            }
        }

        if !buffer.trim().is_empty() {
            let value: JsonValue = serde_json::from_str(buffer.trim()).map_err(|err| {
                LichessApiError::InvalidResponse(format!(
                    "failed to parse final Lichess NDJSON line: {err}; line={}",
                    buffer.trim()
                ))
            })?;
            if let Some(update) = parse_game_update(&value, self.mode) {
                on_update(update).await;
            }
        }

        Ok(())
    }

    fn stream_url(&self) -> String {
        match self.mode {
            LichessStreamMode::Board => {
                format!("{}/api/board/game/stream/{}", self.base_url, self.game_id)
            }
            LichessStreamMode::Public => {
                format!("{}/api/board/game/stream/{}", self.base_url, self.game_id)
            }
        }
    }

    async fn get_stream_response(&self, url: &str) -> Result<reqwest::Response, LichessApiError> {
        let mut request = self.http.get(url);
        if self.mode == LichessStreamMode::Board {
            request = request.header("Authorization", self.auth_header_value());
        }
        let response = request.send().await.map_err(LichessApiError::Http)?;
        let status = response.status();
        if status.is_success() {
            Ok(response)
        } else {
            let body = response.text().await.map_err(LichessApiError::Http)?;
            Err(LichessApiError::Api { status, body })
        }
    }

    fn auth_header_value(&self) -> String {
        format!("Bearer {}", self.api_token.as_deref().unwrap_or_default())
    }

    async fn fetch_broadcast_round(
        &self,
    ) -> Result<LichessBroadcastRoundResponse, LichessApiError> {
        let response = self
            .http
            .get(format!(
                "{}/api/broadcast/-/-/{}",
                self.base_url, self.game_id
            ))
            .send()
            .await
            .map_err(LichessApiError::Http)?;
        let status = response.status();
        let body = response.text().await.map_err(LichessApiError::Http)?;
        if !status.is_success() {
            return Err(LichessApiError::Api { status, body });
        }

        serde_json::from_str(&body).map_err(|err| {
            LichessApiError::InvalidResponse(format!(
                "failed to decode broadcast round response: {err}; body={body}"
            ))
        })
    }
}

pub fn chess_from_uci_moves(moves: &str) -> Result<Chess, LichessApiError> {
    let mut position = Chess::default();
    for token in moves.split_whitespace() {
        let uci = token.parse::<UciMove>().map_err(|err| {
            LichessApiError::InvalidResponse(format!("invalid UCI move '{token}': {err}"))
        })?;
        let chess_move = uci.to_move(&position).map_err(|err| {
            LichessApiError::InvalidResponse(format!("illegal move '{token}' in stream: {err}"))
        })?;
        position.play_unchecked(&chess_move);
    }
    Ok(position)
}

pub fn chess_from_fen(fen_text: &str) -> Result<Chess, LichessApiError> {
    let fen = fen_text.parse::<Fen>().map_err(|err| {
        LichessApiError::InvalidResponse(format!("invalid FEN '{fen_text}': {err}"))
    })?;
    fen.into_position(CastlingMode::Standard).map_err(|err| {
        LichessApiError::InvalidResponse(format!(
            "invalid chess position from FEN '{fen_text}': {err}"
        ))
    })
}

fn parse_game_update(value: &JsonValue, mode: LichessStreamMode) -> Option<LichessGameUpdate> {
    let event_type = value.get("type").and_then(JsonValue::as_str).unwrap_or("");

    match event_type {
        "gameFull" => {
            let moves = value
                .pointer("/state/moves")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned);
            let status = value
                .pointer("/state/status")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned);
            Some(LichessGameUpdate {
                mode,
                moves,
                fen: None,
                status,
            })
        }
        "gameState" => Some(LichessGameUpdate {
            mode,
            moves: value
                .get("moves")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned),
            fen: None,
            status: value
                .get("status")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned),
        }),
        _ => {
            let fen = value
                .get("fen")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned);
            let moves = value
                .get("moves")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned);
            let status = value
                .pointer("/status/name")
                .and_then(JsonValue::as_str)
                .or_else(|| value.get("status").and_then(JsonValue::as_str))
                .map(ToOwned::to_owned);

            if fen.is_none() && moves.is_none() {
                None
            } else {
                Some(LichessGameUpdate {
                    mode,
                    moves,
                    fen,
                    status,
                })
            }
        }
    }
}

fn stream_mode_from_env(has_token: bool) -> LichessStreamMode {
    let configured = std::env::var("LICHESS_STREAM_MODE")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();

    match configured.as_str() {
        "board" => LichessStreamMode::Board,
        "public" => LichessStreamMode::Public,
        _ => {
            let _ = has_token;
            LichessStreamMode::Board
        }
    }
}

fn base_url_from_env() -> String {
    std::env::var("LICHESS_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_LICHESS_BASE_URL.to_string())
        .trim()
        .trim_end_matches('/')
        .to_string()
}

fn api_token_from_env() -> Option<String> {
    std::env::var("LICHESS_API_KEY")
        .or_else(|_| std::env::var("LICHESS_API_TOKEN"))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn reconnect_delay_from_env() -> Duration {
    std::env::var("LICHESS_RECONNECT_DELAY_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(DEFAULT_RECONNECT_DELAY_MS))
}

fn broadcast_visibility_from_env() -> String {
    let visibility = std::env::var("LICHESS_BROADCAST_VISIBILITY")
        .unwrap_or_else(|_| DEFAULT_BROADCAST_VISIBILITY.to_string())
        .to_ascii_lowercase();
    match visibility.as_str() {
        "public" | "unlisted" | "private" => visibility,
        _ => DEFAULT_BROADCAST_VISIBILITY.to_string(),
    }
}

fn broadcast_name_from_env() -> String {
    std::env::var("LICHESS_BROADCAST_NAME")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BROADCAST_NAME.to_string())
}

fn broadcast_round_name_from_env() -> String {
    std::env::var("LICHESS_BROADCAST_ROUND_NAME")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BROADCAST_ROUND_NAME.to_string())
}

fn parse_game_id(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    if let Ok(url) = reqwest::Url::parse(raw) {
        for segment in url.path_segments().into_iter().flatten() {
            if is_game_id_segment(segment) {
                return Some(segment.to_string());
            }
        }
    }

    for segment in raw.split(['/', '?', '#']) {
        let segment = segment.trim();
        if is_game_id_segment(segment) {
            return Some(segment.to_string());
        }
    }

    None
}

fn is_game_id_segment(segment: &str) -> bool {
    segment.len() == 8 && segment.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

#[derive(Debug, Deserialize)]
struct LichessBroadcastTournamentCreateResponse {
    tour: LichessBroadcastTour,
}

#[derive(Debug, Deserialize)]
struct LichessBroadcastRoundCreateResponse {
    round: LichessBroadcastRound,
}

#[derive(Debug, Deserialize)]
struct LichessBroadcastTour {
    id: String,
}

#[derive(Debug, Deserialize)]
struct LichessBroadcastRound {
    id: String,
    url: String,
}

#[derive(Debug, Deserialize)]
struct LichessBroadcastRoundResponse {
    round: LichessBroadcastRoundInfo,
}

#[derive(Debug, Deserialize)]
struct LichessBroadcastRoundInfo {
    name: String,
}

#[derive(Debug)]
pub enum LichessApiError {
    Http(reqwest::Error),
    Api { status: StatusCode, body: String },
    InvalidResponse(String),
    Unsupported(String),
}

impl LichessApiError {
    pub fn is_no_such_game(&self) -> bool {
        matches!(
            self,
            Self::Api { status, body }
            if *status == StatusCode::NOT_FOUND
                && body.to_ascii_lowercase().contains("no such game")
        )
    }
}

impl fmt::Display for LichessApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "{err}"),
            Self::Api { status, body } => write!(
                f,
                "Lichess API error ({status}): {}",
                truncate_for_error(body, 300)
            ),
            Self::InvalidResponse(message) => {
                write!(f, "Lichess API invalid response: {message}")
            }
            Self::Unsupported(message) => write!(f, "Lichess unsupported operation: {message}"),
        }
    }
}

impl Error for LichessApiError {}

fn truncate_for_error(value: &str, max_len: usize) -> String {
    if value.len() <= max_len {
        value.to_string()
    } else {
        format!("{}...", &value[..max_len])
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn scope_or_api_error(status: StatusCode, body: String, scope: &str) -> LichessApiError {
    if status == StatusCode::FORBIDDEN
        && body
            .to_ascii_lowercase()
            .contains(&format!("missing scope: {scope}"))
    {
        LichessApiError::Unsupported(format!(
            "Lichess token is missing `{scope}` scope. Create a new Personal API token at https://lichess.org/account/oauth/token with `{scope}` enabled."
        ))
    } else {
        LichessApiError::Api { status, body }
    }
}
