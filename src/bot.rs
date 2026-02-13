use crate::recall::{ParticipantEventWebhookPayload, TranscriptWebhookPayload};
use crate::voice_commands;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotAction {
    None,
    LeaveAllManagedBots,
}

pub fn on_participant_join(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_leave(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_update(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_speech_on(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_speech_off(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_webcam_on(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_webcam_off(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_screenshare_on(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_screenshare_off(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let _ = payload;
    BotAction::None
}

pub fn on_participant_chat_message(payload: &ParticipantEventWebhookPayload) -> BotAction {
    let text = payload
        .data
        .data
        .as_ref()
        .and_then(|chat| chat.text.as_deref())
        .map(str::trim)
        .unwrap_or("");
    if text.eq_ignore_ascii_case("leave") {
        println!("Received leave command from {}", participant_label(payload));
        BotAction::LeaveAllManagedBots
    } else {
        BotAction::None
    }
}

pub fn on_transcript_data(payload: &TranscriptWebhookPayload) -> BotAction {
    voice_commands::ingest_transcript_chunk(payload, false);
    BotAction::None
}

pub fn on_transcript_partial_data(payload: &TranscriptWebhookPayload) -> BotAction {
    voice_commands::ingest_transcript_chunk(payload, true);
    BotAction::None
}

pub fn on_transcript_provider_data(payload: &serde_json::Value) -> BotAction {
    let _ = payload;
    BotAction::None
}

fn participant_label(payload: &ParticipantEventWebhookPayload) -> String {
    match payload.data.participant.name.as_deref() {
        Some(name) if !name.trim().is_empty() => name.to_string(),
        _ => format!("id={}", payload.data.participant.id),
    }
}
