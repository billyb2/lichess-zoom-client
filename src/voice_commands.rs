use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::recall::TranscriptWebhookPayload;

const MAX_BUFFERED_TRANSCRIPTS: usize = 512;

#[derive(Debug, Clone)]
pub struct TranscriptChunk {
    pub sequence: u64,
    pub bot_id: Option<String>,
    pub participant_id: Option<i64>,
    pub participant_name: Option<String>,
    pub text: String,
    pub is_partial: bool,
    pub ingested_at: Instant,
}

static TRANSCRIPT_BUFFER: OnceLock<Mutex<VecDeque<TranscriptChunk>>> = OnceLock::new();
static NEXT_TRANSCRIPT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

pub fn ingest_transcript_chunk(payload: &TranscriptWebhookPayload, is_partial: bool) {
    let text = payload
        .data
        .words
        .iter()
        .map(|word| word.text.trim())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if text.is_empty() {
        return;
    }

    let participant = payload.data.participant.as_ref();
    let participant_name = participant.and_then(|value| value.name.clone());
    let participant_id = participant.map(|value| value.id);
    let bot_id = payload.bot.as_ref().map(|bot| bot.id.clone());

    let mut buffer = TRANSCRIPT_BUFFER
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .expect("voice transcript buffer mutex poisoned");

    let chunk = TranscriptChunk {
        sequence: NEXT_TRANSCRIPT_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        bot_id,
        participant_id,
        participant_name,
        text,
        is_partial,
        ingested_at: Instant::now(),
    };

    buffer.push_back(chunk);

    while buffer.len() > MAX_BUFFERED_TRANSCRIPTS {
        let _ = buffer.pop_front();
    }
}

pub fn drain_buffered_transcripts() -> Vec<TranscriptChunk> {
    let mut buffer = TRANSCRIPT_BUFFER
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .expect("voice transcript buffer mutex poisoned");
    buffer.drain(..).collect()
}

pub fn drain_finalized_transcripts() -> Vec<TranscriptChunk> {
    drain_buffered_transcripts()
        .into_iter()
        .filter(|chunk| !chunk.is_partial)
        .collect()
}
