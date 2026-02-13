// https://recallai.zoom.us/j/6240386581?pwd=AQuhOddMvrdo1iy7NABARNz6JuKoE7.1

mod bot;
mod lichess;
mod openai;
mod recall;
mod render;
mod voice_commands;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use futures_util::stream::{self, StreamExt};
use shakmaty::san::SanPlus;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color as ChessColor, Move, Position, Square};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::time::sleep;

use bot::BotAction;
use lichess::{LichessApiError, LichessClient};
use openai::OpenAiClient;
use recall::{
    Bot, BotRuntimeVariant, BotVariantConfig, CreateBotRequest, DEFAULT_WEBHOOK_URL,
    ParticipantEventWebhookPayload, RecallClient, RecallRealtimeWebhook,
    VoiceCommandTranscriptMode, parse_realtime_webhooks, voice_command_assembly_ai_provider,
    voice_command_deepgram_provider,
};

const DEFAULT_BIND_ADDR: &str = "0.0.0.0:7777";
const DEFAULT_STARTUP_BOT_COUNT: usize = 16;
const DEFAULT_LEAVE_DELAY_MS: u64 = 250;
const DEFAULT_CHAT_SEND_TO: &str = "everyone";
const DEFAULT_OUTPUT_VIDEO_PARALLELISM: usize = 8;
const CHAT_RERENDER_DEBOUNCE_MS: u64 = 200;
const METADATA_PROJECT_KEY: &str = "hackathon_project";
const METADATA_PROJECT_VALUE: &str = "chess_wall";
const METADATA_RUN_ID_KEY: &str = "hackathon_run_id";
const METADATA_SLOT_KEY: &str = "tile_slot";
const METADATA_SQUARES_KEY: &str = "tile_squares";
const CHAT_BUBBLE_TTL: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
struct BotTileAssignment {
    participant_slot: usize,
    controlled_squares: [String; 4],
}

#[derive(Clone, Debug)]
struct ManagedBot {
    bot: Bot,
    assignment: BotTileAssignment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlayerColor {
    White,
    Black,
}

impl PlayerColor {
    fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::White => "White",
            Self::Black => "Black",
        }
    }

    fn to_chess_color(self) -> ChessColor {
        match self {
            Self::White => ChessColor::White,
            Self::Black => ChessColor::Black,
        }
    }
}

#[derive(Clone, Debug)]
struct PlayerSeat {
    participant_id: i64,
    participant_name: String,
}

#[derive(Clone, Debug, Default)]
struct PlayerSeats {
    white: Option<PlayerSeat>,
    black: Option<PlayerSeat>,
}

#[derive(Clone, Debug)]
struct MeetingParticipant {
    id: i64,
    name: String,
    join_order: u64,
}

#[derive(Clone, Debug)]
struct PlayerChatBubble {
    text: String,
    message_id: u64,
    expires_at: Instant,
}

#[derive(Clone, Debug, Default)]
struct PlayerChatBubbles {
    white: Option<PlayerChatBubble>,
    black: Option<PlayerChatBubble>,
}

#[derive(Clone, Debug)]
enum VoiceCommandTranscriptProvider {
    AssemblyAi {
        speech_model: String,
        format_turns: bool,
        keyterms_prompt: Vec<String>,
    },
    Deepgram {
        model: String,
        language: String,
    },
    RecallAi {
        mode: VoiceCommandTranscriptMode,
    },
}

impl VoiceCommandTranscriptProvider {
    fn description(&self) -> String {
        match self {
            Self::AssemblyAi {
                speech_model,
                format_turns,
                keyterms_prompt,
            } => {
                format!(
                    "assembly_ai_v3_streaming (speech_model={speech_model}, format_turns={format_turns}, keyterms={})",
                    keyterms_prompt.len()
                )
            }
            Self::Deepgram { model, language } => {
                format!("deepgram_streaming (model={model}, language={language})")
            }
            Self::RecallAi { mode } => format!("recallai_streaming ({})", mode.as_str()),
        }
    }
}

#[derive(Clone)]
struct AppState {
    recall_client: Option<RecallClient>,
    lichess_client: Option<LichessClient>,
    openai_client: Option<OpenAiClient>,
    webhook_url: String,
    chess_position: Arc<Mutex<Chess>>,
    move_history: Arc<Mutex<Vec<Move>>>,
    meeting_participants: Arc<Mutex<BTreeMap<i64, MeetingParticipant>>>,
    active_meeting_url: Arc<Mutex<Option<String>>>,
    player_seats: Arc<Mutex<PlayerSeats>>,
    player_chat_bubbles: Arc<Mutex<PlayerChatBubbles>>,
    next_join_order: Arc<AtomicU64>,
    lichess_chat_supported: Arc<AtomicBool>,
    managed_bots: Arc<Mutex<Vec<ManagedBot>>>,
    transcript_dispatch_lock: Arc<Mutex<()>>,
    transcript_apply_lock: Arc<Mutex<()>>,
    pending_transcript_results: Arc<Mutex<BTreeMap<u64, TranscriptGptResult>>>,
    next_transcript_dispatch_order: Arc<AtomicU64>,
    next_transcript_apply_order: Arc<AtomicU64>,
    launch_lock: Arc<Mutex<()>>,
    chat_rerender_generation: Arc<AtomicU64>,
    render_lock: Arc<Mutex<()>>,
    leave_requested: Arc<AtomicBool>,
    chat_send_to: String,
    run_id: String,
}

#[derive(Clone, Debug)]
struct TranscriptGptResult {
    order_id: u64,
    sequence: u64,
    participant_id: Option<i64>,
    speaker_label: String,
    transcript_text: String,
    classification: Result<openai::VoiceCommandClassification, String>,
    timing: TranscriptPipelineTiming,
}

#[derive(Clone, Debug)]
struct TranscriptPipelineTiming {
    ingested_at: Instant,
    dispatch_started_at: Instant,
    gpt_started_at: Option<Instant>,
    gpt_finished_at: Option<Instant>,
    result_ready_at: Instant,
}

#[derive(Clone, Copy, Debug)]
enum CommandExecutionOutcome {
    Applied,
    InvalidUciFormat,
    IllegalInCurrentPosition,
    NotPlayersTurn,
    SpeakerNotSeated,
}

impl CommandExecutionOutcome {
    fn as_label(self) -> &'static str {
        match self {
            Self::Applied => "applied",
            Self::InvalidUciFormat => "invalid_uci_format",
            Self::IllegalInCurrentPosition => "illegal_in_current_position",
            Self::NotPlayersTurn => "not_players_turn",
            Self::SpeakerNotSeated => "speaker_not_seated",
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = dotenvy::dotenv();

    let bind_addr =
        env::var("RECALL_WEBHOOK_BIND").unwrap_or_else(|_| DEFAULT_BIND_ADDR.to_string());
    let webhook_url =
        env::var("RECALL_WEBHOOK_URL").unwrap_or_else(|_| DEFAULT_WEBHOOK_URL.to_string());

    println!("Recall webhook listener starting on {bind_addr}");
    println!("Recall webhook target configured as {webhook_url}");
    let chat_send_to = recall_chat_send_to_from_env();
    println!("Recall chat target configured as {chat_send_to}");
    let run_id = build_run_id();
    println!("Runtime run id: {run_id}");
    let lichess_enabled = env_flag("LICHESS_ENABLED", false);
    let lichess_client = lichess_client_from_env().await;
    if lichess_enabled && lichess_client.is_none() {
        return Err("Lichess is enabled but broadcast relay initialization failed".into());
    }

    let state = Arc::new(AppState {
        recall_client: recall_client_from_env(),
        lichess_client,
        openai_client: openai_client_from_env(),
        webhook_url: webhook_url.clone(),
        chess_position: Arc::new(Mutex::new(Chess::default())),
        move_history: Arc::new(Mutex::new(Vec::new())),
        meeting_participants: Arc::new(Mutex::new(BTreeMap::new())),
        active_meeting_url: Arc::new(Mutex::new(None)),
        player_seats: Arc::new(Mutex::new(PlayerSeats::default())),
        player_chat_bubbles: Arc::new(Mutex::new(PlayerChatBubbles::default())),
        next_join_order: Arc::new(AtomicU64::new(1)),
        lichess_chat_supported: Arc::new(AtomicBool::new(true)),
        managed_bots: Arc::new(Mutex::new(Vec::new())),
        transcript_dispatch_lock: Arc::new(Mutex::new(())),
        transcript_apply_lock: Arc::new(Mutex::new(())),
        pending_transcript_results: Arc::new(Mutex::new(BTreeMap::new())),
        next_transcript_dispatch_order: Arc::new(AtomicU64::new(1)),
        next_transcript_apply_order: Arc::new(AtomicU64::new(1)),
        launch_lock: Arc::new(Mutex::new(())),
        chat_rerender_generation: Arc::new(AtomicU64::new(0)),
        render_lock: Arc::new(Mutex::new(())),
        leave_requested: Arc::new(AtomicBool::new(false)),
        chat_send_to,
        run_id,
    });

    if let Some(raw_meeting_url) = recall_meeting_url_from_env() {
        let meeting_url = normalize_zoom_meeting_url(&raw_meeting_url)
            .map_err(|message| format!("Invalid RECALL_MEETING_URL: {message}"))?;
        println!("Startup launch requested via RECALL_MEETING_URL: {meeting_url}");
        match relaunch_bots_for_meeting(Arc::clone(&state), meeting_url.clone()).await {
            Ok(bot_count) => {
                println!("Startup launch completed: {bot_count} bot(s) running for {meeting_url}")
            }
            Err(message) => return Err(format!("Startup launch failed: {message}").into()),
        }
    } else {
        println!("Startup bot launch disabled: set RECALL_MEETING_URL to auto-launch bots");
    }

    run_webhook_server(&bind_addr, state).await
}

fn recall_client_from_env() -> Option<RecallClient> {
    let api_key = match env::var("RECALL_API_KEY") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            println!("Recall API disabled: set RECALL_API_KEY to enable bot startup + commands");
            return None;
        }
    };

    let region = env::var("RECALL_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    Some(RecallClient::new(api_key, region))
}

async fn lichess_client_from_env() -> Option<LichessClient> {
    if !env_flag("LICHESS_ENABLED", false) {
        println!("Lichess support disabled: set LICHESS_ENABLED=true to enable");
        return None;
    }

    let client = match LichessClient::from_env() {
        Some(client) => client,
        None => {
            let created = match LichessClient::create_broadcast_round_from_env().await {
                Ok(created) => created,
                Err(err) => {
                    eprintln!(
                        "Lichess support enabled, but no valid round id was set and broadcast creation failed: {err}"
                    );
                    eprintln!("Required Lichess scope for Broadcast relay is: study:write");
                    return None;
                }
            };

            println!("Created Lichess broadcast round: {}", created.share_url);
            println!(
                "Send this link to friends (or yourself): {}",
                created.share_url
            );

            match LichessClient::from_env_for_game_id(&created.game_id) {
                Some(client) => client,
                None => {
                    eprintln!(
                        "Lichess broadcast creation succeeded but returned an invalid round id: {}",
                        created.game_id
                    );
                    return None;
                }
            }
        }
    };

    println!(
        "Lichess broadcast relay enabled for round {} (mode: {})",
        client.game_id(),
        client.mode().as_str()
    );
    println!("Lichess broadcast URL: {}", client.game_url());
    Some(client)
}

fn openai_client_from_env() -> Option<OpenAiClient> {
    let client = OpenAiClient::from_env();
    match &client {
        Some(client) => {
            println!(
                "OpenAI transcript interpreter enabled with model {} (reasoning effort: {})",
                client.model(),
                client.reasoning_effort()
            );
        }
        None => {
            println!(
                "OpenAI transcript interpreter disabled: set OPENAI_API_KEY to enable transcript -> GPT wiring"
            );
        }
    }
    client
}

fn recall_chat_send_to_from_env() -> String {
    env::var("RECALL_CHAT_SEND_TO")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_CHAT_SEND_TO.to_string())
}

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn output_video_parallelism_from_env() -> usize {
    env::var("RECALL_OUTPUT_PARALLELISM")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_OUTPUT_VIDEO_PARALLELISM)
}

fn startup_bot_count_from_env() -> usize {
    env::var("RECALL_BOT_COUNT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_STARTUP_BOT_COUNT)
}

fn zoom_runtime_variant_from_env() -> BotRuntimeVariant {
    let variant = env::var("RECALL_ZOOM_VARIANT")
        .unwrap_or_else(|_| "regular".to_string())
        .trim()
        .to_ascii_lowercase();
    match variant.as_str() {
        "regular" | "web" | "default" => BotRuntimeVariant::Web,
        "web_4_core" | "4core" | "web4core" => BotRuntimeVariant::Web4Core,
        "web_gpu" | "gpu" | "webgpu" => BotRuntimeVariant::WebGpu,
        _ => {
            eprintln!(
                "Unknown RECALL_ZOOM_VARIANT '{}'; defaulting to regular web",
                variant
            );
            BotRuntimeVariant::Web
        }
    }
}

fn zoom_runtime_variant_label(variant: &BotRuntimeVariant) -> &'static str {
    match variant {
        BotRuntimeVariant::Web => "web",
        BotRuntimeVariant::Web4Core => "web_4_core",
        BotRuntimeVariant::WebGpu => "web_gpu",
    }
}

fn recall_meeting_url_from_env() -> Option<String> {
    env::var("RECALL_MEETING_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

async fn create_startup_bots(
    state: Arc<AppState>,
    meeting_url: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = match &state.recall_client {
        Some(client) => client.clone(),
        None => {
            println!("Skipping startup bot creation: Recall client is not configured");
            return Ok(());
        }
    };

    let meeting_url = meeting_url.trim();
    if meeting_url.is_empty() {
        return Err("meeting URL must not be empty".into());
    }
    let webhook_url = state.webhook_url.clone();
    let transcript_provider = voice_command_transcript_provider_from_env();
    let zoom_variant = zoom_runtime_variant_from_env();
    let zoom_variant_label = zoom_runtime_variant_label(&zoom_variant);

    let startup_bot_count = startup_bot_count_from_env();
    let bot_names = bot_names_from_count(startup_bot_count);
    let bot_count = bot_names.len();
    println!(
        "Creating {} startup bot(s) for meeting {} with webhook {} (variant={}, bot 01 handles transcripts via {})",
        bot_count,
        meeting_url,
        webhook_url,
        zoom_variant_label,
        transcript_provider.description()
    );
    let startup_tile_jpegs = {
        let position = state.chess_position.lock().await;
        render::render_current_board_wall_tile_jpegs(
            position.board(),
            bot_count,
            None,
            None,
            None,
            None,
        )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { Box::new(err) })?
    };
    let mut created_count = 0_usize;
    let mut first_create_error: Option<String> = None;

    for (index, bot_name) in bot_names.into_iter().enumerate() {
        if state.leave_requested.load(Ordering::SeqCst) {
            println!("Stopping startup bot creation: leave command was requested");
            break;
        }

        let board_tile_jpeg = match startup_tile_jpegs.get(index) {
            Some(jpeg) => jpeg,
            None => {
                return Err(format!(
                    "Startup tile render did not include tile index {} (got {})",
                    index,
                    startup_tile_jpegs.len()
                )
                .into());
            }
        };
        let assignment = BotTileAssignment {
            participant_slot: index,
            controlled_squares: render::wall_slot_labels(index, bot_count),
        };

        let mut request = CreateBotRequest {
            meeting_url: meeting_url.to_string(),
            bot_name,
            recording_config: None,
            ..Default::default()
        };
        request.variant = Some(BotVariantConfig {
            zoom: Some(zoom_variant.clone()),
            ..Default::default()
        });
        request.metadata = Some(build_bot_metadata(&state.run_id, &assignment, meeting_url));
        if index == 0 {
            request.configure_transcript_without_video_recording();
            match &transcript_provider {
                VoiceCommandTranscriptProvider::AssemblyAi {
                    speech_model,
                    format_turns,
                    keyterms_prompt,
                } => {
                    request.configure_voice_command_transcript_webhooks_with_provider(
                        &webhook_url,
                        voice_command_assembly_ai_provider(
                            speech_model,
                            *format_turns,
                            keyterms_prompt,
                        ),
                    );
                }
                VoiceCommandTranscriptProvider::Deepgram { model, language } => {
                    request.configure_voice_command_transcript_webhooks_with_provider(
                        &webhook_url,
                        voice_command_deepgram_provider(model, language),
                    );
                }
                VoiceCommandTranscriptProvider::RecallAi { mode } => {
                    request
                        .configure_voice_command_transcript_webhooks_with_mode(&webhook_url, *mode);
                }
            }
        }
        request.configure_automatic_video_output_jpeg(board_tile_jpeg);
        println!(
            "Attempting startup bot '{:02}' (slot {}): variant={} transcripts={} squares={}",
            index + 1,
            assignment.participant_slot + 1,
            zoom_variant_label,
            if index == 0 { "enabled" } else { "disabled" },
            assignment.controlled_squares.join(", ")
        );

        match client.create_bot(&request).await {
            Ok(bot) => {
                if state.leave_requested.load(Ordering::SeqCst) {
                    println!(
                        "Bot '{}' created after leave command; removing from call immediately",
                        bot.name
                    );
                    if let Err(err) = client.leave_call(bot).await {
                        eprintln!(
                            "Failed immediate leave for post-command startup bot '{}': {err}",
                            request.bot_name
                        );
                    }
                    continue;
                }
                println!(
                    "Created startup bot '{}' with id {} at participant slot {} controlling {}",
                    bot.name,
                    bot.id,
                    assignment.participant_slot + 1,
                    assignment.controlled_squares.join(", ")
                );
                let mut managed_bots = state.managed_bots.lock().await;
                managed_bots.push(ManagedBot { bot, assignment });
                created_count += 1;
            }
            Err(err) => {
                if first_create_error.is_none() {
                    first_create_error = Some(err.to_string());
                }
                eprintln!("Failed to create startup bot '{}': {err}", request.bot_name);
            }
        }
    }

    if created_count == 0 {
        if let Some(err) = first_create_error {
            return Err(std::io::Error::other(err).into());
        }
    }

    Ok(())
}

fn bot_names_from_count(count: usize) -> Vec<String> {
    (1..=count).map(|index| format!("{index:02}")).collect()
}

fn voice_command_transcript_provider_from_env() -> VoiceCommandTranscriptProvider {
    let provider = env::var("RECALL_TRANSCRIPT_PROVIDER")
        .unwrap_or_else(|_| "assembly_ai_v3_streaming".to_string())
        .to_ascii_lowercase();
    match provider.as_str() {
        "assembly_ai_v3_streaming" | "assemblyai" | "assembly_ai" => {
            let speech_model = env::var("RECALL_ASSEMBLY_SPEECH_MODEL")
                .unwrap_or_else(|_| "universal-streaming-english".to_string());
            let format_turns = env::var("RECALL_ASSEMBLY_FORMAT_TURNS")
                .ok()
                .map(|value| value.trim().eq_ignore_ascii_case("true"))
                .unwrap_or(true);
            let keyterms_prompt = assembly_ai_keyterms_prompt_from_env();
            VoiceCommandTranscriptProvider::AssemblyAi {
                speech_model,
                format_turns,
                keyterms_prompt,
            }
        }
        "deepgram_streaming" | "deepgram" => {
            let model = env::var("RECALL_DEEPGRAM_MODEL").unwrap_or_else(|_| "nova-3".to_string());
            let language =
                env::var("RECALL_DEEPGRAM_LANGUAGE").unwrap_or_else(|_| "en".to_string());
            VoiceCommandTranscriptProvider::Deepgram { model, language }
        }
        "recallai_streaming" | "recallai" => VoiceCommandTranscriptProvider::RecallAi {
            mode: voice_command_transcript_mode_from_env(),
        },
        _ => {
            eprintln!(
                "Unknown RECALL_TRANSCRIPT_PROVIDER '{}'; defaulting to assembly_ai_v3_streaming",
                provider
            );
            VoiceCommandTranscriptProvider::AssemblyAi {
                speech_model: "universal-streaming-english".to_string(),
                format_turns: true,
                keyterms_prompt: assembly_ai_keyterms_prompt_from_env(),
            }
        }
    }
}

fn assembly_ai_keyterms_prompt_from_env() -> Vec<String> {
    let mut keyterms = default_chess_assembly_keyterms();
    if let Ok(raw_extra) = env::var("RECALL_ASSEMBLY_KEYTERMS") {
        for term in raw_extra.split(',') {
            let term = term.trim();
            if !term.is_empty() {
                keyterms.push(term.to_string());
            }
        }
    }
    dedupe_case_insensitive(keyterms)
}

fn default_chess_assembly_keyterms() -> Vec<String> {
    let mut terms = vec![
        "check".to_string(),
        "checkmate".to_string(),
        "stalemate".to_string(),
        "draw".to_string(),
        "resign".to_string(),
        "castle".to_string(),
        "castles".to_string(),
        "castling".to_string(),
        "kingside".to_string(),
        "queenside".to_string(),
        "en passant".to_string(),
        "promotion".to_string(),
        "promote".to_string(),
        "underpromotion".to_string(),
        "fork".to_string(),
        "pin".to_string(),
        "skewer".to_string(),
        "discovered attack".to_string(),
        "double check".to_string(),
        "zugzwang".to_string(),
        "sacrifice".to_string(),
        "gambit".to_string(),
        "fianchetto".to_string(),
        "pawn".to_string(),
        "knight".to_string(),
        "bishop".to_string(),
        "rook".to_string(),
        "queen".to_string(),
        "king".to_string(),
    ];

    for file in b'a'..=b'h' {
        for rank in 1..=8 {
            terms.push(format!("{}{}", char::from(file), rank));
        }
    }

    terms
}

fn dedupe_case_insensitive(terms: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut deduped = Vec::new();
    for term in terms {
        let normalized = term.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            continue;
        }
        if seen.insert(normalized) {
            deduped.push(term);
        }
    }
    deduped
}

fn voice_command_transcript_mode_from_env() -> VoiceCommandTranscriptMode {
    let mode = env::var("RECALL_TRANSCRIPT_MODE")
        .unwrap_or_else(|_| "prioritize_low_latency".to_string())
        .to_ascii_lowercase();
    match mode.as_str() {
        "prioritize_accuracy" | "accuracy" => VoiceCommandTranscriptMode::PrioritizeAccuracy,
        "prioritize_low_latency" | "low_latency" => {
            VoiceCommandTranscriptMode::PrioritizeLowLatency
        }
        _ => {
            eprintln!(
                "Unknown RECALL_TRANSCRIPT_MODE '{}'; defaulting to prioritize_low_latency",
                mode
            );
            VoiceCommandTranscriptMode::PrioritizeLowLatency
        }
    }
}

fn build_bot_metadata(
    run_id: &str,
    assignment: &BotTileAssignment,
    meeting_url: &str,
) -> BTreeMap<String, Option<String>> {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        METADATA_PROJECT_KEY.to_string(),
        Some(METADATA_PROJECT_VALUE.to_string()),
    );
    metadata.insert(METADATA_RUN_ID_KEY.to_string(), Some(run_id.to_string()));
    metadata.insert(
        METADATA_SLOT_KEY.to_string(),
        Some((assignment.participant_slot + 1).to_string()),
    );
    metadata.insert(
        METADATA_SQUARES_KEY.to_string(),
        Some(assignment.controlled_squares.join(",")),
    );
    metadata.insert(
        "meeting_url".to_string(),
        Some(meeting_url.trim().to_string()),
    );
    metadata
}

fn normalize_zoom_meeting_url(raw: &str) -> Result<String, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("Meeting URL is required.".to_string());
    }

    let mut parsed =
        reqwest::Url::parse(raw).map_err(|_| "Meeting URL must be a valid URL.".to_string())?;
    let host = parsed
        .host_str()
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| "Meeting URL must include a host.".to_string())?;
    if !host.ends_with("zoom.us") {
        return Err("Meeting URL must point to zoom.us.".to_string());
    }
    if !matches!(parsed.scheme(), "https" | "http") {
        return Err("Meeting URL must start with http:// or https://.".to_string());
    }
    let path = parsed.path();
    if !(path.contains("/j/") || path.contains("/wc/") || path.contains("/s/")) {
        return Err(
            "Meeting URL must be a Zoom join link (for example /j/<meeting_id>).".to_string(),
        );
    }
    parsed.set_fragment(None);
    Ok(parsed.to_string())
}

async fn relaunch_bots_for_meeting(
    state: Arc<AppState>,
    meeting_url: String,
) -> Result<usize, String> {
    let _launch_guard = state.launch_lock.lock().await;

    leave_managed_bots_for_relaunch(Arc::clone(&state)).await?;
    reset_game_session_for_relaunch(Arc::clone(&state), &meeting_url).await;
    state.leave_requested.store(false, Ordering::SeqCst);

    create_startup_bots(Arc::clone(&state), &meeting_url)
        .await
        .map_err(|err| err.to_string())?;

    let bot_count = {
        let managed_bots = state.managed_bots.lock().await;
        managed_bots.len()
    };
    if bot_count == 0 {
        return Err(
            "No startup bots were created. Check Recall credentials and meeting URL.".to_string(),
        );
    }

    Ok(bot_count)
}

async fn reset_game_session_for_relaunch(state: Arc<AppState>, meeting_url: &str) {
    {
        let mut position = state.chess_position.lock().await;
        *position = Chess::default();
    }
    {
        let mut move_history = state.move_history.lock().await;
        move_history.clear();
    }
    {
        let mut participants = state.meeting_participants.lock().await;
        participants.clear();
    }
    {
        let mut seats = state.player_seats.lock().await;
        *seats = PlayerSeats::default();
    }
    {
        let mut bubbles = state.player_chat_bubbles.lock().await;
        *bubbles = PlayerChatBubbles::default();
    }
    {
        let mut pending = state.pending_transcript_results.lock().await;
        pending.clear();
    }
    {
        let mut active_meeting_url = state.active_meeting_url.lock().await;
        *active_meeting_url = Some(meeting_url.to_string());
    }
    state.next_join_order.store(1, Ordering::Relaxed);
    state
        .next_transcript_dispatch_order
        .store(1, Ordering::Relaxed);
    state
        .next_transcript_apply_order
        .store(1, Ordering::Relaxed);
    state.lichess_chat_supported.store(true, Ordering::Relaxed);
    state
        .chat_rerender_generation
        .fetch_add(1, Ordering::Relaxed);
}

async fn run_webhook_server(bind_addr: &str, state: Arc<AppState>) -> Result<(), Box<dyn Error>> {
    let shutdown_state = Arc::clone(&state);
    let app = Router::new()
        .route("/", post(handle_webhook))
        .route("/recall/webhook", post(handle_webhook))
        .with_state(state);

    let listener = TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_state))
        .await?;
    Ok(())
}

async fn shutdown_signal(state: Arc<AppState>) {
    let ctrl_c = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            eprintln!("Failed to listen for Ctrl+C: {err}");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                sigterm.recv().await;
            }
            Err(err) => {
                eprintln!("Failed to install SIGTERM handler: {err}");
            }
        }
    };

    #[cfg(unix)]
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    #[cfg(not(unix))]
    ctrl_c.await;

    println!("Shutdown signal received. Leaving managed bots before exit...");
    leave_all_managed_bots(Arc::clone(&state)).await;
    finalize_lichess_broadcast_on_shutdown(state).await;
}

async fn finalize_lichess_broadcast_on_shutdown(state: Arc<AppState>) {
    let lichess_client = match &state.lichess_client {
        Some(client) => client.clone(),
        None => return,
    };

    let (position, move_history) = {
        let position = state.chess_position.lock().await.clone();
        let move_history = state.move_history.lock().await.clone();
        (position, move_history)
    };
    if move_history.is_empty() {
        println!(
            "Skipping Lichess broadcast finalization: no moves were played in this game session"
        );
        return;
    }

    let (white_name, black_name) = current_player_names(Arc::clone(&state)).await;
    let final_result = relay_shutdown_result(&position);
    let final_pgn = build_relay_pgn_with_result(
        &move_history,
        final_result,
        white_name.as_deref(),
        black_name.as_deref(),
    );

    match lichess_client.push_broadcast_pgn(&final_pgn).await {
        Ok(()) => {
            println!(
                "Pushed final PGN to Lichess broadcast round {} with result {}",
                lichess_client.game_id(),
                final_result
            );
        }
        Err(err) => {
            eprintln!(
                "Failed to push final PGN to Lichess broadcast round {}: {}",
                lichess_client.game_id(),
                err
            );
        }
    }

    match lichess_client.finish_broadcast_round().await {
        Ok(()) => {
            println!(
                "Marked Lichess broadcast round {} as finished",
                lichess_client.game_id()
            );
        }
        Err(err) => {
            eprintln!(
                "Failed to mark Lichess broadcast round {} as finished: {}",
                lichess_client.game_id(),
                err
            );
        }
    }
}

async fn handle_webhook(
    State(state): State<Arc<AppState>>,
    _uri: OriginalUri,
    _headers: HeaderMap,
    body: Bytes,
) -> StatusCode {
    if body.is_empty() {
        return StatusCode::OK;
    }

    match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(value) => {
            for parsed in parse_realtime_webhooks(&value) {
                match parsed {
                    Ok(event) => {
                        apply_participant_event_updates(Arc::clone(&state), &event).await;
                        let action = process_webhook_event(&event);
                        apply_bot_action(Arc::clone(&state), action).await;
                    }
                    Err(err) => {
                        eprintln!("Failed to parse Recall webhook: {err}");
                    }
                }
            }

            maybe_dispatch_transcripts_to_openai(Arc::clone(&state));
        }
        Err(_) => {
            eprintln!("Received non-JSON Recall webhook payload; ignoring");
        }
    }

    StatusCode::OK
}

fn process_webhook_event(event: &RecallRealtimeWebhook) -> BotAction {
    match event {
        RecallRealtimeWebhook::ParticipantJoin(payload) => bot::on_participant_join(payload),
        RecallRealtimeWebhook::ParticipantLeave(payload) => bot::on_participant_leave(payload),
        RecallRealtimeWebhook::ParticipantUpdate(payload) => bot::on_participant_update(payload),
        RecallRealtimeWebhook::ParticipantSpeechOn(payload) => {
            bot::on_participant_speech_on(payload)
        }
        RecallRealtimeWebhook::ParticipantSpeechOff(payload) => {
            bot::on_participant_speech_off(payload)
        }
        RecallRealtimeWebhook::ParticipantWebcamOn(payload) => {
            bot::on_participant_webcam_on(payload)
        }
        RecallRealtimeWebhook::ParticipantWebcamOff(payload) => {
            bot::on_participant_webcam_off(payload)
        }
        RecallRealtimeWebhook::ParticipantScreenshareOn(payload) => {
            bot::on_participant_screenshare_on(payload)
        }
        RecallRealtimeWebhook::ParticipantScreenshareOff(payload) => {
            bot::on_participant_screenshare_off(payload)
        }
        RecallRealtimeWebhook::ParticipantChatMessage(payload) => {
            bot::on_participant_chat_message(payload)
        }
        RecallRealtimeWebhook::TranscriptData(payload) => bot::on_transcript_data(payload),
        RecallRealtimeWebhook::TranscriptPartialData(payload) => {
            bot::on_transcript_partial_data(payload)
        }
        RecallRealtimeWebhook::TranscriptProviderData(payload) => {
            bot::on_transcript_provider_data(payload)
        }
    }
}

async fn apply_participant_event_updates(state: Arc<AppState>, event: &RecallRealtimeWebhook) {
    match event {
        RecallRealtimeWebhook::ParticipantJoin(payload) => {
            on_human_participant_join_or_update(state, payload, true).await;
        }
        RecallRealtimeWebhook::ParticipantUpdate(payload) => {
            on_human_participant_join_or_update(state, payload, false).await;
        }
        RecallRealtimeWebhook::ParticipantLeave(payload) => {
            on_human_participant_leave(state, payload).await;
        }
        _ => {}
    }
}

async fn on_human_participant_join_or_update(
    state: Arc<AppState>,
    payload: &ParticipantEventWebhookPayload,
    is_join: bool,
) {
    let participant_id = payload.data.participant.id;
    let participant_name = participant_display_name(payload);

    if is_managed_bot_name(&state, &participant_name).await {
        return;
    }

    {
        let mut participants = state.meeting_participants.lock().await;
        let join_order_seed = state.next_join_order.fetch_add(1, Ordering::Relaxed);
        let entry = participants
            .entry(participant_id)
            .or_insert(MeetingParticipant {
                id: participant_id,
                name: participant_name.clone(),
                join_order: join_order_seed,
            });
        entry.name = participant_name.clone();
    }

    let mut announcement = None::<String>;
    let mut should_rerender_labels = false;
    {
        let mut seats = state.player_seats.lock().await;

        if let Some(white) = seats.white.as_mut()
            && white.participant_id == participant_id
            && white.participant_name != participant_name
        {
            white.participant_name = participant_name.clone();
            should_rerender_labels = true;
        }
        if let Some(black) = seats.black.as_mut()
            && black.participant_id == participant_id
            && black.participant_name != participant_name
        {
            black.participant_name = participant_name.clone();
            should_rerender_labels = true;
        }

        if is_join
            && seat_color_for_participant(&seats, participant_id).is_none()
            && seats.white.is_some()
            && seats.black.is_none()
        {
            seats.black = Some(PlayerSeat {
                participant_id,
                participant_name: participant_name.clone(),
            });
            should_rerender_labels = true;
            announcement = Some(format!(
                "{} joined and was assigned Black.",
                participant_name
            ));
        } else if is_join
            && seat_color_for_participant(&seats, participant_id).is_none()
            && seats.black.is_some()
            && seats.white.is_none()
        {
            seats.white = Some(PlayerSeat {
                participant_id,
                participant_name: participant_name.clone(),
            });
            should_rerender_labels = true;
            announcement = Some(format!(
                "{} joined and was assigned White.",
                participant_name
            ));
        }
    }

    if should_rerender_labels {
        rerender_king_label_tiles(Arc::clone(&state)).await;
    }
    if let Some(message) = announcement {
        send_chat_message_from_bot01(state, "engine".to_string(), message).await;
    }
}

async fn on_human_participant_leave(
    state: Arc<AppState>,
    payload: &ParticipantEventWebhookPayload,
) {
    let participant_id = payload.data.participant.id;
    let participant_name = participant_display_name(payload);

    if is_managed_bot_name(&state, &participant_name).await {
        return;
    }

    {
        let mut participants = state.meeting_participants.lock().await;
        participants.remove(&participant_id);
    }

    let mut removed_color = None::<PlayerColor>;
    {
        let mut seats = state.player_seats.lock().await;
        if seats
            .white
            .as_ref()
            .is_some_and(|seat| seat.participant_id == participant_id)
        {
            seats.white = None;
            removed_color = Some(PlayerColor::White);
        } else if seats
            .black
            .as_ref()
            .is_some_and(|seat| seat.participant_id == participant_id)
        {
            seats.black = None;
            removed_color = Some(PlayerColor::Black);
        }
    }

    if let Some(color) = removed_color {
        clear_player_chat_bubble_for_color(Arc::clone(&state), color).await;
        rerender_king_label_tiles(Arc::clone(&state)).await;
        send_chat_message_from_bot01(
            state,
            "engine".to_string(),
            format!(
                "{} left. {} seat is now open; next human joiner gets this color.",
                participant_name,
                color.as_str()
            ),
        )
        .await;
    }
}

fn participant_display_name(payload: &ParticipantEventWebhookPayload) -> String {
    payload
        .data
        .participant
        .name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("unknown-participant")
        .to_string()
}

async fn is_managed_bot_name(state: &AppState, participant_name: &str) -> bool {
    let participant_name = participant_name.trim();
    if participant_name.is_empty() {
        return false;
    }
    if participant_name.len() == 2 && participant_name.chars().all(|ch| ch.is_ascii_digit()) {
        return true;
    }
    let managed_bots = state.managed_bots.lock().await;
    managed_bots
        .iter()
        .any(|managed_bot| managed_bot.bot.name.eq_ignore_ascii_case(participant_name))
}

fn seat_color_for_participant(seats: &PlayerSeats, participant_id: i64) -> Option<PlayerColor> {
    if seats
        .white
        .as_ref()
        .is_some_and(|seat| seat.participant_id == participant_id)
    {
        Some(PlayerColor::White)
    } else if seats
        .black
        .as_ref()
        .is_some_and(|seat| seat.participant_id == participant_id)
    {
        Some(PlayerColor::Black)
    } else {
        None
    }
}

async fn gpt_precheck_skip_reason(
    state: Arc<AppState>,
    participant_id: i64,
) -> Option<&'static str> {
    let speaker_color = assigned_color_for_participant(Arc::clone(&state), participant_id).await;
    match speaker_color {
        None => Some("spectator"),
        Some(color) => {
            if is_color_turn(state, color).await {
                None
            } else {
                Some("off_turn")
            }
        }
    }
}

async fn classify_with_openai(
    state: Arc<AppState>,
    openai_client: &OpenAiClient,
    sequence: u64,
    transcript_text: &str,
    latency_label_suffix: &str,
) -> (
    Result<openai::VoiceCommandClassification, String>,
    Instant,
    Instant,
) {
    let legal_moves_uci = {
        let position = state.chess_position.lock().await;
        legal_moves_uci(&position)
    };
    let gpt_started_at = Instant::now();
    let classification = openai_client
        .classify_voice_command(transcript_text, &legal_moves_uci)
        .await
        .map_err(|err| format!("after {}ms: {err}", gpt_started_at.elapsed().as_millis()));
    let gpt_finished_at = Instant::now();
    println!(
        "voice_commands bot01 gpt latency{} #{}: {}ms",
        latency_label_suffix,
        sequence,
        gpt_finished_at
            .saturating_duration_since(gpt_started_at)
            .as_millis()
    );
    (classification, gpt_started_at, gpt_finished_at)
}

fn maybe_dispatch_transcripts_to_openai(state: Arc<AppState>) {
    let openai_client = match &state.openai_client {
        Some(client) => client.clone(),
        None => return,
    };
    tokio::spawn(async move {
        let _dispatch_guard = state.transcript_dispatch_lock.lock().await;

        loop {
            let finalized_transcripts = voice_commands::drain_finalized_transcripts();
            if finalized_transcripts.is_empty() {
                break;
            }

            for transcript in finalized_transcripts {
                let order_id = state
                    .next_transcript_dispatch_order
                    .fetch_add(1, Ordering::Relaxed);
                launch_parallel_gpt_for_transcript(
                    Arc::clone(&state),
                    openai_client.clone(),
                    order_id,
                    transcript,
                );
            }
        }
    });
}

fn launch_parallel_gpt_for_transcript(
    state: Arc<AppState>,
    openai_client: OpenAiClient,
    order_id: u64,
    transcript: voice_commands::TranscriptChunk,
) {
    tokio::spawn(async move {
        let ingested_at = transcript.ingested_at;
        let dispatch_started_at = Instant::now();
        let sequence = transcript.sequence;
        let participant_id = transcript.participant_id;
        let speaker_label = transcript
            .participant_name
            .as_deref()
            .map(str::to_string)
            .or_else(|| participant_id.map(|id| format!("id={id}")))
            .unwrap_or_else(|| "unknown-speaker".to_string());
        let transcript_text = transcript.text;
        let mut gpt_started_at = None;
        let mut gpt_finished_at = None;

        let classification = match participant_id {
            None => Err("missing participant id in transcript payload".to_string()),
            Some(participant_id) => {
                if let Some(skip_reason) =
                    gpt_precheck_skip_reason(Arc::clone(&state), participant_id).await
                {
                    Err(format!("gpt_skipped_precheck:{skip_reason}"))
                } else {
                    let (classification, gpt_start, gpt_done) = classify_with_openai(
                        Arc::clone(&state),
                        &openai_client,
                        sequence,
                        &transcript_text,
                        "",
                    )
                    .await;
                    gpt_started_at = Some(gpt_start);
                    gpt_finished_at = Some(gpt_done);
                    classification
                }
            }
        };
        let result_ready_at = Instant::now();

        let result = TranscriptGptResult {
            order_id,
            sequence,
            participant_id,
            speaker_label,
            transcript_text,
            classification,
            timing: TranscriptPipelineTiming {
                ingested_at,
                dispatch_started_at,
                gpt_started_at,
                gpt_finished_at,
                result_ready_at,
            },
        };

        store_and_apply_transcript_result(state, result).await;
    });
}

async fn store_and_apply_transcript_result(state: Arc<AppState>, result: TranscriptGptResult) {
    {
        let mut pending = state.pending_transcript_results.lock().await;
        pending.insert(result.order_id, result);
    }
    apply_ready_transcript_results_in_order(state).await;
}

async fn apply_ready_transcript_results_in_order(state: Arc<AppState>) {
    let _apply_guard = state.transcript_apply_lock.lock().await;

    loop {
        let expected_order = state.next_transcript_apply_order.load(Ordering::Relaxed);
        let next_result = {
            let mut pending = state.pending_transcript_results.lock().await;
            pending.remove(&expected_order)
        };
        let Some(result) = next_result else {
            break;
        };

        process_ordered_transcript_result(Arc::clone(&state), result).await;
        state
            .next_transcript_apply_order
            .fetch_add(1, Ordering::Relaxed);
    }
}

async fn process_ordered_transcript_result(state: Arc<AppState>, result: TranscriptGptResult) {
    let apply_started_at = Instant::now();
    let sequence = result.sequence;
    let mut timing = result.timing;
    let participant_id = match result.participant_id {
        Some(id) => id,
        None => {
            eprintln!(
                "OpenAI voice command interpretation skipped for #{}: missing participant id",
                sequence
            );
            log_transcript_pipeline_latency(
                sequence,
                &result.speaker_label,
                "missing_participant_id",
                &timing,
                apply_started_at,
            );
            return;
        }
    };
    let speaker_label = result.speaker_label;
    let transcript_text = result.transcript_text;

    if maybe_handle_color_claim(
        Arc::clone(&state),
        participant_id,
        speaker_label.clone(),
        &transcript_text,
    )
    .await
    {
        log_transcript_pipeline_latency(
            sequence,
            &speaker_label,
            "color_claim_handled",
            &timing,
            apply_started_at,
        );
        return;
    }

    let speaker_color =
        match assigned_color_for_participant(Arc::clone(&state), participant_id).await {
            Some(color) => color,
            None => {
                println!(
                    "voice_commands bot01 spectator ignored #{} [{}]: {}",
                    sequence, speaker_label, transcript_text
                );
                log_transcript_pipeline_latency(
                    sequence,
                    &speaker_label,
                    "spectator_ignored",
                    &timing,
                    apply_started_at,
                );
                return;
            }
        };
    if !is_color_turn(Arc::clone(&state), speaker_color).await {
        let chat_message = transcript_text.trim().to_string();
        println!(
            "voice_commands bot01 off-turn chat #{} [{}]: {}",
            sequence, speaker_label, chat_message
        );
        handle_player_chat_message(
            Arc::clone(&state),
            sequence,
            participant_id,
            speaker_label.clone(),
            chat_message,
        )
        .await;
        log_transcript_pipeline_latency(
            sequence,
            &speaker_label,
            "off_turn_chat_displayed",
            &timing,
            apply_started_at,
        );
        return;
    }

    let classification = match result.classification {
        Ok(classification) => Ok(classification),
        Err(err) if err.starts_with("gpt_skipped_precheck:") => {
            let openai_client = state.openai_client.clone();
            match openai_client {
                Some(openai_client) => {
                    let (classification, gpt_started_at, gpt_finished_at) = classify_with_openai(
                        Arc::clone(&state),
                        &openai_client,
                        sequence,
                        &transcript_text,
                        " fallback",
                    )
                    .await;
                    timing.gpt_started_at.get_or_insert(gpt_started_at);
                    timing.gpt_finished_at.get_or_insert(gpt_finished_at);
                    classification
                }
                None => Err("openai client unavailable for fallback classification".to_string()),
            }
        }
        Err(err) => Err(err),
    };

    match classification {
        Ok(classification) => match classification.kind {
            openai::VoiceCommandKind::Command => {
                let command = classification.content.trim().to_string();
                println!(
                    "voice_commands bot01 command #{} [{}]: {}",
                    sequence, speaker_label, command
                );
                let command_outcome = apply_command_to_engine(
                    Arc::clone(&state),
                    sequence,
                    participant_id,
                    speaker_label.clone(),
                    command,
                )
                .await;
                let outcome = format!("command_{}", command_outcome.as_label());
                log_transcript_pipeline_latency(
                    sequence,
                    &speaker_label,
                    &outcome,
                    &timing,
                    apply_started_at,
                );
            }
            openai::VoiceCommandKind::Chat => {
                let chat_message = classification.content.trim().to_string();
                println!(
                    "voice_commands bot01 chat #{} [{}]: {}",
                    sequence, speaker_label, chat_message
                );
                handle_player_chat_message(
                    Arc::clone(&state),
                    sequence,
                    participant_id,
                    speaker_label.clone(),
                    chat_message,
                )
                .await;
                log_transcript_pipeline_latency(
                    sequence,
                    &speaker_label,
                    "chat_displayed",
                    &timing,
                    apply_started_at,
                );
            }
        },
        Err(err) => {
            eprintln!(
                "OpenAI voice command interpretation failed for #{}: {err}",
                sequence
            );
            log_transcript_pipeline_latency(
                sequence,
                &speaker_label,
                "gpt_classification_error",
                &timing,
                apply_started_at,
            );
        }
    }
}

fn log_transcript_pipeline_latency(
    sequence: u64,
    speaker_label: &str,
    outcome: &str,
    timing: &TranscriptPipelineTiming,
    apply_started_at: Instant,
) {
    let completed_at = Instant::now();
    let ingest_to_dispatch_ms = timing
        .dispatch_started_at
        .saturating_duration_since(timing.ingested_at)
        .as_millis();
    let dispatch_to_gpt_start_ms = timing
        .gpt_started_at
        .map(|gpt_started_at| {
            gpt_started_at
                .saturating_duration_since(timing.dispatch_started_at)
                .as_millis()
        })
        .unwrap_or(0);
    let gpt_ms = timing.gpt_started_at.zip(timing.gpt_finished_at).map(
        |(gpt_started_at, gpt_finished_at)| {
            gpt_finished_at
                .saturating_duration_since(gpt_started_at)
                .as_millis()
        },
    );
    let result_queue_wait_ms = apply_started_at
        .saturating_duration_since(timing.result_ready_at)
        .as_millis();
    let apply_ms = completed_at
        .saturating_duration_since(apply_started_at)
        .as_millis();
    let total_ms = completed_at
        .saturating_duration_since(timing.ingested_at)
        .as_millis();

    match gpt_ms {
        Some(gpt_ms) => println!(
            "voice_commands bot01 e2e latency #{} [{}] outcome={} total={}ms ingest_to_dispatch={}ms dispatch_to_gpt={}ms gpt={}ms result_queue_wait={}ms apply={}ms",
            sequence,
            speaker_label,
            outcome,
            total_ms,
            ingest_to_dispatch_ms,
            dispatch_to_gpt_start_ms,
            gpt_ms,
            result_queue_wait_ms,
            apply_ms
        ),
        None => println!(
            "voice_commands bot01 e2e latency #{} [{}] outcome={} total={}ms ingest_to_dispatch={}ms gpt=n/a result_queue_wait={}ms apply={}ms",
            sequence,
            speaker_label,
            outcome,
            total_ms,
            ingest_to_dispatch_ms,
            result_queue_wait_ms,
            apply_ms
        ),
    }
}

async fn assigned_color_for_participant(
    state: Arc<AppState>,
    participant_id: i64,
) -> Option<PlayerColor> {
    let seats = state.player_seats.lock().await;
    seat_color_for_participant(&seats, participant_id)
}

async fn is_color_turn(state: Arc<AppState>, color: PlayerColor) -> bool {
    let position = state.chess_position.lock().await;
    position.turn() == color.to_chess_color()
}

fn player_chat_bubble_slot_mut(
    bubbles: &mut PlayerChatBubbles,
    color: PlayerColor,
) -> &mut Option<PlayerChatBubble> {
    match color {
        PlayerColor::White => &mut bubbles.white,
        PlayerColor::Black => &mut bubbles.black,
    }
}

fn player_chat_bubble_slot_ref(
    bubbles: &PlayerChatBubbles,
    color: PlayerColor,
) -> &Option<PlayerChatBubble> {
    match color {
        PlayerColor::White => &bubbles.white,
        PlayerColor::Black => &bubbles.black,
    }
}

async fn set_player_chat_bubble(
    state: Arc<AppState>,
    color: PlayerColor,
    message_id: u64,
    message: String,
) {
    if message.trim().is_empty() {
        return;
    }

    {
        let mut bubbles = state.player_chat_bubbles.lock().await;
        *player_chat_bubble_slot_mut(&mut bubbles, color) = Some(PlayerChatBubble {
            text: message,
            message_id,
            expires_at: Instant::now() + CHAT_BUBBLE_TTL,
        });
    }
    schedule_chat_rerender(Arc::clone(&state));

    tokio::spawn(async move {
        sleep(CHAT_BUBBLE_TTL).await;
        clear_player_chat_bubble_if_current(state, color, message_id).await;
    });
}

async fn clear_player_chat_bubble_if_current(
    state: Arc<AppState>,
    color: PlayerColor,
    message_id: u64,
) {
    let mut cleared = false;
    {
        let mut bubbles = state.player_chat_bubbles.lock().await;
        if player_chat_bubble_slot_ref(&bubbles, color)
            .as_ref()
            .is_some_and(|bubble| bubble.message_id == message_id)
        {
            *player_chat_bubble_slot_mut(&mut bubbles, color) = None;
            cleared = true;
        }
    }
    if cleared {
        schedule_chat_rerender(state);
    }
}

async fn clear_player_chat_bubble_for_color(state: Arc<AppState>, color: PlayerColor) {
    let mut cleared = false;
    {
        let mut bubbles = state.player_chat_bubbles.lock().await;
        if player_chat_bubble_slot_ref(&bubbles, color).is_some() {
            *player_chat_bubble_slot_mut(&mut bubbles, color) = None;
            cleared = true;
        }
    }
    if cleared {
        schedule_chat_rerender(state);
    }
}

async fn maybe_forward_player_chat_to_lichess(
    state: Arc<AppState>,
    speaker_label: &str,
    chat_message: &str,
) {
    if chat_message.trim().is_empty() {
        return;
    }
    if !state.lichess_chat_supported.load(Ordering::Relaxed) {
        return;
    }

    let lichess_client = match &state.lichess_client {
        Some(client) => client.clone(),
        None => return,
    };
    let lichess_message = format!("{speaker_label}: {chat_message}");

    match lichess_client
        .send_broadcast_chat_message(&lichess_message)
        .await
    {
        Ok(()) => {
            println!(
                "Forwarded player chat to Lichess broadcast round {}",
                lichess_client.game_id()
            );
        }
        Err(LichessApiError::Api { status, .. }) if status == StatusCode::NOT_FOUND => {
            if state.lichess_chat_supported.swap(false, Ordering::SeqCst) {
                eprintln!(
                    "Lichess chat forwarding disabled: broadcast chat endpoint is unavailable"
                );
            }
        }
        Err(err) => {
            eprintln!("Failed to forward player chat to Lichess broadcast: {err}");
        }
    }
}

async fn handle_player_chat_message(
    state: Arc<AppState>,
    sequence: u64,
    participant_id: i64,
    speaker_label: String,
    chat_message: String,
) {
    let chat_message = chat_message.trim().to_string();
    if chat_message.is_empty() {
        return;
    }

    let speaker_color =
        match assigned_color_for_participant(Arc::clone(&state), participant_id).await {
            Some(color) => color,
            None => return,
        };
    set_player_chat_bubble(
        Arc::clone(&state),
        speaker_color,
        sequence,
        chat_message.clone(),
    )
    .await;

    let state_for_side_effects = Arc::clone(&state);
    tokio::spawn(async move {
        send_chat_message_from_bot01(
            Arc::clone(&state_for_side_effects),
            speaker_label.clone(),
            chat_message.clone(),
        )
        .await;
        maybe_forward_player_chat_to_lichess(state_for_side_effects, &speaker_label, &chat_message)
            .await;
    });
}

async fn maybe_handle_color_claim(
    state: Arc<AppState>,
    participant_id: i64,
    _speaker_label: String,
    transcript_text: &str,
) -> bool {
    let requested_color = match parse_color_claim_from_text(transcript_text) {
        Some(color) => color,
        None => return false,
    };

    let participants = state.meeting_participants.lock().await;
    let claimer = match participants.get(&participant_id) {
        Some(participant) => participant.clone(),
        None => return true,
    };

    let mut should_rerender_labels = false;
    let mut announcement = None::<String>;
    {
        let mut seats = state.player_seats.lock().await;
        if let Some(existing) = seat_color_for_participant(&seats, participant_id) {
            announcement = Some(format!(
                "{} already has {}.",
                claimer.name,
                existing.as_str()
            ));
        } else if seats.white.is_some() && seats.black.is_some() {
            // Spectators remain spectators once both seats are occupied.
        } else if seats.white.is_none() && seats.black.is_none() {
            set_player_seat(
                &mut seats,
                requested_color,
                PlayerSeat {
                    participant_id: claimer.id,
                    participant_name: claimer.name.clone(),
                },
            );
            should_rerender_labels = true;

            if let Some((second_id, second_name)) =
                first_available_spectator(&participants, &seats, participant_id)
            {
                let second_color = requested_color.opposite();
                set_player_seat(
                    &mut seats,
                    second_color,
                    PlayerSeat {
                        participant_id: second_id,
                        participant_name: second_name.clone(),
                    },
                );
                announcement = Some(format!(
                    "{} claimed {}. {} was assigned {}.",
                    claimer.name,
                    requested_color.as_str(),
                    second_name,
                    second_color.as_str()
                ));
            } else {
                announcement = Some(format!(
                    "{} claimed {}. Waiting for an opponent to join.",
                    claimer.name,
                    requested_color.as_str()
                ));
            }
        } else {
            let assigned_color = if seats.white.is_none() {
                PlayerColor::White
            } else {
                PlayerColor::Black
            };
            set_player_seat(
                &mut seats,
                assigned_color,
                PlayerSeat {
                    participant_id: claimer.id,
                    participant_name: claimer.name.clone(),
                },
            );
            should_rerender_labels = true;
            announcement = Some(format!(
                "{} was assigned {}.",
                claimer.name,
                assigned_color.as_str()
            ));
        }
    }
    drop(participants);

    if should_rerender_labels {
        rerender_king_label_tiles(Arc::clone(&state)).await;
    }
    if let Some(message) = announcement {
        tokio::spawn(async move {
            send_chat_message_from_bot01(state, "engine".to_string(), message).await;
        });
    }
    true
}

fn parse_color_claim_from_text(text: &str) -> Option<PlayerColor> {
    let normalized = text
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    let has_white = normalized.split_whitespace().any(|token| token == "white");
    let has_black = normalized.split_whitespace().any(|token| token == "black");
    if has_white == has_black {
        return None;
    }

    let has_claim_language = ["claim", "take", "choose", "play", "color"]
        .iter()
        .any(|needle| normalized.contains(needle))
        || normalized.contains("i am")
        || normalized.contains("i m");
    if !has_claim_language {
        return None;
    }

    if has_white {
        Some(PlayerColor::White)
    } else {
        Some(PlayerColor::Black)
    }
}

fn first_available_spectator(
    participants: &BTreeMap<i64, MeetingParticipant>,
    seats: &PlayerSeats,
    claimer_id: i64,
) -> Option<(i64, String)> {
    participants
        .values()
        .filter(|participant| participant.id != claimer_id)
        .filter(|participant| seat_color_for_participant(seats, participant.id).is_none())
        .min_by_key(|participant| participant.join_order)
        .map(|participant| (participant.id, participant.name.clone()))
}

fn set_player_seat(seats: &mut PlayerSeats, color: PlayerColor, seat: PlayerSeat) {
    match color {
        PlayerColor::White => seats.white = Some(seat),
        PlayerColor::Black => seats.black = Some(seat),
    }
}

async fn apply_command_to_engine(
    state: Arc<AppState>,
    sequence: u64,
    participant_id: i64,
    speaker_label: String,
    command: String,
) -> CommandExecutionOutcome {
    let uci = match command.parse::<UciMove>() {
        Ok(uci) => uci,
        Err(_) => {
            println!(
                "voice_commands bot01 invalid command #{} [{}]: '{}' (not valid UCI format)",
                sequence, speaker_label, command
            );
            return CommandExecutionOutcome::InvalidUciFormat;
        }
    };

    let speaker_color =
        match assigned_color_for_participant(Arc::clone(&state), participant_id).await {
            Some(color) => color,
            None => return CommandExecutionOutcome::SpeakerNotSeated,
        };

    let (position_snapshot, changed_tiles, move_history_snapshot) = {
        let mut position = state.chess_position.lock().await;
        if position.turn() != speaker_color.to_chess_color() {
            println!(
                "voice_commands bot01 ignored command #{} [{}]: not {}'s turn",
                sequence,
                speaker_label,
                speaker_color.as_str()
            );
            return CommandExecutionOutcome::NotPlayersTurn;
        }
        let chess_move = match uci.to_move(&*position) {
            Ok(chess_move) => chess_move,
            Err(_) => {
                println!(
                    "voice_commands bot01 invalid command #{} [{}]: '{}' (illegal in current position)",
                    sequence, speaker_label, command
                );
                return CommandExecutionOutcome::IllegalInCurrentPosition;
            }
        };
        let changed = changed_tile_indices_for_move(&chess_move);

        position.play_unchecked(&chess_move);
        let mut history = state.move_history.lock().await;
        history.push(chess_move);

        (position.clone(), changed, history.clone())
    };
    println!(
        "voice_commands bot01 applied command #{} [{}]: '{}'",
        sequence, speaker_label, command
    );
    let chat_notice = format!("Applied command #{sequence} from {speaker_label}: {command}");
    let state_for_chat = Arc::clone(&state);
    tokio::spawn(async move {
        send_chat_message_from_bot01(state_for_chat, "engine".to_string(), chat_notice).await;
    });

    let lichess_client = state.lichess_client.clone();
    if let Some(lichess_client) = lichess_client {
        let state_for_lichess = Arc::clone(&state);
        let move_history_for_lichess = move_history_snapshot.clone();
        tokio::spawn(async move {
            let (white_name, black_name) =
                current_player_names(Arc::clone(&state_for_lichess)).await;
            let pgn = build_relay_pgn(
                &move_history_for_lichess,
                white_name.as_deref(),
                black_name.as_deref(),
            );
            match lichess_client.push_broadcast_pgn(&pgn).await {
                Ok(()) => {
                    println!(
                        "Pushed move #{} to Lichess broadcast round {}",
                        move_history_for_lichess.len(),
                        lichess_client.game_id()
                    );
                }
                Err(err) => {
                    eprintln!(
                        "Failed to push move #{} to Lichess broadcast round {}: {}",
                        move_history_for_lichess.len(),
                        lichess_client.game_id(),
                        err
                    );
                }
            }
        });
    }

    rerender_tiles_after_move(Arc::clone(&state), &position_snapshot, &changed_tiles).await;

    CommandExecutionOutcome::Applied
}

fn legal_moves_uci(position: &Chess) -> Vec<String> {
    position
        .legal_moves()
        .into_iter()
        .map(|chess_move| UciMove::from_move(&chess_move, CastlingMode::Standard).to_string())
        .collect()
}

async fn current_player_names(state: Arc<AppState>) -> (Option<String>, Option<String>) {
    let seats = state.player_seats.lock().await;
    (
        seats
            .white
            .as_ref()
            .map(|seat| seat.participant_name.clone()),
        seats
            .black
            .as_ref()
            .map(|seat| seat.participant_name.clone()),
    )
}

fn build_relay_pgn(
    move_history: &[Move],
    white_name: Option<&str>,
    black_name: Option<&str>,
) -> String {
    build_relay_pgn_with_result(move_history, "*", white_name, black_name)
}

fn build_relay_pgn_with_result(
    move_history: &[Move],
    result: &str,
    white_name: Option<&str>,
    black_name: Option<&str>,
) -> String {
    let result = normalize_pgn_result(result);
    let white = sanitize_pgn_header_value(white_name.unwrap_or("Zoom White"));
    let black = sanitize_pgn_header_value(black_name.unwrap_or("Zoom Black"));
    let mut position = Chess::default();
    let mut movetext_parts = Vec::with_capacity(move_history.len() + 4);
    for (index, chess_move) in move_history.iter().enumerate() {
        if index % 2 == 0 {
            movetext_parts.push(format!("{}.", index / 2 + 1));
        }
        let san = SanPlus::from_move_and_play_unchecked(&mut position, chess_move).to_string();
        movetext_parts.push(san);
    }
    movetext_parts.push(result.to_string());

    format!(
        "[Event \"Hackathon Zoom Relay\"]\n[Site \"Lichess\"]\n[Date \"????.??.??\"]\n[Round \"1\"]\n[White \"{white}\"]\n[Black \"{black}\"]\n[Result \"{result}\"]\n\n{}\n",
        movetext_parts.join(" ")
    )
}

fn sanitize_pgn_header_value(value: &str) -> String {
    let compact = value
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_graphic() || *ch == ' ')
        .collect::<String>();
    let escaped = compact.replace('\\', "\\\\").replace('"', "\\\"");
    if escaped.is_empty() {
        "Unknown".to_string()
    } else {
        escaped
    }
}

fn normalize_pgn_result(result: &str) -> &'static str {
    match result.trim() {
        "1-0" => "1-0",
        "0-1" => "0-1",
        "1/2-1/2" => "1/2-1/2",
        "*" => "*",
        _ => "*",
    }
}

fn relay_shutdown_result(position: &Chess) -> &'static str {
    position
        .outcome()
        .map(|outcome| outcome.as_str())
        .unwrap_or("1/2-1/2")
}

fn changed_tile_indices_for_move(chess_move: &Move) -> BTreeSet<usize> {
    let mut changed_tiles = BTreeSet::new();

    if let Some(from) = chess_move.from() {
        changed_tiles.insert(tile_index_for_square(from));
    }
    changed_tiles.insert(tile_index_for_square(chess_move.to()));

    match chess_move {
        Move::EnPassant { from, to } => {
            let to_rank = u32::from(to.rank());
            let from_rank = u32::from(from.rank());
            let capture_square = if to_rank > from_rank {
                to.offset(-8)
            } else {
                to.offset(8)
            };
            if let Some(capture_square) = capture_square {
                changed_tiles.insert(tile_index_for_square(capture_square));
            }
        }
        Move::Castle { king, rook } => {
            let (king_delta, rook_delta) = if king < rook { (2, 1) } else { (-2, -1) };
            if let Some(king_to) = king.offset(king_delta) {
                changed_tiles.insert(tile_index_for_square(king_to));
            }
            if let Some(rook_to) = king.offset(rook_delta) {
                changed_tiles.insert(tile_index_for_square(rook_to));
            }
        }
        _ => {}
    }

    changed_tiles
}

fn schedule_chat_rerender(state: Arc<AppState>) {
    let generation = state
        .chat_rerender_generation
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    tokio::spawn(async move {
        sleep(Duration::from_millis(CHAT_RERENDER_DEBOUNCE_MS)).await;
        if state.chat_rerender_generation.load(Ordering::Relaxed) != generation {
            return;
        }
        rerender_all_tiles(state).await;
    });
}

fn tile_index_for_square(square: Square) -> usize {
    let file = u32::from(square.file());
    let rank_from_bottom = u32::from(square.rank());
    let rank_from_top = 7 - rank_from_bottom;
    let tile_file = file / 2;
    let tile_rank_top = rank_from_top / 2;
    (tile_rank_top * 4 + tile_file) as usize
}

fn king_tile_indices(position: &Chess) -> BTreeSet<usize> {
    let mut changed_tiles = BTreeSet::new();
    if let Some(white_king) = position.board().king_of(ChessColor::White) {
        changed_tiles.insert(tile_index_for_square(white_king));
    }
    if let Some(black_king) = position.board().king_of(ChessColor::Black) {
        changed_tiles.insert(tile_index_for_square(black_king));
    }
    changed_tiles
}

async fn rerender_king_label_tiles(state: Arc<AppState>) {
    let _render_guard = state.render_lock.lock().await;
    let position_snapshot = { state.chess_position.lock().await.clone() };
    let changed_tiles = king_tile_indices(&position_snapshot);
    rerender_tiles_after_move_internal(Arc::clone(&state), &position_snapshot, &changed_tiles)
        .await;
}

async fn rerender_all_tiles(state: Arc<AppState>) {
    let _render_guard = state.render_lock.lock().await;
    let position_snapshot = { state.chess_position.lock().await.clone() };
    let all_tiles = BTreeSet::new();
    rerender_tiles_after_move_internal(Arc::clone(&state), &position_snapshot, &all_tiles).await;
}

async fn rerender_tiles_after_move(
    state: Arc<AppState>,
    position: &Chess,
    changed_tiles: &BTreeSet<usize>,
) {
    let _render_guard = state.render_lock.lock().await;
    rerender_tiles_after_move_internal(Arc::clone(&state), position, changed_tiles).await;
}

async fn rerender_tiles_after_move_internal(
    state: Arc<AppState>,
    position: &Chess,
    changed_tiles: &BTreeSet<usize>,
) {
    let rerender_started_at = Instant::now();
    let client = match &state.recall_client {
        Some(client) => client.clone(),
        None => {
            eprintln!("Could not rerender tiles: Recall client is not configured");
            return;
        }
    };

    let bots = {
        let managed_bots = state.managed_bots.lock().await;
        managed_bots
            .iter()
            .map(|managed_bot| {
                (
                    managed_bot.bot.id.clone(),
                    managed_bot.bot.name.clone(),
                    managed_bot.assignment.participant_slot,
                )
            })
            .collect::<Vec<_>>()
    };
    let (white_player_name, black_player_name) = {
        let seats = state.player_seats.lock().await;
        (
            seats
                .white
                .as_ref()
                .map(|seat| seat.participant_name.clone()),
            seats
                .black
                .as_ref()
                .map(|seat| seat.participant_name.clone()),
        )
    };
    let (white_chat_message, black_chat_message) = {
        let bubbles = state.player_chat_bubbles.lock().await;
        let now = Instant::now();
        (
            bubbles
                .white
                .as_ref()
                .filter(|bubble| bubble.expires_at > now)
                .map(|bubble| bubble.text.clone()),
            bubbles
                .black
                .as_ref()
                .filter(|bubble| bubble.expires_at > now)
                .map(|bubble| bubble.text.clone()),
        )
    };
    let render_started_at = Instant::now();
    let output_tile_count = bots.len();
    let wall_tile_jpegs = match render::render_current_board_wall_tile_jpegs(
        position.board(),
        output_tile_count,
        white_player_name.as_deref(),
        black_player_name.as_deref(),
        white_chat_message.as_deref(),
        black_chat_message.as_deref(),
    ) {
        Ok(jpegs) => jpegs,
        Err(err) => {
            eprintln!("Failed to render board wall tiles: {err}");
            return;
        }
    };
    let render_ms = render_started_at.elapsed().as_millis();

    let can_filter_by_changed_tiles = !changed_tiles.is_empty() && output_tile_count == 16;
    let target_bot_count = if !can_filter_by_changed_tiles {
        bots.len()
    } else {
        bots.iter()
            .filter(|(_, _, tile_index)| changed_tiles.contains(tile_index))
            .count()
    };
    let mut output_jobs = Vec::with_capacity(target_bot_count);
    for (bot_id, bot_name, tile_index) in bots {
        if can_filter_by_changed_tiles && !changed_tiles.contains(&tile_index) {
            continue;
        }
        match wall_tile_jpegs.get(tile_index).cloned() {
            Some(jpeg) => output_jobs.push((bot_id, bot_name, tile_index, jpeg)),
            None => {
                eprintln!(
                    "Rendered board wall is missing tile {} for bot {} ({})",
                    tile_index + 1,
                    bot_id,
                    bot_name
                );
            }
        }
    }
    let output_parallelism = output_video_parallelism_from_env();
    let output_started_at = Instant::now();
    let output_results = stream::iter(output_jobs.into_iter().map(
        |(bot_id, bot_name, tile_index, jpeg)| {
            let client = client.clone();
            async move {
                let output_result = client.output_video_jpeg(&bot_id, &jpeg).await;
                (bot_id, bot_name, tile_index, output_result)
            }
        },
    ))
    .buffer_unordered(output_parallelism)
    .collect::<Vec<_>>()
    .await;
    let mut output_failures = 0usize;
    for (bot_id, bot_name, tile_index, output_result) in output_results {
        if let Err(err) = output_result {
            output_failures += 1;
            eprintln!(
                "Failed to output tile {} for bot {} ({}): {err}",
                tile_index + 1,
                bot_id,
                bot_name
            );
        }
    }

    println!(
        "render latency: total={}ms render={}ms output={}ms target_bots={} failures={} parallelism={}",
        rerender_started_at.elapsed().as_millis(),
        render_ms,
        output_started_at.elapsed().as_millis(),
        target_bot_count,
        output_failures,
        output_parallelism
    );
}

async fn send_chat_message_from_bot01(
    state: Arc<AppState>,
    speaker_label: String,
    chat_message: String,
) {
    if chat_message.trim().is_empty() {
        return;
    }

    let client = match &state.recall_client {
        Some(client) => client.clone(),
        None => {
            eprintln!("Could not send chat message: Recall client is not configured");
            return;
        }
    };

    let bot01_id = {
        let managed_bots = state.managed_bots.lock().await;
        managed_bots
            .iter()
            .find(|managed_bot| managed_bot.assignment.participant_slot == 0)
            .map(|managed_bot| managed_bot.bot.id.clone())
    };
    let bot01_id = match bot01_id {
        Some(bot_id) => bot_id,
        None => {
            eprintln!("Could not send chat message: bot 01 is not ready yet");
            return;
        }
    };

    match client
        .send_chat_message(&bot01_id, &state.chat_send_to, &chat_message)
        .await
    {
        Ok(()) => {
            println!(
                "voice_commands bot01 chat sent [{} -> {}]: {}",
                speaker_label, state.chat_send_to, chat_message
            );
        }
        Err(err) => {
            eprintln!("Failed to send bot01 chat message via Recall API: {err}");
        }
    }
}

async fn apply_bot_action(state: Arc<AppState>, action: BotAction) {
    if action == BotAction::LeaveAllManagedBots {
        leave_all_managed_bots(state).await;
    }
}

async fn leave_all_managed_bots(state: Arc<AppState>) {
    let _ = leave_managed_bots_internal(state, true, "leave command").await;
}

async fn leave_managed_bots_for_relaunch(state: Arc<AppState>) -> Result<(), String> {
    leave_managed_bots_internal(state, false, "meeting relaunch").await
}

async fn leave_managed_bots_internal(
    state: Arc<AppState>,
    set_leave_requested: bool,
    reason: &str,
) -> Result<(), String> {
    if set_leave_requested {
        state.leave_requested.store(true, Ordering::SeqCst);
    }
    let client = match &state.recall_client {
        Some(client) => client.clone(),
        None => {
            return Err("Recall client is not configured".to_string());
        }
    };

    let bots = {
        let mut managed_bots = state.managed_bots.lock().await;
        std::mem::take(&mut *managed_bots)
    };

    if bots.is_empty() {
        let mut active_meeting_url = state.active_meeting_url.lock().await;
        *active_meeting_url = None;
        return Ok(());
    }

    println!(
        "Removing {} managed bots from the call ({reason})...",
        bots.len()
    );

    let delay = leave_call_delay();
    let total_bots = bots.len();
    let mut failed_bots = Vec::new();
    for (index, managed_bot) in bots.into_iter().enumerate() {
        let bot_id = managed_bot.bot.id.clone();
        let bot_name = managed_bot.bot.name.clone();
        let slot = managed_bot.assignment.participant_slot + 1;
        let controlled = managed_bot.assignment.controlled_squares.join(", ");
        let bot_for_retry = managed_bot.clone();
        match client.leave_call(managed_bot.bot).await {
            Ok(()) => {
                println!(
                    "Left call for bot {} ({}) in slot {} controlling {}",
                    bot_id, bot_name, slot, controlled
                );
            }
            Err(err) => {
                eprintln!(
                    "Failed to leave call for bot {} ({}) in slot {} controlling {}: {err}",
                    bot_id, bot_name, slot, controlled
                );
                failed_bots.push(bot_for_retry);
            }
        }

        if index + 1 < total_bots {
            sleep(delay).await;
        }
    }

    if !failed_bots.is_empty() {
        let failed_count = failed_bots.len();
        let mut managed_bots = state.managed_bots.lock().await;
        managed_bots.extend(failed_bots);
        return Err(format!(
            "{failed_count} bots could not be removed and were kept in managed state for retry"
        ));
    }

    let mut active_meeting_url = state.active_meeting_url.lock().await;
    *active_meeting_url = None;
    Ok(())
}

fn leave_call_delay() -> Duration {
    let millis = env::var("RECALL_LEAVE_CALL_DELAY_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_LEAVE_DELAY_MS);
    Duration::from_millis(millis)
}

fn build_run_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}-{}-{}", now.as_secs(), now.subsec_nanos(), process::id())
}
