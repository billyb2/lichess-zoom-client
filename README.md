# Zoom Chess Wall (Recall Hackathon)

This is a hackathon project built with the Recall API. So AI generated it hurts.

**WARNING:** This is insanely expensive to run if you don't work at Recall. Recommend setting the

It runs a multi-bot chess wall in a Zoom meeting:
- Configurable number of Recall bots render chess tiles
- Bot `01` handles transcription + voice command interpretation
- Commands are applied to the engine and reflected in the wall

## Run

1. Set env vars in `.env`:
- `RECALL_API_KEY`
- `RECALL_MEETING_URL` (Zoom join URL used for startup bot launch)
- `RECALL_BOT_COUNT` (optional, default `16`; set to `49` for a 49-bot wall)
- `RECALL_ZOOM_VARIANT` (optional, default `regular`; options: `regular`/`web`, `web_4_core`, `web_gpu`)
- `RECALL_WEBHOOK_URL` (public URL Recall can call, e.g. `/recall/webhook`)
- `OPENAI_API_KEY` (for transcript -> command/chat classification)

2. Start the app:

```bash
cargo run --release
```

## Ports

- Recall webhook server: `0.0.0.0:7777` (override with `RECALL_WEBHOOK_BIND`)

## Notes

- Bots launch automatically at startup from `RECALL_MEETING_URL`.
- Launch logs print each bot creation attempt, including the selected Zoom variant.
