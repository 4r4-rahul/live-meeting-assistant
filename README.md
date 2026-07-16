# Live Meeting Assistant — Mock Interview Edition

A Claude-powered, real-time interview assistant. Two modes:

- **Live Coach** — during a real interview, it transcribes the interviewer and shows you tight,
  truthful answer cues grounded in *your* experience (prep brief / résumé / experience bank).
- **AI Interviewer** — a realistic mock interviewer that asks role-relevant questions one at a time
  and follows up on your answers.

> Rewritten in 2026 from the original OpenAI/LangChain/Whisper prototype onto **Anthropic Claude**.
> Backend is async, streaming, and grounded; the old RAG/FAISS/Whisper plumbing was removed.

## Architecture

```
Browser (Next.js)                         FastAPI backend (whisper-backend/)
─────────────────                         ──────────────────────────────────
mic / system-audio → STT  ──transcript──► POST /coach        → Claude (Haiku, fast)   → answer cues
React UI + context upload ──context─────► POST /interviewer  → Claude (Sonnet)        → next question
                                          POST /upload-context/ (PDF/TXT/MD/DOCX → text, no LLM)
                                          GET  /health
```

## Setup

### 1. Backend
```bash
cd whisper-backend
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
# Create ../.env from ../.env.example and set ANTHROPIC_API_KEY
uvicorn main:app --reload --port 8000
```

### 2. Frontend
```bash
npm install
npm run dev          # http://localhost:3000
```

## Configuration (`.env`)
| var | default | purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — (required) | Claude API key |
| `COACH_MODEL` | `claude-haiku-4-5` | live-coach model (latency-critical) |
| `INTERVIEWER_MODEL` | `claude-sonnet-4-6` | mock-interviewer model |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS allowlist |

## Audio capture (how transcription works)
Capture is **local Whisper**, not the browser's Web Speech API. The app records the **selected input
device** in ~5s segments and transcribes each via `POST /transcribe` (faster-whisper, offline, no API
key, ~0.4s/segment). Because it captures the device you pick, the **Input device** dropdown is real —
selecting BlackHole captures BlackHole, selecting your mic captures your mic.

**Mock practice (your own voice):** just pick your microphone in the dropdown and Start Listening.

**Live coach (the interviewer's voice):** route the meeting's audio into BlackHole, then pick
BlackHole in the dropdown:
1. `brew install blackhole-2ch`
2. **Audio MIDI Setup** → create a **Multi-Output Device** = your headphones + BlackHole 2ch (so you
   still hear the interviewer).
3. Set your meeting app's **speaker/output** to that Multi-Output Device. Wear headphones.
4. In the app, pick **🎧 BlackHole 2ch** as the Input device. (No need to change the macOS default
   input — the app captures the selected device directly.)

Config: `STT_MODEL` (default `base.en`; use `small.en`/`medium.en` for more accuracy, `tiny.en` for
speed). First transcription downloads the model (~150 MB for base).

## Security
- The previous version committed a live OpenAI key in `environment/dev.env` — **it is in git history
  and must be rotated.** It is no longer used (we're on Claude) but rotate/revoke it anyway.
- Real secrets live in `.env` (gitignored). Never commit them.

## Status / roadmap
Next up: streaming token UI, getDisplayMedia tab-audio capture + a streaming STT (Deepgram) for
robust interviewer capture, and pulling grounding straight from the Job_Finder experience bank.
