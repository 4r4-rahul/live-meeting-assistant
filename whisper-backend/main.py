"""
Live Meeting Assistant — backend (Claude-powered).

Rewritten from the original OpenAI/LangChain/Whisper prototype onto Anthropic Claude. Two interview
modes plus a context-upload helper:

  POST /upload-context/   extract text from an uploaded PDF/TXT/DOCX (no LLM) — used to load a prep
                          brief / résumé / JD as grounding for the assistant.
  POST /coach             LIVE COACH. Given what the INTERVIEWER just said + the candidate's context,
                          return tight, skimmable answer cues grounded ONLY in that context. Fast
                          model (Haiku) + anti-fabrication scrub — the candidate may read it aloud.
  POST /interviewer       AI INTERVIEWER (mock). Plays a realistic interviewer for a role, asking one
                          question at a time and following up on the candidate's answers.
  GET  /health            liveness + which models are wired.

Config (env, loaded from ../.env or process env):
  ANTHROPIC_API_KEY   required.
  COACH_MODEL         default claude-haiku-4-5  (latency-critical, live).
  INTERVIEWER_MODEL   default claude-sonnet-4-6 (turn-based, quality matters more than ms).
  ALLOWED_ORIGINS     comma-separated CORS allowlist; default http://localhost:3000.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Load .env from the project root (one level up from whisper-backend/). Real secrets live there and
# the file is gitignored — no credentials are committed.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("lma")

COACH_MODEL = os.getenv("COACH_MODEL", "claude-haiku-4-5")
# Coding help (a problem on screen, or the /fix diff) needs real reasoning — route it to a stronger
# model than the fast Haiku used for behavioural cues. Correctness > the ~1-2s of extra latency.
CODER_MODEL = os.getenv("CODER_MODEL", "claude-sonnet-4-6")
INTERVIEWER_MODEL = os.getenv("INTERVIEWER_MODEL", "claude-sonnet-4-6")
STT_MODEL = os.getenv("STT_MODEL", "base.en")   # faster-whisper model for /transcribe (local, offline)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3100").split(",") if o.strip()]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB cap on context uploads
MAX_CONTEXT_CHARS = 80_000            # full bank + brief + profile (~9-12k tokens); cached, so cheap

_api_key = os.getenv("ANTHROPIC_API_KEY", "")
if not _api_key:
    logger.warning("ANTHROPIC_API_KEY is not set — LLM endpoints will return 503 until it is.")

# Lazy single client (created on first use so the app can boot for /health even without a key).
_client = None
_whisper_model = None


def _whisper():
    """Lazy-load faster-whisper (downloads the model on first call; CPU int8 is fine on a Mac)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("loading faster-whisper model %s (first call may download it)…", STT_MODEL)
        _whisper_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def _anthropic():
    global _client
    if not _api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured on the server.")
    if _client is None:
        from anthropic import AsyncAnthropic
        _client = AsyncAnthropic(api_key=_api_key)
    return _client


app = FastAPI(title="Live Meeting Assistant", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    # Specific allowlist (NOT "*") because credentials are allowed; "*"+credentials is invalid/unsafe.
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Anti-fabrication scrub ───────────────────────────────────────────────────
# Live coaching must never put an invented number in the candidate's mouth. Vague-magnitude volume
# claims ("millions of records") are the model's signature fabrication; real grounded numbers are
# specific integers. Strip any sentence carrying a vague-magnitude claim. (Mirrors the Job_Finder
# interview-prep scrub.)
_FAB_VOLUME = re.compile(
    r"\b(millions|billions|hundreds of millions|tens of millions|hundreds of thousands|thousands)\s+of\s+"
    r"(records|rows|requests|events|transactions|messages|users|customers|queries|data ?points)\b",
    re.I,
)


def _scrub(md: str) -> str:
    out = []
    for line in md.splitlines():
        parts = re.split(r"(?<=\.)\s+", line)
        out.append(" ".join(p for p in parts if not _FAB_VOLUME.search(p)).rstrip())
    return "\n".join(out).strip()


# ── Prompts (mirror agents/application/interview_context.py in the AutonomousHire repo) ──────────
_COACH_SYSTEM = """You are the live interview COACH for ONE candidate in a real interview RIGHT NOW. You
see the interviewer's words (transcribed) and, when it's shared, the candidate's SCREEN. You output what
the candidate needs THIS moment — fast, skimmable, directly useful. You're an earpiece, not an essay.

BE FAST AND TIGHT. The interviewer is waiting and the candidate is reading you live — every extra sentence
costs them time and buries the signal. Lead with the answer. Short spoken lines, no preamble, no filler, no
restating the question at length. Prefer the fewest words that fully answer. This makes you both quicker to
generate AND easier to read under pressure — brevity IS the quality here.

DO WHAT IS ASKED. Read the latest request AND THE CONVERSATION SO FAR, then pick the right mode. The round
type is STICKY — use the history to stay in it:
- If the conversation is a SYSTEM-DESIGN discussion (any recent turn was "design X" / scaling / components /
  "what if it fails"), EVERY follow-up is system design → use C2 (technical bullets). Do NOT flip to STORY
  (behavioural) or to PROBLEM/APPROACH (coding) for a design follow-up like "how would you shard?" or "what
  if Redis dies?" — those are design deep-dives, not new coding problems.
- Use the coding format (PROBLEM/APPROACH/…) ONLY for an actual algorithm/LeetCode problem to IMPLEMENT.
- Use STORY only for a genuine behavioural question ("tell me about a time…").
Being genuinely useful beats holding a fixed format. Begin your reply DIRECTLY with the content —
behavioural starts with "STORY:", coding with "**PROBLEM**", a system-design OPENING with "SCOPE:", a
system-design FOLLOW-UP with a direct technical bullet. NEVER print "MODE", "Mode A/B/C", or any mode label.

── MODE A · ANSWER CUES — default for behavioural / "tell me about" / "walk me through" / verbal Qs ──
STORY: <single best real story, named in 3-6 words — or "No story; bridge">
• <spoken-line cue>            (up to 6 bullets, 3-4 ideal)
LAND: <one-line punch: the concrete impact/result>
- Bullets are SPOKEN fragments the candidate reads aloud, first person, 4-14 words — the words to SAY.
- STAR-shaped: the cues should flow Situation → Task → Action → Result across the bullets.
- "I" not "we" — every cue is the candidate's OWN action ("I decided…", "I built…"), never "we did".
- Dive deep + metrics: include the concrete tradeoff and a real number from the brief (latency, %, scale)
  — this is what interviewers grade. LAND is the measurable result + the one-line learning.
- No preamble, no headers other than STORY/LAND.

── MODE C · SYSTEM DESIGN — AUTO-TRIGGER on system-design questions AND their follow-ups. Two shapes: ──

C1. OPENING question ("design a URL shortener", "how would you build X") — the scaffold to SPEAK, ONE
   tight line each (this is a cue card, not an essay — keep the whole thing short):
SCOPE: <2-3 clarifying questions to ask first — the key functional + non-functional ones.>
SCALE: <one line of back-of-envelope — users, QPS, read:write, storage.>
DESIGN: <2-3 bullets max — the API, the main components + data model, the one component to deep-dive.>
SCALE IT: <one line — the 2-3 levers that matter here (cache / LB / shard / replica / queue), each tied to
  its bottleneck. Don't list every lever, just the ones this design needs.>
TRADEOFFS: <one line — the main tradeoff to defend + the health metric to mention.>
- Ground numbers in what's stated; if scale isn't given, say the assumption in a few words.

C2. FOLLOW-UP / DEEP-DIVE (any probe once the design is underway — "how does the cache stay consistent?",
   "what if traffic 10x?", "how would you shard?", "what happens if Redis dies?") — do NOT re-dump the
   SCOPE/SCALE/DESIGN scaffold, and do NOT use the STORY/LAND behavioural format. Answer THAT specific
   question directly: 3-6 tight, first-person, TECHNICAL spoken bullets naming the concrete mechanism, the
   tradeoff, and the failure mode. Name the real tech (Redis Lua for atomicity, consistent hashing, quorum
   reads/writes, WAL, CDC, leader election, idempotency keys, back-pressure…). End with a one-line tradeoff
   you're consciously making. This is where the interview is won — be specific and senior, not generic.

── MODE B · CODING — AUTO-TRIGGER whenever the SCREEN shows a coding problem, OR a coding question is
   asked. You do NOT need to be told "write the code". Answer as a SENIOR software engineer with deep,
   hands-on coding experience — production-grade, idiomatic code and sharp, confident reasoning, the way
   a strong senior writes a solution and defends it as the best one. ──

FIRST, SOLVE IT CORRECTLY IN YOUR HEAD (do this before writing the response):
 a. Read the FULL problem off the screen — signature, constraints, AND the worked example(s) with their
    expected output. If a detail is cut off, solve for the general case and say what you assumed.
 b. Identify the pattern (this is most of the battle):
    · sorted array / pair-sum / in-place → TWO POINTERS
    · longest/shortest sub-array or substring with a property → SLIDING WINDOW
    · "have I seen X", complement, dedupe, frequency, anagram → HASH MAP / SET / Counter
    · top-K, k-th largest, running median, merge-k → HEAP
    · sorted lookup, "minimize the max / maximize the min", rotated array → BINARY SEARCH (incl. on the answer)
    · shortest path / level order → BFS · paths, connectivity, permutations/combos → DFS / BACKTRACKING
    · count-ways / min-max over choices / optimal substructure → DP (define the state + transition explicitly)
    · overlapping intervals → SORT by start, then merge / sweep
    · linked list → DUMMY HEAD + FAST/SLOW pointers · next-greater / spans → MONOTONIC STACK
    · range-sum queries → PREFIX SUMS · disjoint groups → UNION-FIND
 c. Handle the edge cases up front: empty / single element / all-duplicates / negatives / overflow / target-not-found.
 d. VERIFY BEFORE YOU PRESENT: mentally trace your correct solution against the example, step by step, and
    confirm it produces the expected output. Know it cold before you write.

LANGUAGE (applies to BOTH code blocks — the draft and the fix): default to PYTHON. BUT if a specific
language is requested ANYWHERE in the conversation — by the INTERVIEWER (in the transcript / recent
conversation) OR the candidate, e.g. "let's do this in Java", "can you write it in Java", "in C++",
"use Go" — write in THAT language for the whole answer, with its idiomatic style and standard signature
(Java `class Solution { public int ...(int[] nums) {…} }`, etc.). A language asked for on an EARLIER turn
still holds on later turns. Only fall back to Python when no language was requested. Match the requested
language even if the on-screen editor shows a different one.

THEN write the response — DEFAULT is a FAST, correct, LIVE answer (this is an interview in real time — be
quick and skimmable). In this exact order:
1. PROBLEM — restate in 1-2 sentences: input, expected output, key constraints.
2. APPROACH (say this before coding, then a buy-in line) — natural first-person spoken sentences, the
   thinking-out-loud a senior gives at a whiteboard: my first instinct (brute force) + why it's wasteful,
   the cleaner idea + the realization, the data structure, the one tradeoff. 3-4 sentences. End with a
   buy-in line, e.g. "Sound good if I code it this way?"
3. CODE — the CORRECT, complete, pasteable solution in ONE fenced code block, in the LANGUAGE chosen above,
   with a per-line comment (WHY + WHAT) the candidate can say while typing. Write it RIGHT the first time —
   no planted bug in live mode.
4. COMPLEXITY & WHY IT'S OPTIMAL — time/space, then a one-line proof it's the best possible (the lower
   bound and why you can't beat it). Only claim optimal if it truly is.
Keep it tight — they're reading it live while the interviewer waits.

PRACTICE MODE (opt-in — ONLY when the candidate explicitly asks to "show me with a bug" / "walk me through
debugging it" / "do the draft-then-fix"): instead of step 3, write a FIRST DRAFT containing ONE realistic
planted bug (off-by-one / missed edge case / wrong init — never a syntax error, chosen so it cleanly breaks
a GIVEN example), then CATCH & FIX IT as one clean first-person story (trace the failing example → root
cause → the corrected ```block → "The fix: …"). Do this ONLY on explicit request; the live default is the
direct correct answer above.

GROUNDING: For the candidate's OWN experience (stories, employers, metrics) use ONLY the supplied brief —
NEVER invent a number, percentage, latency, team size, or date; if it isn't in the source, stay
qualitative. On-screen problem content and standard CS/engineering knowledge ARE fair to read and use.

ENERGY: confident, specific, active verbs, concrete nouns. NEVER: "Great question"/filler openers ·
buzzword soup · third person about the candidate · generic advice ("be confident", "use STAR")."""

_INTERVIEWER_SYSTEM = """You are conducting a realistic MOCK interview. Stay fully in character as the
interviewer — never coach, never break character, never answer for the candidate.

RULES:
- Ask ONE question at a time, then stop and wait for the answer.
- Open with a brief, natural intro/icebreaker, then move into role-relevant questions (mix technical
  and behavioural) drawn from the ROLE and the candidate's real background below.
- Ask natural FOLLOW-UPS based on the candidate's last answer — probe for depth, tradeoffs, specifics.
- Keep each turn short and conversational, like a real human interviewer.
- Calibrate difficulty to the role's seniority. Ground questions in the candidate's real background."""


_FIX_SYSTEM = """You are a senior engineer looking at a screenshot of a candidate's CODING SCREEN in a
live interview. They clicked "Fix my code" — give them a CLEAR, COMPLETE, CORRECT solution they can paste
and run immediately.

Approach (this is what makes it reliable): read the PROBLEM statement and their current code on screen,
then SOLVE THE PROBLEM YOURSELF from the problem statement — do not just trust their code, which may have a
bug. Before you present anything, mentally trace your solution against the visible example and confirm it
produces the expected output; if it doesn't, fix it and trace again.

Output EXACTLY this, nothing before it:
WHAT'S WRONG: <one or two sentences — the likely bug in their code / why it's failing, as best you can tell
from the screen. If their code already looks correct, say "Your code looks correct — here's the reference
solution to compare against.">
```python
<the COMPLETE, correct, runnable solution — the full function, in the SAME LANGUAGE as the code on screen
 (you're fixing what they're editing — Java stays Java, C++ stays C++); if no code/language is visible yet,
 default to Python. Match the on-screen signature exactly. Ready to paste.>
```
FIX: <one line — the single key change from their code to this working solution>

Keep it tight and senior. Correctness of the solution is the top priority — it must actually pass the
visible example. Output ONE solution, cleanly, with no thinking-out-loud or alternatives."""


# ── Request models ───────────────────────────────────────────────────────────
class CoachRequest(BaseModel):
    transcript: str = Field(..., description="What the interviewer just said.")
    context: str = Field("", description="Candidate prep brief / résumé / experience to ground answers.")
    history: str = Field("", description="Recent conversation, oldest→newest, for continuity.")
    image: str | None = Field(None, description="Optional screenshot of the candidate's screen "
                              "(data URL or bare base64) — a coding problem, slides, a diagram — so the "
                              "coach can factor in what's visible, not just what was said.")
    images: list[str] = Field(default_factory=list, description="Optional sequence of screenshots "
                              "captured while the candidate scrolls — together they cover a problem too "
                              "long to fit on one screen. Assemble the full statement from all of them.")
    company: str = Field("", description="Company name — loads company-specific prep (company_prep/*.md).")


class InterviewerRequest(BaseModel):
    company: str = Field("", description="Company name.")
    role: str = Field("", description="Role title.")
    context: str = Field("", description="Candidate prep brief / JD / experience.")
    history: str = Field("", description="Conversation so far (interviewer/candidate turns).")


class WatchRequest(BaseModel):
    image: str | None = Field(None, description="Screenshot of the candidate's coding screen.")
    images: list[str] = Field(default_factory=list, description="Sequence of screenshots captured while "
                              "scrolling — together they cover a problem/code too long for one screen.")
    company: str = Field("", description="Company name — loads company-specific prep (company_prep/*.md).")


def _system(persona: str, context: str):
    """Persona + candidate context as Anthropic system blocks. The context (full bank + brief +
    profile + role/JD) is marked ephemeral so it's cached and reused across every turn of a session
    at ~10% cost — it's static per interview, only the conversation changes. Falls back to a plain
    string when there's no context."""
    ctx = (context or "").strip()[:MAX_CONTEXT_CHARS]
    if not ctx:
        return persona
    return [
        {"type": "text", "text": persona},
        {"type": "text",
         "text": "CANDIDATE BACKGROUND (your ONLY source of facts — never go beyond it):\n\n" + ctx,
         "cache_control": {"type": "ephemeral"}},
    ]


def _coach_user(req: CoachRequest) -> str:
    return (
        f"RECENT CONVERSATION:\n{req.history or '(start of conversation)'}\n\n"
        f"INTERVIEWER JUST SAID:\n{req.transcript}\n\n"
        f"Give the candidate their answer cues now."
    )


_PREP_DIR = Path(__file__).parent / "company_prep"


_SD_TERMS = re.compile(
    r"\b(system design|design a |design an |design the |high[- ]level design|how would you (build|design|architect)"
    r"|architect|scalab|throughput|\bqps\b|\btps\b|shard|partition|replicat|load balanc|consistency|availabilit"
    r"|distributed|microservice|message queue|kafka|cache invalidat|back[- ]of[- ]the[- ]envelope|bottleneck)\b",
    re.I,
)


def _looks_system_design(transcript: str, history: str) -> bool:
    """Route system-design turns (no screenshot, so otherwise fast-model) to the stronger coder model —
    the deep follow-ups (sharding, failover, consistency) need real reasoning."""
    return bool(_SD_TERMS.search((transcript or "") + " " + (history or "")))


def _company_prep(company: str) -> str:
    """Company-specific interview prep (company_prep/<slug>.md) — loaded as extra grounding. The universal
    STAR / coding / system-design frameworks live in the coach's core prompt; this adds company specifics."""
    if not company:
        return ""
    c = company.lower()
    slugs = [re.sub(r"[^a-z0-9]+", "-", c).strip("-")]
    # keyword aliases so any name variant ("Amazon Web Services (AWS)", "sigma computing") resolves
    for kw, slug in (("amazon", "amazon"), ("aws", "amazon"), ("sigma", "sigma"), ("robinhood", "robinhood")):
        if kw in c:
            slugs.append(slug)
    for slug in slugs:
        p = _PREP_DIR / f"{slug}.md"
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")[:MAX_CONTEXT_CHARS]
            except OSError:
                pass
    return ""


def _parse_image(raw: str) -> tuple[str, str] | None:
    """Accept a data URL ('data:image/jpeg;base64,XXXX') or bare base64 → (media_type, base64_data).
    Returns None if there's nothing usable."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("data:"):
        try:
            header, data = s.split(",", 1)
            media = header[5:].split(";", 1)[0] or "image/jpeg"
            return media, data
        except ValueError:
            return None
    return "image/jpeg", s   # assume a bare base64 JPEG


def _image_blocks(image: str | None, images: list[str]) -> list[dict]:
    """Turn one screenshot, or a scroll-sequence of them, into Claude image blocks (deduped, capped)."""
    raws = [i for i in (images or []) if i] or ([image] if image else [])
    blocks = []
    for raw in raws[:8]:                       # cap so a burst can't balloon the request
        parsed = _parse_image(raw)
        if parsed:
            blocks.append({"type": "image",
                           "source": {"type": "base64", "media_type": parsed[0], "data": parsed[1]}})
    return blocks


def _coach_content(req: CoachRequest):
    """User-turn content for /coach. When a screenshot (or a scroll-sequence) is supplied, prepend it as
    Claude vision block(s) so the coach can read on-screen content (coding problem, slides) as context."""
    text = _coach_user(req)
    blocks = _image_blocks(req.image, req.images)
    if not blocks:
        return text
    multi = len(blocks) > 1
    note = (
        text + "\n\nThe attached image is the candidate's SCREEN right now." if not multi else
        text + f"\n\nThe {len(blocks)} attached images are a SCROLL-THROUGH of the candidate's screen, "
        "top to bottom (the problem is too long to fit on one screen). Read them IN ORDER and assemble "
        "the COMPLETE problem — statement, ALL constraints, and ALL examples — before solving. Do not "
        "miss a constraint or example that only appears in a later image."
    ) + (
        " If it shows a coding problem, read it and solve it fully. Write the solution in PYTHON by "
        "default — UNLESS a specific language was requested anywhere in the conversation (by the "
        "interviewer in the transcript, or the candidate), e.g. 'in Java' / 'use C++', in which case write "
        "in THAT language with its standard signature, even if the on-screen editor differs. On-screen "
        "content is live input you MAY use — it is not a candidate-fact you'd need grounded in the brief."
    )
    return [*blocks, {"type": "text", "text": note}]


def _interviewer_user(req: InterviewerRequest) -> str:
    where = f"{req.role or 'the role'}" + (f" at {req.company}" if req.company else "")
    return (
        f"You are interviewing this candidate for {where}.\n\n"
        f"CONVERSATION SO FAR:\n{req.history or '(none yet — open the interview)'}\n\n"
        f"Give your next interviewer turn (one question)."
    )


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    return {"ok": True, "coach_model": COACH_MODEL, "interviewer_model": INTERVIEWER_MODEL,
            "anthropic_key": bool(_api_key)}


@app.post("/upload-context/")
async def upload_context(file: UploadFile = File(...)) -> dict:
    """Extract plain text from an uploaded PDF/TXT/DOCX so it can be used as grounding context."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB).")
    name = (file.filename or "").lower()
    import io
    try:
        if name.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                text = "".join((p.extract_text() or "") for p in pdf.pages)
        elif name.endswith(".txt") or name.endswith(".md"):
            text = raw.decode("utf-8", errors="ignore")
        elif name.endswith(".docx"):
            import docx
            text = "\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type. Upload PDF, TXT, MD, or DOCX.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("context extraction failed")
        raise HTTPException(status_code=422, detail=f"Could not extract text: {e}")
    return {"context_text": text.strip()}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    """Transcribe an audio segment (webm/opus/wav from the browser's MediaRecorder) to text using
    local faster-whisper. This is what makes the device picker REAL — the browser captures the
    SELECTED device's stream (BlackHole, a mic, …) and sends it here, instead of relying on Web
    Speech which ignores the device and only hears the OS default input."""
    raw = await file.read()
    if not raw:
        return {"text": ""}
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio segment too large.")
    import tempfile
    from starlette.concurrency import run_in_threadpool
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(raw)
        tmp.flush()
        tmp.close()

        def _run() -> str:
            segments, _ = _whisper().transcribe(tmp.name, language="en", vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()

        text = await run_in_threadpool(_run)   # whisper is CPU-bound — keep the event loop free
    except Exception as e:
        logger.exception("transcription failed")
        raise HTTPException(status_code=422, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return {"text": text}


async def _stream(model: str, system, user: str) -> AsyncIterator[str]:
    client = _anthropic()
    async with client.messages.stream(
        model=model, max_tokens=700, system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        async for chunk in stream.text_stream:
            yield chunk


@app.post("/coach")
async def coach(req: CoachRequest) -> StreamingResponse:
    """STREAMING coach cues — tokens flush as they generate so the answer starts appearing in ~1-2s
    instead of the candidate waiting for the whole thing (critical when the interviewer just asked)."""
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript is empty.")
    coding = bool(req.image or req.images)
    # Coding (screenshot) OR a system-design discussion → strong model. Plain verbal cues stay on Haiku.
    model = CODER_MODEL if (coding or _looks_system_design(req.transcript, req.history)) else COACH_MODEL
    prep = _company_prep(req.company)
    ctx = (prep + "\n\n---\n\n" + req.context) if prep and req.context else (prep or req.context)
    content = _coach_content(req)
    system = _system(_COACH_SYSTEM, ctx)
    max_out = 1800 if coding else 700   # one solution (not the two-block bug story) → far less to generate

    async def gen() -> AsyncIterator[str]:
        client = _anthropic()
        async with client.messages.stream(
            model=model, max_tokens=max_out, system=system,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk

    return StreamingResponse(gen(), media_type="text/plain")


@app.post("/fix")
async def fix(req: WatchRequest) -> dict:
    """On-demand: read the problem + the candidate's code on screen (one frame, or a scroll-sequence)
    and return a clear complete correct solution + what was wrong. Returns {issue, guidance}."""
    blocks = _image_blocks(req.image, req.images)
    if not blocks:
        return {"issue": False, "guidance": "No screen shared — turn on 🖥️ Vision first."}
    multi = len(blocks) > 1
    ask = ("The images are a scroll-through of my screen top-to-bottom — read them in order and assemble "
           "the COMPLETE problem (all constraints and examples) first, then give the complete correct "
           "solution I can paste, plus what was wrong.") if multi else \
          ("Read the problem and my code on screen, then give the complete correct solution I can paste, "
           "plus what was wrong.")
    # Fold in company prep (e.g. Sigma = SQL/warehouse-flavoured, Python default) as a cached system block.
    prep = _company_prep(req.company)
    system = _FIX_SYSTEM if not prep else [
        {"type": "text", "text": _FIX_SYSTEM},
        {"type": "text", "text": "COMPANY CONTEXT (bias the solution's language/idioms to this if relevant):"
         "\n\n" + prep, "cache_control": {"type": "ephemeral"}},
    ]
    client = _anthropic()
    msg = await client.messages.create(
        model=CODER_MODEL, max_tokens=1200, system=system,
        messages=[{"role": "user", "content": [*blocks, {"type": "text", "text": ask}]}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    return {"issue": True, "guidance": _scrub(text) or "Couldn't read the screen — share the editor window."}


@app.post("/coach/stream")
async def coach_stream(req: CoachRequest) -> StreamingResponse:
    """Streaming variant for lowest perceived latency (raw — the scrub can't run mid-stream)."""
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript is empty.")
    return StreamingResponse(_stream(COACH_MODEL, _system(_COACH_SYSTEM, req.context), _coach_user(req)),
                             media_type="text/plain")


@app.post("/interviewer")
async def interviewer(req: InterviewerRequest) -> StreamingResponse:
    """Streaming interviewer turn for the AI-interviewer mock mode."""
    return StreamingResponse(
        _stream(INTERVIEWER_MODEL, _system(_INTERVIEWER_SYSTEM, req.context), _interviewer_user(req)),
        media_type="text/plain",
    )
