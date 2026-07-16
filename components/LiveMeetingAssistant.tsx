import React, { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "./ui/button";
import { Mic, MicOff, User, Bot, Volume2, Lightbulb, MessageSquare, MonitorSpeaker, Monitor, Eye, AlertTriangle, Wrench, BookOpen } from "lucide-react";

// Cheap 32-bit string hash — used to drop duplicate frames during a scroll-capture burst.
const hashStr = (s: string): number => {
  let h = 0;
  for (let i = 0; i < s.length; i++) { h = (h << 5) - h + s.charCodeAt(i); h |= 0; }
  return h;
};

// LMA's own backend (moved to :8100 to avoid colliding with AutonomousHire's API on :8000).
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";
// AutonomousHire (Job_Finder) API — used to pull the prep brief when deep-linked with ?company=.
const JOBFINDER_API = process.env.NEXT_PUBLIC_JOBFINDER_API ?? "http://localhost:8000/api/v1";

type Mode = "coach" | "interviewer";
type Msg = { role: "user" | "ai" | "watch"; text: string; timestamp: number };

export default function LiveMeetingAssistant() {
  const [mode, setMode] = useState<Mode>("coach");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [contextText, setContextText] = useState<string>("");
  const [company, setCompany] = useState("");
  const [role, setRole] = useState("");
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isListening, setIsListening] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [history, setHistory] = useState<Msg[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const [level, setLevel] = useState(0);        // live input level (0..1) — shows if the device has signal
  const streamRef = useRef<MediaStream | null>(null);
  const displayStreamRef = useRef<MediaStream | null>(null);   // full getDisplayMedia stream (holds the share)
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const listeningRef = useRef(false);          // mirror of isListening for use inside callbacks
  const chatBoxRef = useRef<HTMLDivElement>(null);
  const pendingRef = useRef("");               // transcript accumulating for the CURRENT turn (not sent yet)
  const utteranceStartRef = useRef(0);         // when the current turn began (for the max-length safety flush)
  // ── Screen vision — a SEPARATE screen share whose video frames get attached to coach calls, so the
  // coach can "see" a coding problem / slides on screen. Independent of audio (entire-screen capture
  // has no audio on macOS, but its video works fine for this).
  const [visionOn, setVisionOn] = useState(false);
  const visionStreamRef = useRef<MediaStream | null>(null);
  const visionVideoRef = useRef<HTMLVideoElement | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [fixing, setFixing] = useState(false); // an on-demand /fix (diff) call is in flight
  const SEGMENT_MS = 2000;                      // record/transcribe in 2s chunks — faster end-of-turn detection
  const MAX_UTTERANCE_MS = 45000;               // safety: send an unbroken monologue after this long even w/o a pause

  // Auto-dismiss toasts.
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3500);
    return () => clearTimeout(t);
  }, [toast]);

  // Mount-once: permission + device list + change listener. (Was incorrectly keyed on
  // selectedDeviceId, which re-ran getUserMedia on every dropdown change.)
  useEffect(() => {
    const enumerate = () =>
      navigator.mediaDevices.enumerateDevices().then((devices) => {
        const inputs = devices.filter((d) => d.kind === "audioinput");
        setAudioDevices(inputs);
        setSelectedDeviceId((cur) => cur || inputs[0]?.deviceId || "");
      });
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        // Release this probe stream immediately — holding the mic open blocks capture later.
        stream.getTracks().forEach((t) => t.stop());
        return enumerate();
      })
      .catch(() => setToast("Microphone permission denied — allow access to use live capture."));
    navigator.mediaDevices.addEventListener("devicechange", enumerate);
    return () => navigator.mediaDevices.removeEventListener("devicechange", enumerate);
  }, []);

  // Deep-link from AutonomousHire: ?company=X[&role=Y][&mode=coach|interviewer] → preload the prep
  // brief from Job_Finder's /insights/prep so the user never re-uploads anything.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const co = params.get("company");
    const r = params.get("role");
    const m = params.get("mode");
    if (m === "interviewer") setMode("interviewer");
    if (r) setRole(r);
    if (!co) return;
    setCompany(co);
    // Pull the FULL interview grounding (entire experience bank + per-company brief + profile +
    // role) from AutonomousHire — not just the one brief — so the coach can reach any real story.
    const qs = new URLSearchParams({ company: co, ...(r ? { role: r } : {}) }).toString();
    fetch(`${JOBFINDER_API}/mock/context?${qs}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error("no context"))))
      .then((data: { context: string; found_brief: boolean }) => {
        setContextText(data.context || "");
        setToast(
          `Loaded ${co} background${data.found_brief ? " + prep brief" : ""} — coach is grounded in your full experience bank.`,
        );
      })
      .catch(() => setToast(`Couldn't load ${co} context — is AutonomousHire running? You can upload a brief instead.`));
  }, []);

  useEffect(() => {
    if (chatBoxRef.current) chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
  }, [history, interimTranscript]);

  // Clean up capture on unmount so nothing fires setState (or holds the mic) after teardown.
  useEffect(() => () => {
    listeningRef.current = false;
    try { recorderRef.current?.stop(); } catch { /* */ }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    displayStreamRef.current?.getTracks().forEach((t) => t.stop());
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    audioCtxRef.current?.close().catch(() => {});
    visionStreamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  const recentHistory = useCallback(
    () => history.slice(-8).filter((m) => m.role !== "watch").map((m) => `${m.role === "user" ? "Them" : "Coach"}: ${m.text}`).join("\n"),
    [history],
  );

  // ── Backend calls ──────────────────────────────────────────────────────────
  const askCoach = useCallback(
    async (transcript: string, image?: string | null, images?: string[]) => {
      setError(null);
      setIsThinking(true);
      try {
        const res = await fetch(`${API_BASE}/coach`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transcript, context: contextText, history: recentHistory(), image: image || null, images: images || [], company }),
        });
        if (!res.ok || !res.body) {
          let detail = "Coach error";
          try { const j = await res.json(); detail = j.detail || j.error || detail; } catch { /* not json */ }
          throw new Error(detail);
        }
        // Stream tokens in so the answer starts showing in ~1-2s instead of waiting for the whole thing.
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let acc = "";
        const at = Date.now();
        setHistory((prev) => [...prev, { role: "ai", text: "", timestamp: at }]);
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          acc += decoder.decode(value, { stream: true });
          setHistory((prev) => prev.map((m) => (m.timestamp === at && m.role === "ai" ? { ...m, text: acc } : m)));
        }
        if (!acc.trim()) setHistory((prev) => prev.map((m) => (m.timestamp === at ? { ...m, text: "(no suggestion)" } : m)));
      } catch (e: any) {
        setError(e.message || "Request failed");
        setToast(e.message || "Request failed");
      } finally {
        setIsThinking(false);
      }
    },
    [contextText, recentHistory, company],
  );

  // Streaming interviewer turn — renders tokens as they arrive.
  const askInterviewer = useCallback(
    async (lastAnswer: string) => {
      setError(null);
      setIsThinking(true);
      const convo = [...history, ...(lastAnswer ? [{ role: "user" as const, text: lastAnswer, timestamp: Date.now() }] : [])];
      const histStr = convo.map((m) => `${m.role === "user" ? "Candidate" : "Interviewer"}: ${m.text}`).join("\n");
      try {
        const res = await fetch(`${API_BASE}/interviewer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ company, role, context: contextText, history: histStr }),
        });
        if (!res.ok || !res.body) throw new Error("Interviewer error");
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let acc = "";
        const at = Date.now();
        setHistory((prev) => [...prev, { role: "ai", text: "", timestamp: at }]);
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          acc += decoder.decode(value, { stream: true });
          setHistory((prev) => prev.map((m) => (m.timestamp === at && m.role === "ai" ? { ...m, text: acc } : m)));
        }
      } catch (e: any) {
        setError(e.message || "Request failed");
        setToast(e.message || "Request failed");
      } finally {
        setIsThinking(false);
      }
    },
    [history, company, role, contextText],
  );

  // ── Screen vision ────────────────────────────────────────────────────────────
  // Grab the current screen frame as a downscaled JPEG data URL (or null if vision is off / not
  // ready). Downscale keeps it to ~1-1.5k image tokens per call.
  const captureFrame = useCallback((): string | null => {
    const video = visionVideoRef.current;
    const canvas = frameCanvasRef.current;
    if (!visionOn || !video || !canvas || !video.videoWidth) return null;
    // Cap at 1568px (Anthropic downscales beyond that anyway) and use HIGH JPEG quality — code is
    // small text, and quality 0.6 blurred out fine syntax (a missing ':' or ')'). 0.92 keeps it crisp.
    const maxW = 1568;
    const scale = Math.min(1, maxW / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    try { return canvas.toDataURL("image/jpeg", 0.92); } catch { return null; }
  }, [visionOn]);

  // Capture a sequence of DISTINCT frames over `durationMs` (deduping identical ones) — the user
  // scrolls through a long problem while this runs, so the frames together cover the whole statement.
  const captureBurst = useCallback(async (durationMs: number, intervalMs: number): Promise<string[]> => {
    const frames: string[] = [];
    const seen = new Set<number>();
    const start = Date.now();
    while (Date.now() - start < durationMs && frames.length < 8) {
      const f = captureFrame();
      if (f) { const h = hashStr(f); if (!seen.has(h)) { seen.add(h); frames.push(f); } }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    return frames;
  }, [captureFrame]);

  const stopVision = useCallback(() => {
    visionStreamRef.current?.getTracks().forEach((t) => t.stop());
    visionStreamRef.current = null;
    if (visionVideoRef.current) visionVideoRef.current.srcObject = null;
    setVisionOn(false);
  }, []);

  const startVision = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        // Ask for a crisp source so downscaled frames stay legible enough to read code/syntax.
        video: { width: { ideal: 2560 }, height: { ideal: 1440 }, frameRate: { ideal: 5 } },
        audio: false,
      });
      visionStreamRef.current = stream;
      const video = visionVideoRef.current;
      if (video) { video.srcObject = stream; await video.play().catch(() => {}); }
      // If the user hits Chrome's "Stop sharing", tear down cleanly.
      stream.getVideoTracks()[0]?.addEventListener("ended", stopVision);
      setVisionOn(true);
      setToast("Screen vision on — the coach will factor in what's on screen.");
    } catch {
      setToast("Screen share cancelled — vision stays off.");
    }
  }, [stopVision]);

  const toggleVision = () => (visionOn ? stopVision() : startVision());

  // Route a finalized utterance / typed input to the active mode. In coach mode, attach the current
  // screen frame so the coach sees what the candidate sees.
  const dispatch = useCallback(
    (text: string) => {
      const t = text.trim();
      if (!t) return;
      setHistory((prev) => [...prev, { role: "user", text: t, timestamp: Date.now() }]);
      if (mode === "coach") askCoach(t, captureFrame());
      else askInterviewer(t);
    },
    [mode, askCoach, askInterviewer, captureFrame],
  );

  // On-demand "fix my code": snapshot the screen ONCE and return the smallest change as a diff
  // (present code → corrected code). No polling — the user clicks when they want it.
  const fixMyCode = useCallback(async () => {
    const frame = captureFrame();
    if (!frame) { setToast("Turn on screen vision first (🖥️ Share screen)."); return; }
    setFixing(true);
    try {
      const res = await fetch(`${API_BASE}/fix`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frame, company }),
      });
      const data = await res.json();
      const g = (data.guidance || "").trim();
      setHistory((prev) => [...prev, { role: "ai", text: g || "The code looks correct.", timestamp: Date.now() }]);
    } catch (e: any) {
      setToast(e?.message || "Couldn't analyse the screen.");
    } finally {
      setFixing(false);
    }
  }, [captureFrame, company]);

  // Manual pure-vision test: coach purely on what's on screen, no interviewer line needed.
  const askAboutScreen = useCallback(() => {
    const frame = captureFrame();
    if (!frame) { setToast("Turn on screen vision first (🖥️ Share screen)."); return; }
    const prompt = "Read what's on my screen and coach me on it — if it's a coding problem, give me the approach.";
    setHistory((prev) => [...prev, { role: "user", text: "🖥️ (what's on my screen?)", timestamp: Date.now() }]);
    askCoach(prompt, frame);
  }, [captureFrame, askCoach]);

  // For a LONG problem that doesn't fit on one screen: capture a burst of frames while the user scrolls,
  // then solve from ALL of them so no constraint/example is missed.
  const [scrolling, setScrolling] = useState(false);
  const solveScroll = useCallback(async () => {
    if (!captureFrame()) { setToast("Turn on screen vision first (🖥️ Share screen)."); return; }
    setScrolling(true);
    setToast("📖 Scroll slowly through the WHOLE problem now — capturing for ~7s…");
    try {
      const frames = await captureBurst(7000, 800);
      if (!frames.length) { setToast("No frames captured — is the screen still shared?"); return; }
      setToast(`Captured ${frames.length} frames — assembling the full problem…`);
      setHistory((prev) => [...prev, { role: "user", text: `📖 (solve the full problem — ${frames.length} screens)`, timestamp: Date.now() }]);
      await askCoach("Solve the full coding problem shown across these scroll screenshots.", undefined, frames);
    } finally {
      setScrolling(false);
    }
  }, [captureFrame, captureBurst, askCoach]);

  // ── Audio capture → local Whisper ────────────────────────────────────────────
  // We capture the SELECTED device's stream and transcribe server-side. This is the fix for the
  // device picker: getUserMedia({deviceId}) actually honors the choice (BlackHole, a mic, …),
  // unlike Web Speech which ignored it and only heard the OS default input.
  // A chunk that transcribes to nothing means ~SEGMENT_MS of silence — i.e. the speaker paused. We
  // also swallow the handful of tokens Whisper hallucinates on near-silence so a stray "you" / "thank
  // you" between sentences doesn't count as speech and keep the turn open forever.
  const HALLUCINATED_SILENCE = new Set(["you", "thanks", "thankyou", "thanksforwatching"]);
  const isSilent = (text: string) => {
    const t = text.toLowerCase().replace(/[^a-z]/g, "");
    return t === "" || HALLUCINATED_SILENCE.has(t);
  };

  // Send the accumulated turn to the coach/interviewer as ONE message, then reset the buffer.
  const flushUtterance = () => {
    const t = pendingRef.current.trim();
    pendingRef.current = "";
    utteranceStartRef.current = 0;
    setInterimTranscript("");
    if (t) dispatch(t);
  };

  // Transcribe one segment and fold it into the current turn. We DON'T send on every chunk — we wait
  // for the speaker to finish (a silent chunk) so the model sees a complete question, not a fragment.
  const transcribeSegment = async (blob: Blob) => {
    let text = "";
    try {
      const fd = new FormData();
      fd.append("file", blob, "segment.webm");
      const res = await fetch(`${API_BASE}/transcribe`, { method: "POST", body: fd });
      const data = await res.json();
      text = (data.text || "").trim();
    } catch {
      return; // drop a failed segment, keep listening
    }
    if (isSilent(text)) {
      // End of a turn: if we've buffered anything, send it now. Bare silence is a no-op.
      if (pendingRef.current.trim()) flushUtterance();
      return;
    }
    // Still talking — accumulate and show it building live; hold off on sending.
    pendingRef.current = `${pendingRef.current} ${text}`.trim();
    if (!utteranceStartRef.current) utteranceStartRef.current = Date.now();
    setInterimTranscript(pendingRef.current);
    if (Date.now() - utteranceStartRef.current >= MAX_UTTERANCE_MS) flushUtterance();
  };

  const pickMime = () => {
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
    return types.find((t) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) || "";
  };

  // Record one ~5s segment, transcribe it, then loop while still listening.
  const recordSegment = () => {
    const stream = streamRef.current;
    if (!stream || !listeningRef.current) return;
    const mime = pickMime();
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    recorderRef.current = rec;
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = async () => {
      if (chunks.length) await transcribeSegment(new Blob(chunks, { type: rec.mimeType }));
      if (listeningRef.current) recordSegment();
      else { flushUtterance(); setIsListening(false); }   // stopped: send whatever's left buffered
    };
    rec.start();
    setTimeout(() => { if (rec.state !== "inactive") rec.stop(); }, SEGMENT_MS);
  };

  // Live input-level meter (Web Audio) — proves the SELECTED device actually has signal. If you pick
  // BlackHole and this stays flat while audio plays, the OS isn't routing output into BlackHole yet.
  const startMeter = (stream: MediaStream) => {
    try {
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new Ctx();
      audioCtxRef.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const buf = new Uint8Array(analyser.frequencyBinCount);
      let lastEmit = 0, lastQ = -1;
      const tick = (now: number) => {
        // Throttle to ~12fps AND only re-render when the quantized level actually changes — the raw
        // meter runs at 60fps and re-rendering the whole control row that fast made the buttons jitter.
        if (now - lastEmit > 80) {
          analyser.getByteTimeDomainData(buf);
          let sum = 0;
          for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
          const q = Math.round(Math.min(1, Math.sqrt(sum / buf.length) * 4) * 20) / 20; // 5% steps
          if (q !== lastQ) { lastQ = q; setLevel(q); }
          lastEmit = now;
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch { /* metering is best-effort */ }
  };

  const stopMeter = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    setLevel(0);
  };

  // Shared: given an audio stream from any source, start the record→transcribe loop + level meter.
  const beginCapture = (stream: MediaStream) => {
    streamRef.current = stream;
    listeningRef.current = true;
    pendingRef.current = "";           // fresh turn buffer for this session
    utteranceStartRef.current = 0;
    setInterimTranscript("");
    setIsListening(true);
    setError(null);
    startMeter(stream);
    recordSegment();
    // If the user ends the share (Chrome's "Stop sharing" bar), tear down cleanly.
    stream.getAudioTracks()[0]?.addEventListener("ended", stopListening);
  };

  // Source A — a selected input device (your mic, or BlackHole if you've routed audio into it).
  const startListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        // Disable Chrome's voice DSP — echo-cancel / noise-suppress / auto-gain are tuned for a
        // person's mic and will GATE/mangle a virtual-loopback (BlackHole) system-audio signal to
        // near-silence. We want the raw device audio.
        audio: {
          ...(selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : {}),
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      beginCapture(stream);
    } catch {
      setToast("Couldn't open the selected audio device — try a different one in the dropdown.");
    }
  };

  // Source B — capture a browser TAB's audio directly (YouTube, Google Meet in a browser). No
  // BlackHole, no system routing: Chrome hands us the tab's audio via the screen-share picker.
  const startTabCapture = async () => {
    try {
      const display = await navigator.mediaDevices.getDisplayMedia({
        video: true,                       // Chrome requires a video request to show the picker…
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const audioTracks = display.getAudioTracks();
      if (!audioTracks.length) {
        display.getTracks().forEach((t) => t.stop());
        setToast('No tab audio — in the picker choose the "Chrome Tab" and tick "Share tab audio".');
        return;
      }
      // Keep the FULL display stream alive (its video track is what holds the share open) and record
      // from an audio-ONLY clone — stopping the video track or feeding video to the recorder both
      // break capture. End cleanly if the user hits Chrome's "Stop sharing".
      displayStreamRef.current = display;
      display.getVideoTracks()[0]?.addEventListener("ended", stopListening);
      setToast("Capturing tab audio ✓");
      beginCapture(new MediaStream(audioTracks));
    } catch {
      setToast("Tab capture cancelled.");
    }
  };

  const stopListening = () => {
    listeningRef.current = false;
    try { recorderRef.current?.stop(); } catch { /* */ }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    displayStreamRef.current?.getTracks().forEach((t) => t.stop());   // ends the tab share
    streamRef.current = null;
    displayStreamRef.current = null;
    stopMeter();
    setIsListening(false);
  };

  const toggleMic = () => (isListening ? stopListening() : startListening());

  // ── Handlers ─────────────────────────────────────────────────────────────────
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setContextText("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/upload-context/`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.detail || data.error || "Upload failed");
      setContextText(data.context_text || "");
      setToast("Context loaded — answers will be grounded in it.");
    } catch (err: any) {
      setUploadError(err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleChatSend = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const t = chatInput.trim();
    if (!t) return;
    dispatch(t);
    setChatInput("");
  };

  const startInterview = () => askInterviewer("");

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="relative flex flex-col min-h-screen bg-[#23272f] text-[#e5e7eb]">
      {toast && (
        <div className="fixed top-4 right-4 z-50 bg-emerald-600 text-white px-4 py-2 rounded shadow-lg cursor-pointer" onClick={() => setToast(null)}>
          {toast}
        </div>
      )}

      {/* Mode toggle */}
      <div className="flex items-center gap-2 px-6 pt-6">
        <button
          onClick={() => setMode("coach")}
          className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${mode === "coach" ? "bg-sky-600 text-white" : "bg-[#343541] text-slate-300"}`}
        >
          <Lightbulb className="w-4 h-4" /> Live Coach
        </button>
        <button
          onClick={() => setMode("interviewer")}
          className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${mode === "interviewer" ? "bg-sky-600 text-white" : "bg-[#343541] text-slate-300"}`}
        >
          <MessageSquare className="w-4 h-4" /> AI Interviewer
        </button>
        <span className="text-xs text-slate-500 ml-2">
          {mode === "coach"
            ? "Transcribes the interviewer → suggests grounded answers."
            : "Plays the interviewer for a mock — asks & follows up."}
        </span>
      </div>

      {/* Context upload */}
      <div className="flex items-center gap-3 px-6 pt-4 pb-2">
        <label className="font-semibold pr-2">Prep brief / résumé:</label>
        <input
          type="file"
          accept=".pdf,.txt,.md,.docx"
          onChange={handleFileUpload}
          disabled={uploading}
          className="block text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-sky-600 file:text-white hover:file:bg-sky-700"
        />
        {uploading && <span className="text-sky-400 animate-pulse">Uploading…</span>}
        {uploadError && <span className="text-red-400 ml-2">{uploadError}</span>}
        {contextText && <span className="text-emerald-400 text-sm">✓ {contextText.length.toLocaleString()} chars loaded</span>}
      </div>

      {/* Interviewer-mode role inputs */}
      {mode === "interviewer" && (
        <div className="flex items-center gap-2 px-6 pb-2">
          <input value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Company"
            className="border border-[#343541] rounded px-3 py-2 bg-[#23272f] text-sm w-40" />
          <input value={role} onChange={(e) => setRole(e.target.value)} placeholder="Role"
            className="border border-[#343541] rounded px-3 py-2 bg-[#23272f] text-sm w-56" />
          <Button onClick={startInterview} disabled={isThinking} className="rounded-full px-4 py-2 bg-sky-600 text-white hover:bg-sky-700">
            Start interview
          </Button>
        </div>
      )}

      {/* Audio device selection */}
      <div className="flex items-center gap-2 pb-3 pt-4 px-6">
        <label className="font-semibold pr-2">Input device:</label>
        <select
          value={selectedDeviceId}
          onChange={(e) => setSelectedDeviceId(e.target.value)}
          className="border border-[#343541] rounded px-3 py-2 min-w-[220px] bg-[#23272f] text-[#e5e7eb]"
        >
          {audioDevices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label?.includes("BlackHole") ? "🎧 " : d.label?.includes("Mic") ? "🎤 " : ""}
              {d.label || `Device ${d.deviceId}`}
            </option>
          ))}
        </select>
        <span className="text-xs text-slate-400 ml-1">
          Captures the <b>selected</b> device. For coach mode pick <b>BlackHole</b> (route the meeting's
          audio to it); to mock-practice with your voice, pick your mic.
        </span>
      </div>

      {/* Listen control */}
      <div className="flex items-center gap-2 px-6 pb-4">
        <Button
          onClick={toggleMic}
          className={`rounded-full px-6 py-2 font-semibold ${isListening ? "bg-emerald-600 text-white animate-pulse" : "bg-sky-600 text-white hover:bg-sky-700"}`}
        >
          {isListening ? <MicOff className="mr-2" /> : <Mic className="mr-2" />}
          {isListening ? "Stop Listening" : "Start Listening"}
        </Button>
        {!isListening && (
          <Button
            onClick={startTabCapture}
            title="Capture a browser tab's audio (YouTube, Google Meet) — no BlackHole needed"
            className="rounded-full px-5 py-2 font-semibold bg-violet-600 text-white hover:bg-violet-700"
          >
            <MonitorSpeaker className="mr-2 w-4 h-4" /> Capture tab audio
          </Button>
        )}
        {/* Screen vision — independent of audio; entire-screen capture works for video on macOS. */}
        <Button
          onClick={toggleVision}
          title="Share your screen so the coach can SEE a coding problem / slides and coach on them"
          className={`rounded-full px-5 py-2 font-semibold ${visionOn ? "bg-amber-600 text-white animate-pulse hover:bg-amber-700" : "bg-slate-700 text-slate-100 hover:bg-slate-600"}`}
        >
          <Monitor className="mr-2 w-4 h-4" /> {visionOn ? "Vision on — stop" : "Share screen"}
        </Button>
        {visionOn && mode === "coach" && (
          <Button
            onClick={askAboutScreen}
            disabled={isThinking}
            title="Coach purely on what's on screen right now"
            className="rounded-full px-5 py-2 font-semibold bg-amber-500 text-white hover:bg-amber-600 disabled:opacity-60"
          >
            <Eye className="mr-2 w-4 h-4" /> Ask about screen
          </Button>
        )}
        {visionOn && mode === "coach" && (
          <Button
            onClick={solveScroll}
            disabled={isThinking || scrolling}
            title="For a long problem: click, then scroll through the whole thing — it reads every part and solves"
            className="rounded-full px-5 py-2 font-semibold bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-60"
          >
            <BookOpen className="mr-2 w-4 h-4" /> {scrolling ? "Scroll now…" : "Read full problem"}
          </Button>
        )}
        {visionOn && mode === "coach" && (
          <Button
            onClick={fixMyCode}
            disabled={fixing}
            title="Snapshot the screen once and show the diff to fix the code on it"
            className="rounded-full px-4 py-2 font-medium text-sm bg-amber-700/80 text-amber-100 hover:bg-amber-700 disabled:opacity-60"
          >
            <Wrench className="mr-1.5 w-4 h-4" /> {fixing ? "Checking…" : "Fix my code"}
          </Button>
        )}
        {isThinking && <span className="text-sky-300 animate-pulse text-sm">thinking…</span>}
      </div>

      {/* Audio level + status live on their OWN line (not in the button row) so the meter's flicker can
          never reflow / jitter the buttons above it. Inline pixel widths = guaranteed fixed, no reflow. */}
      {isListening && (
        <div className="flex items-center gap-2 px-6 pb-3 -mt-2 text-emerald-300 font-medium">
          <Volume2 className="w-5 h-5 shrink-0" /> Listening
          <div className="h-2.5 bg-slate-700 rounded-full overflow-hidden shrink-0 ml-1" style={{ width: 128 }}>
            <div className={`h-full ${level > 0.02 ? "bg-emerald-400" : "bg-slate-600"}`} style={{ width: `${Math.round(level * 100)}%` }} />
          </div>
          <span className="text-xs text-slate-500 whitespace-nowrap overflow-hidden shrink-0" style={{ width: 220 }}>
            {level > 0.02 ? `signal ✓ · every ${SEGMENT_MS / 1000}s` : "no signal — route audio here"}
          </span>
        </div>
      )}

      {/* Hidden capture surfaces for screen vision (kept off-screen but rendered so the video decodes). */}
      <video ref={visionVideoRef} autoPlay muted playsInline style={{ position: "fixed", left: -10000, top: 0, width: 320, height: 180 }} />
      <canvas ref={frameCanvasRef} style={{ display: "none" }} />

      {/* Conversation */}
      <div className="flex-1 flex flex-col overflow-hidden px-2 sm:px-6 pb-4">
        <div ref={chatBoxRef} className="flex-1 overflow-y-auto border border-[#343541] rounded-xl bg-[#23272f] p-4 flex flex-col space-y-3" style={{ maxHeight: "calc(100vh - 360px)" }}>
          {history.map((msg, idx) => (
            <div key={idx} className={msg.role === "user" ? "flex justify-end" : "flex justify-start"}>
              <div className={`rounded-2xl px-4 py-3 inline-block max-w-[75%] shadow flex items-start gap-2 ${
                msg.role === "user" ? "bg-[#40414f]"
                : msg.role === "watch" ? "bg-amber-500/10 border border-amber-500/40 text-amber-200"
                : "bg-[#2d2f3a] border border-sky-700/40"}`}>
                {msg.role === "user" ? <User className="w-5 h-5 mt-0.5 text-blue-400 shrink-0" />
                 : msg.role === "watch" ? <AlertTriangle className="w-5 h-5 mt-0.5 text-amber-400 shrink-0" />
                 : <Bot className="w-5 h-5 mt-0.5 text-sky-400 shrink-0" />}
                <span className="whitespace-pre-wrap">{msg.role === "watch" ? <><b>Heads up:</b> {msg.text}</> : msg.text}</span>
              </div>
            </div>
          ))}
          {isListening && interimTranscript && (
            <div className="flex justify-end">
              <div className="bg-[#40414f] text-emerald-300 rounded-2xl px-4 py-3 inline-block opacity-80 max-w-[75%] animate-pulse flex flex-col">
                <div className="flex items-center gap-2">
                  <User className="w-5 h-5 text-blue-200" />
                  <span>{interimTranscript}</span>
                </div>
                <span className="text-[11px] text-emerald-400/70 mt-1 self-end">listening… sends when you pause</span>
              </div>
            </div>
          )}
        </div>
        <form className="flex space-x-2 pt-4 sticky bottom-0" onSubmit={handleChatSend}>
          <input
            type="text"
            className="flex-1 border border-[#343541] rounded-full px-4 py-2 bg-[#343541] text-[#e5e7eb] focus:outline-none focus:ring focus:ring-sky-400"
            placeholder={mode === "coach" ? "Paste/type what the interviewer asked…" : "Type your answer…"}
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            aria-label="Type your message"
          />
          <Button type="submit" disabled={!chatInput.trim() || isThinking} className="rounded-full px-6 py-2 bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-60">
            Send
          </Button>
        </form>
        {error && <div className="text-red-400 text-sm pt-1">{error}</div>}
      </div>
    </div>
  );
}
