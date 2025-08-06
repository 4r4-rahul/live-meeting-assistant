import React, { useState, useRef, useEffect } from "react";
import { Button } from "./ui/button";
import { Mic, MicOff, User, Bot, Volume2 } from "lucide-react";

export default function LiveMeetingAssistant() {
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isListening, setIsListening] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: "user" | "ai"; text: string; timestamp: number }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);
  const chatBoxRef = useRef<HTMLDivElement>(null);



  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(() => {
        navigator.mediaDevices.enumerateDevices().then(devices => {
          const audioInputs = devices.filter(d => d.kind === "audioinput");
          setAudioDevices(audioInputs);
          if (audioInputs.length > 0 && !selectedDeviceId) {
            setSelectedDeviceId(audioInputs[0].deviceId);
          }
        });
      })
      .catch(() => {
        setError("Microphone permission denied. Please allow access to use this feature.");
        setToast("Microphone permission denied. Please allow access to use this feature.");
      });
    // Listen for device changes
    const handleDeviceChange = () => {
      navigator.mediaDevices.enumerateDevices().then(devices => {
        const audioInputs = devices.filter(d => d.kind === "audioinput");
        setAudioDevices(audioInputs);
      });
    };
    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, [selectedDeviceId]);
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const startRecognitionWithDevice = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      setError('Web Speech API is not supported in this browser.');
      setToast('Web Speech API is not supported in this browser.');
      return;
    }
    navigator.mediaDevices.getUserMedia({ audio: { deviceId: selectedDeviceId } }).then(stream => {
      // Placeholder for backend audio upload (feature 8)
      // You can send 'stream' to your backend here if needed

      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.onresult = async (event: any) => {
        let interim = '';
        let final = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            final += transcript;
          } else {
            interim += transcript;
          }
        }
        setInterimTranscript(interim);
        if (final.trim()) {
          setChatHistory((prev) => {
            const updated: { role: "user" | "ai"; text: string; timestamp: number }[] = [...prev, { role: "user", text: final.trim(), timestamp: Date.now() }];
            sendToConversationalQA(final.trim(), updated);
            return updated;
          });
        }
      };
      recognition.onerror = (event: any) => {
        setError(event.error || 'Speech recognition error');
        setToast(event.error || 'Speech recognition error');
        setIsListening(false);
      };
      recognitionRef.current = recognition;
      recognition.start();
      setIsListening(true);
    }).catch(err => {
      setError('Could not access selected audio device: ' + err.message);
      setToast('Could not access selected audio device: ' + err.message);
    });
  };

  const toggleMic = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      startRecognitionWithDevice();
    }
  };

  const sendToConversationalQA = async (question: string, updatedHistory: { role: "user" | "ai"; text: string }[]) => {
    setError(null);
    try {
      const response = await fetch("http://localhost:8000/conversational-qa/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: updatedHistory.map((m) => m.text).join("\n"),
          question,
          chat_history: updatedHistory
            .filter((m) => m.role === 'user' || m.role === 'ai')
            .reduce((arr, m, idx, src) => {
              if (m.role === 'user' && src[idx + 1]?.role === 'ai') {
                arr.push([m.text, src[idx + 1].text]);
              }
              return arr;
            }, [] as [string, string][]),
        }),
      });
      const data = await response.json();
      if (data.answer) {
        setChatHistory((prev) => [
          ...prev,
          { role: 'ai', text: data.answer, timestamp: Date.now() }
        ]);
      } else if (data.error) {
        setError(data.error);
        setToast(data.error);
      }
    } catch (err: any) {
      setError(err.message || 'Request failed');
      setToast(err.message || 'Request failed');
    }
  };

  const handleChatInputSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = chatInput.trim();
    if (!trimmed) return;
    let updated: { role: "user" | "ai"; text: string; timestamp: number }[] = [];
    setChatHistory((prev) => {
      updated = [...prev, { role: "user", text: trimmed, timestamp: Date.now() }];
      return updated;
    });
    sendToConversationalQA(trimmed, updated);
    setChatInput("");
  };

  return (
    <div
      className="relative flex flex-col min-h-screen bg-[#23272f] text-[#e5e7eb] transition-colors duration-300"
    >
      {/* Theme toggle button */}
      {/* No theme toggle button */}
      {/* Toast notification */}
      {toast && (
        <div className="fixed top-4 right-4 z-50 bg-green-600 text-white px-4 py-2 rounded shadow-lg animate-fade-in-out cursor-pointer" onClick={() => setToast(null)}>
          {toast}
        </div>
      )}

      {/* Audio input device selection */}
      <div className="flex items-center gap-2 pb-4 pt-8 px-6">
        <label className="font-semibold pr-2 text-[#e5e7eb]">Input Device:</label>
        <select
          value={selectedDeviceId}
          onChange={e => setSelectedDeviceId(e.target.value)}
          className="border border-[#343541] rounded px-3 py-2 min-w-[220px] bg-[#23272f] text-[#e5e7eb] shadow-sm focus:ring focus:ring-green-400 focus:outline-none"
        >
          {audioDevices.map(device => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label?.includes('BlackHole') ? '🎧 ' : device.label?.includes('Mic') ? '🎤 ' : ''}
              {device.label || `Device ${device.deviceId}`}
            </option>
          ))}
        </select>
      </div>

      {/* Animated Listening Indicator */}
      <div className="flex items-center gap-2 px-6 pb-4">
        <Button
          onClick={toggleMic}
          className={`rounded-full px-6 py-2 font-semibold shadow-md transition-colors duration-200 ${isListening ? "bg-green-600 text-white animate-pulse shadow-lg shadow-green-400/50 hover:bg-green-700" : "bg-green-600 text-white hover:bg-green-700"}`}
        >
          {isListening ? <MicOff className="mr-2 animate-pulse" /> : <Mic className="mr-2" />}
          {isListening ? "Stop Listening" : "Start Listening"}
        </Button>
        {isListening && (
          <span className="flex items-center gap-1 text-green-300 animate-pulse font-medium">
            <Volume2 className="w-5 h-5 animate-bounce" /> Listening...
          </span>
        )}
      </div>

      {/* Big chat dialogue box with sticky input */}
      <div className="flex-1 flex flex-col overflow-hidden px-2 sm:px-6 pb-4">
        <div ref={chatBoxRef} className="flex-1 overflow-y-auto border border-[#343541] rounded-xl bg-[#23272f] text-[#e5e7eb] p-4 flex flex-col space-y-3 text-base shadow-md" style={{ maxHeight: 'calc(100vh - 220px)' }}>
          {chatHistory.map((msg, idx) => (
            <div key={idx} className={msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
              <div className={
                msg.role === 'user'
                  ? 'bg-[#40414f] text-[#e5e7eb] rounded-2xl px-4 py-3 inline-block max-w-[70%] shadow flex items-end gap-2'
                  : 'bg-[#343541] text-[#e5e7eb] rounded-2xl px-4 py-3 inline-block max-w-[70%] shadow flex items-end gap-2'
              }>
                {msg.role === 'user' ? <User className="w-5 h-5 mr-1 text-blue-400" /> : <Bot className="w-5 h-5 mr-1 text-gray-400" />}
                <span>{msg.text}</span>
              </div>
            </div>
          ))}
          {isListening && interimTranscript && (
            <div className="flex justify-end">
              <div className="bg-[#40414f] text-green-300 rounded-2xl px-4 py-3 inline-block opacity-80 max-w-[70%] shadow flex items-center gap-2 animate-pulse">
                <User className="w-5 h-5 mr-1 text-blue-200" />
                <span>{interimTranscript}</span>
              </div>
            </div>
          )}
        </div>
        {/* Sticky chat input */}
        <form className="flex space-x-2 pt-4 bg-[#23272f] text-[#e5e7eb] sticky bottom-0 z-10 rounded-b-xl border-t border-[#343541]" onSubmit={handleChatInputSend}>
          <input
            type="text"
            className="flex-1 border border-[#343541] rounded-full px-4 py-2 text-base shadow-sm focus:ring focus:ring-green-400 bg-[#343541] text-[#e5e7eb] focus:outline-none"
            placeholder={isListening ? "Mic is active..." : "Type your message..."}
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
            disabled={isListening}
            aria-label="Type your message"
          />
          <Button type="submit" disabled={isListening || !chatInput.trim()} className="rounded-full px-6 py-2 font-semibold bg-green-600 text-white hover:bg-green-700 transition-colors duration-200 disabled:opacity-60">
            Send
          </Button>
        </form>
      </div>
    </div>
  );
}