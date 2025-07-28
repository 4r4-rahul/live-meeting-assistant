
import React, { useState, useRef, useEffect } from "react";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Mic, MicOff } from "lucide-react";

export default function LiveMeetingAssistant() {
  const [isListening, setIsListening] = useState(false);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: "user" | "ai"; text: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);


  // Start/stop Web Speech API recognition
  const toggleMic = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        setError('Web Speech API is not supported in this browser.');
        return;
      }
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
            const updated: { role: "user" | "ai"; text: string }[] = [...prev, { role: "user", text: final.trim() }];
            sendToConversationalQA(final.trim(), updated);
            return updated;
          });
        }
      };
      recognition.onerror = (event: any) => {
        setError(event.error || 'Speech recognition error');
      };
      recognitionRef.current = recognition;
      recognition.start();
      setIsListening(true);
    }
  };

  // Send question and chat history to backend

  const chatBoxRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when chatHistory changes
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [chatHistory]);

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
        setChatHistory((prev) => {
          // Only add AI response if last message is user
          if (prev.length > 0 && prev[prev.length - 1].role === 'user') {
            return [...prev, { role: 'ai', text: data.answer }];
          }
          return prev;
        });
      } else if (data.error) {
        setError(data.error);
      }
    } catch (err: any) {
      setError(err.message || 'Request failed');
    }
  };




  return (
    <div className="p-4 space-y-4">
      <Button
        onClick={toggleMic}
        className={
          isListening
            ? "bg-red-600 text-white animate-pulse shadow-lg shadow-red-400/50 hover:bg-red-700"
            : ""
        }
      >
        {isListening ? <MicOff className="mr-2" /> : <Mic className="mr-2" />}
        {isListening ? "Stop Listening" : "Start Listening"}
      </Button>

      {error && (
        <div className="text-red-600 font-bold p-2">Error: {error}</div>
      )}

      {/* Big chat dialogue box */}
      <div ref={chatBoxRef} className="w-full h-[400px] overflow-y-auto border rounded bg-white p-4 flex flex-col space-y-2 text-base" style={{ maxHeight: 500 }}>
        {chatHistory.map((msg, idx) => (
          <div key={idx} className={msg.role === 'user' ? 'text-right' : 'text-left'}>
            <span className={msg.role === 'user' ? 'bg-blue-100 text-blue-900 rounded-lg px-3 py-2 inline-block' : 'bg-gray-100 text-gray-900 rounded-lg px-3 py-2 inline-block'}>
              {msg.text}
            </span>
          </div>
        ))}
        {isListening && interimTranscript && (
          <div className="text-right">
            <span className="bg-blue-50 text-blue-400 rounded-lg px-3 py-2 inline-block opacity-70">{interimTranscript}</span>
          </div>
        )}
      </div>
    </div>
  );
}
