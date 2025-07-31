import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import SearchButton from "./searchButton"; // adjust path if needed

const API = "http://localhost:5000";

interface Message {
  type: "user" | "bot";
  text: string;
}

export default function App() {
  const [pdfs, setPdfs] = useState<string[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<string[]>([]); // Store selected PDFs
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [warning, setWarning] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Fetch the list of PDFs from the backend
    axios.get(`${API}/list-pdfs`).then(res => {
      const pdfList = res.data.pdfs;
      setPdfs(pdfList);

      // Pre-select all PDFs by default
      setSelected(pdfList); 
    });
  }, []);

  useEffect(() => {
    // Scroll to the bottom when new messages arrive
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const upload = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);

    try {
      const res = await axios.post(`${API}/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      alert("Upload successful");
      console.log("Upload result:", res.data);

      // Update PDF list and pre-select all
      const updated = await axios.get(`${API}/list-pdfs`);
      setPdfs(updated.data.pdfs);
      setSelected(updated.data.pdfs); // Select all after upload
      setFile(null);
    } catch (err: any) {
      if (err.response?.status === 409) {
        alert("A file with this name already exists. Please rename the file before uploading.");
      } else {
        console.error("Upload failed:", err);
        alert("Upload failed");
      }
    }
  };

  const ask = async () => {
    if (!selected.length) {
      setWarning("Please select at least one PDF before chatting.");
      return;
    }
    if (!question.trim() || loading) return;

    const sentQuestion = question;
    setMessages(prev => [...prev, { type: "user", text: sentQuestion }]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await axios.post(`${API}/query`, {
        filenames: selected,
        question: sentQuestion,
      });

      setMessages(prev => [...prev, { type: "bot", text: res.data.response }]);
    } catch (err) {
      console.error("Query failed:", err);
      setMessages(prev => [
        ...prev,
        { type: "bot", text: "Something went wrong. Please try again." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-64 bg-white p-4 shadow-md flex flex-col">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">📄 PDFs</h2>
          <SearchButton />
         
        </div>

      
        {file && (
          <div className="flex items-center justify-between text-sm text-gray-600 mb-4">
            <span className="truncate max-w-[150px]">📎 {file.name}</span>
            <button
              onClick={() => setFile(null)}
              className="text-red-500 text-xs hover:underline"
            >
              ❌ Remove
            </button>
          </div>
        )}

        {/* Hide PDF list since all PDFs are selected by default */}
      </div>

      {/* Main Chat UI */}
      <div className="flex-1 flex flex-col">
        <div className="p-4 border-b bg-white shadow-sm">
          <h1 className="text-2xl font-bold">DocumentMind</h1>
         
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50 flex flex-col">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`px-4 py-2 rounded-lg whitespace-pre-wrap break-words ${
                msg.type === "user"
                  ? "self-end bg-blue-500 text-white max-w-[75%]"
                  : "self-start bg-gray-300 text-black max-w-[75%]"
              }`}
            >
              {Array.isArray(msg.text)
                ? msg.text.map((line, i) => <p key={i} className="mb-2">{line}</p>)
                : <p>{msg.text}</p>}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {warning && (
          <div className="mx-4 mb-2 bg-red-100 border border-red-400 text-red-700 px-4 py-2 rounded">
            {warning}
          </div>
        )}

        {/* Input Area */}
        <div className="p-4 border-t bg-white flex items-center gap-2">
          <div
            className="flex-1"
            onClick={() => {
              if (!selected.length) {
                setWarning("⚠️ Please select a PDF before chatting.");
              }
            }}
          >
            <input
              type="text"
              placeholder="Ask something..."
              value={question}
              onChange={e => setQuestion(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-4 py-2"
              onKeyDown={e => e.key === "Enter" && ask()}
              disabled={!selected.length || loading}
            />
          </div>
          <button
            onClick={ask}
            className="bg-green-500 text-white px-4 py-2 rounded-lg disabled:opacity-50"
            disabled={!question.trim() || loading}
          >
            {loading ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}