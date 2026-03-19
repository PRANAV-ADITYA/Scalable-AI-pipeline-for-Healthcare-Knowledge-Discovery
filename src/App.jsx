import React, { useState, useEffect, useRef } from "react";
import {
  Activity,
  Database,
  Search,
  ShieldCheck,
  Send,
  Cpu,
  Zap,
  ArrowRight,
  MessageSquare,
  FlaskConical,
} from "lucide-react";

const BACKEND_URL = "https://medai-backend.azurewebsites.net";

// A lightweight helper to format the AI's "clumsy" text into clean HTML
const FormattedText = ({ text }) => {
  if (!text) return null;

  // Basic markdown parsing for a clean UI
  const lines = text.split("\n").map((line, i) => {
    // Handle Bold
    let formattedLine = line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Handle Bullet points
    if (
      formattedLine.trim().startsWith("*") ||
      formattedLine.trim().startsWith("-")
    ) {
      return (
        <li
          key={i}
          className="ml-4 mb-2 list-disc"
          dangerouslySetInnerHTML={{
            __html: formattedLine.replace(/^[*|-]\s*/, ""),
          }}
        />
      );
    }

    // Handle Numbered lists
    if (/^\d+\./.test(formattedLine.trim())) {
      return (
        <li
          key={i}
          className="ml-4 mb-2 list-decimal"
          dangerouslySetInnerHTML={{
            __html: formattedLine.replace(/^\d+\.\s*/, ""),
          }}
        />
      );
    }

    if (formattedLine.trim() === "") return <br key={i} />;

    return (
      <p
        key={i}
        className="mb-3 leading-relaxed"
        dangerouslySetInnerHTML={{ __html: formattedLine }}
      />
    );
  });

  return <div className="text-slate-300">{lines}</div>;
};

const App = () => {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "ai",
      text: "Medical Discovery Engine online. I've indexed the results from your Spark pipeline. How can I help with your research?",
    },
  ]);
  const [stats, setStats] = useState({ count: 0, status: "Offline" });
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/stats`);
        const data = await res.json();
        setStats(data);
      } catch (e) {
        setStats({ count: 0, status: "Offline" });
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const userMsg = query;
    setQuery("");
    setMessages((prev) => [...prev, { role: "user", text: userMsg }]);
    setIsLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg }),
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { role: "ai", text: data.response }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: "Connection error. Ensure backend is running." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#020617] text-slate-200 font-sans overflow-hidden">
      {/* Sidebar */}
      <div className="w-80 bg-[#0f172a] border-r border-slate-800 flex flex-col p-6 space-y-8">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-600 rounded-lg">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            MedAI Insight
          </h1>
        </div>

        <div className="space-y-4">
          <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800">
            <p className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-1">
              Validated Corpus
            </p>
            <div className="flex items-baseline space-x-2">
              <span className="text-4xl font-black text-blue-400">
                {stats.count}
              </span>
              <span className="text-sm text-slate-400">
                Processed Abstracts
              </span>
            </div>
          </div>

          <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800">
            <p className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-1">
              Quality Gate
            </p>
            <div className="flex items-center space-x-2 text-emerald-400 font-medium">
              <ShieldCheck className="w-4 h-4" />
              <span>ROUGE-L {">"} 0.15 ACTIVE</span>
            </div>
          </div>

          <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800">
            <p className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-1">
              Backend Engine
            </p>
            <div className="flex items-center space-x-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  stats.status === "Online"
                    ? "bg-emerald-500 shadow-[0_0_8px_#10b981]"
                    : "bg-red-500"
                }`}
              />
              <span
                className={`text-sm font-bold ${
                  stats.status === "Online"
                    ? "text-emerald-500"
                    : "text-red-500"
                }`}
              >
                {stats.status.toUpperCase()}
              </span>
            </div>
          </div>
        </div>

        <div className="mt-auto pt-6 border-t border-slate-800">
          <div className="flex items-center space-x-3 opacity-60 grayscale">
            <Cpu className="w-4 h-4" />
            <span className="text-xs font-medium">GPU OPTIMIZED</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative">
        <header className="h-16 border-b border-slate-800 flex items-center justify-between px-8 bg-[#020617]/80 backdrop-blur-md">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">
              Scalable Knowledge Discovery Pipeline
            </span>
          </div>
          <div className="flex items-center space-x-6 text-xs font-semibold text-slate-400">
            <span className="flex items-center">
              <Zap className="w-3 h-3 mr-1 text-yellow-500" /> POWERED BY APACHE
              SPARK
            </span>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6 scrollbar-hide">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${
                m.role === "user" ? "justify-end" : "justify-start"
              } animate-in fade-in slide-in-from-bottom-2 duration-300`}
            >
              <div
                className={`max-w-[80%] rounded-2xl p-5 ${
                  m.role === "user"
                    ? "bg-blue-600 text-white shadow-lg"
                    : "bg-slate-900 border border-slate-800 text-slate-200"
                }`}
              >
                {m.role === "ai" ? (
                  <FormattedText text={m.text} />
                ) : (
                  <p>{m.text}</p>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl animate-pulse flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>

        {/* Input Area */}
        <div className="p-8 bg-gradient-to-t from-[#020617] to-transparent">
          <form
            onSubmit={handleQuery}
            className="max-w-4xl mx-auto relative group"
          >
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Query clinical findings or summaries..."
              className="w-full bg-[#0f172a] border border-slate-700 rounded-2xl py-5 pl-6 pr-20 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all text-slate-200 placeholder-slate-500 shadow-2xl"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="absolute right-3 top-3 bottom-3 px-6 bg-blue-600 hover:bg-blue-500 text-white rounded-xl transition-colors flex items-center space-x-2 disabled:opacity-50"
            >
              {isLoading ? (
                <Activity className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </form>
          <p className="text-center text-[10px] text-slate-600 mt-4 uppercase tracking-[0.2em]">
            Verified Research Dataset • RAG-Enabled Synthesis
          </p>
        </div>
      </div>
    </div>
  );
};

export default App;
