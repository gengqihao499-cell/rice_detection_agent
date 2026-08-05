import { useCallback, useMemo, useState } from "react";
import ChatThread from "./components/ChatThread.jsx";
import Composer from "./components/Composer.jsx";
import { BookIcon, ChartIcon, ChatIcon, MenuIcon, RiceMark } from "./components/Icons.jsx";
import QualityPanel from "./components/QualityPanel.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { useSseChat } from "./hooks/useSseChat.js";

function initialSessionId() {
  const saved = localStorage.getItem("ricecare.session.v1");
  if (saved) return saved;
  const created = crypto.randomUUID().replaceAll("-", "");
  localStorage.setItem("ricecare.session.v1", created);
  return created;
}

export default function App() {
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [turnId, setTurnId] = useState("");
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [image, setImage] = useState(null);
  const [stage, setStage] = useState("");
  const [route, setRoute] = useState(null);
  const [sources, setSources] = useState([]);
  const [guard, setGuard] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const [timings, setTimings] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [evaluationPending, setEvaluationPending] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const updateAssistant = useCallback((updater) => {
    setMessages((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === "assistant") {
          next[index] = updater(next[index]);
          break;
        }
      }
      return next;
    });
  }, []);

  const handleEvent = useCallback((event, data) => {
    if (event === "meta") {
      setSessionId(data.session_id);
      setTurnId(data.turn_id);
      localStorage.setItem("ricecare.session.v1", data.session_id);
      return;
    }
    if (event === "status") {
      setStage(data.stage || "");
      if (typeof data.retry_count === "number") setRetryCount(data.retry_count);
      return;
    }
    if (event === "retrieval") {
      setSources(data.sources || []);
      return;
    }
    if (event === "route") {
      setRoute(data.data);
      return;
    }
    if (event === "guard") {
      setStage("guard");
      setGuard({ score: data.score, method: data.method, state: data.state });
      setRetryCount(data.retry_count || 0);
      return;
    }
    if (event === "answer_start") {
      setRoute(data.route || null);
      setSources(data.sources || []);
      updateAssistant((message) => ({ ...message, pending: false, streaming: true }));
      return;
    }
    if (event === "answer_delta") {
      updateAssistant((message) => ({ ...message, text: message.text + (data.text || "") }));
      return;
    }
    if (event === "answer_end") {
      setRoute(data.route || null);
      setSources(data.sources || []);
      setGuard(data.guard || null);
      setRetryCount(data.retry_count || 0);
      setTimings(data.timings || null);
      updateAssistant((message) => ({
        ...message,
        text: data.answer || message.text,
        pending: false,
        streaming: false,
      }));
      return;
    }
    if (event === "evaluation_pending") {
      setEvaluationPending(true);
      return;
    }
    if (event === "evaluation") {
      setEvaluation(data);
      setEvaluationPending(false);
      return;
    }
    if (event === "evaluation_error") {
      setEvaluationPending(false);
      return;
    }
    if (event === "error" || event === "client_error") {
      setEvaluationPending(false);
      updateAssistant((message) => ({
        ...message,
        text: `请求未完成：${data.message || "未知错误"}`,
        pending: false,
        streaming: false,
        error: true,
      }));
    }
  }, [updateAssistant]);

  const { send, isStreaming } = useSseChat(handleEvent);

  const submit = useCallback(() => {
    const cleanQuestion = question.trim();
    if ((!cleanQuestion && !image) || isStreaming) return;
    const selectedImage = image;
    const imageUrl = selectedImage ? URL.createObjectURL(selectedImage) : null;
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: cleanQuestion || "请分析这张水稻图片。",
        imageUrl,
      },
      {
        id: crypto.randomUUID(),
        role: "assistant",
        text: "",
        pending: true,
        streaming: false,
      },
    ]);
    setQuestion("");
    setImage(null);
    setStage("detect");
    setRoute(null);
    setSources([]);
    setGuard(null);
    setRetryCount(0);
    setTimings(null);
    setEvaluation(null);
    setEvaluationPending(false);
    setFeedback("");
    send({ question: cleanQuestion, image: selectedImage, sessionId });
  }, [image, isStreaming, question, send, sessionId]);

  const newChat = useCallback(() => {
    if (sessionId) fetch(`/api/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {});
    const nextId = crypto.randomUUID().replaceAll("-", "");
    localStorage.setItem("ricecare.session.v1", nextId);
    setSessionId(nextId);
    setMessages([]);
    setRoute(null);
    setSources([]);
    setGuard(null);
    setEvaluation(null);
    setTimings(null);
    setFeedback("");
    setSidebarOpen(false);
  }, [sessionId]);

  const sendFeedback = useCallback(async (value) => {
    if (!turnId || !evaluation) return;
    setFeedback(value);
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, turn_id: turnId, value }),
      });
    } catch {
      setFeedback("");
    }
  }, [evaluation, sessionId, turnId]);

  const activeNav = useMemo(() => "智能问诊", []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="menu-button" type="button" onClick={() => setSidebarOpen(true)} aria-label="打开历史对话"><MenuIcon /></button>
        <div className="brand"><RiceMark /><span><strong>稻问</strong> RiceCare</span></div>
        <nav className="topnav" aria-label="主导航">
          <button className={activeNav === "智能问诊" ? "is-active" : ""} type="button"><ChatIcon />智能问诊</button>
          <button type="button"><BookIcon />知识库</button>
          <button type="button"><ChartIcon />评估记录</button>
        </nav>
        <div className="profile">农艺师</div>
      </header>

      <div className="workspace">
        <Sidebar open={sidebarOpen} onNewChat={newChat} onClose={() => setSidebarOpen(false)} />
        {sidebarOpen ? <button className="sidebar-backdrop" type="button" onClick={() => setSidebarOpen(false)} aria-label="关闭侧栏" /> : null}
        <main className="conversation">
          <div className="conversation-scroll">
            <ChatThread messages={messages} route={route} stage={stage} sources={sources} timings={timings} onSuggestion={setQuestion} />
          </div>
          <div className="composer-wrap">
            <Composer value={question} onChange={setQuestion} image={image} onImage={setImage} onSend={submit} disabled={isStreaming} />
            <p className="safety-note">结果仅供辅助判断，不替代专业诊断</p>
          </div>
        </main>
        <QualityPanel
          evaluation={evaluation}
          evaluationPending={evaluationPending}
          guard={guard}
          route={route}
          retryCount={retryCount}
          feedback={feedback}
          onFeedback={sendFeedback}
        />
      </div>
    </div>
  );
}
