import { useEffect, useRef } from "react";
import { BotIcon } from "./Icons.jsx";
import MessageContent from "./MessageContent.jsx";
import RouteProgress from "./RouteProgress.jsx";

const SUGGESTIONS = [
  "叶片出现梭形褐斑，可能是什么？",
  "白叶枯病有哪些典型表现？",
  "如何区分褐斑病和稻瘟病？",
];

export default function ChatThread({ messages, route, stage, sources, timings, onSuggestion }) {
  const endRef = useRef(null);
  useEffect(() => {
    const container = endRef.current?.closest(".conversation-scroll");
    container?.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [messages, stage]);

  if (!messages.length) {
    return (
      <section className="empty-chat" aria-labelledby="welcome-title">
        <div className="empty-mark"><BotIcon /></div>
        <h1 id="welcome-title">今天想了解哪种水稻病害？</h1>
        <p>描述田间症状，或上传一张清晰的叶片照片。</p>
        <div className="suggestion-list">
          {SUGGESTIONS.map((item) => (
            <button type="button" key={item} onClick={() => onSuggestion(item)}>{item}</button>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="chat-thread" aria-live="polite">
      <div className="day-divider"><span>今天</span></div>
      {messages.map((message) => (
        <article className={`message-row is-${message.role}`} key={message.id}>
          {message.role === "assistant" ? (
            <div className="avatar"><BotIcon /></div>
          ) : null}
          <div className="message-body">
            {message.imageUrl ? (
              <img className="message-image" src={message.imageUrl} alt="用户上传的水稻图片" />
            ) : null}
            {message.role === "assistant" && message.pending ? (
              <div className="assistant-status">
                <span className="spinner" />
                {stage === "guard" ? "正在核验忠实度…" : "正在检索并生成…"}
              </div>
            ) : null}
            {message.role === "assistant" ? (
              <RouteProgress route={route} stage={stage} />
            ) : null}
            {message.text ? <MessageContent text={message.text} /> : null}
            {message.role === "assistant" && message.streaming ? <span className="stream-cursor" /> : null}
            {message.role === "assistant" && !message.pending && sources.length ? (
              <div className="source-list">
                <strong>参考来源</strong>
                {sources.slice(0, 4).map((source) => (
                  <span key={`${source.index}-${source.chunk_id || source.source}`}>
                    [{source.index}] {source.source} · {Number(source.relevance_score || 0).toFixed(2)}
                  </span>
                ))}
              </div>
            ) : null}
            {message.role === "assistant" && !message.pending && timings?.total_ms ? (
              <div className="answer-meta">
                路由：{route?.label || "—"} · 总耗时 {(timings.total_ms / 1000).toFixed(2)} 秒
              </div>
            ) : null}
          </div>
        </article>
      ))}
      <div ref={endRef} />
    </section>
  );
}
