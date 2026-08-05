import { ChatIcon, PlusIcon } from "./Icons.jsx";

const HISTORY = [
  { id: "current", title: "当前问诊", time: "刚刚" },
  { id: "blast", title: "稻瘟病叶斑与防治", time: "今天 10:32" },
  { id: "blight", title: "细菌性条斑病识别", time: "昨天 16:48" },
  { id: "brown", title: "胡麻斑病症状咨询", time: "5 月 13 日" },
];

export default function Sidebar({ open, onNewChat, onClose }) {
  return (
    <aside className={`sidebar ${open ? "is-open" : ""}`} aria-label="会话历史">
      <button className="new-chat" type="button" onClick={onNewChat}>
        <PlusIcon />
        新对话
      </button>
      <div className="sidebar-heading">
        <span>历史对话</span>
        <button className="mobile-close" type="button" onClick={onClose} aria-label="关闭历史对话">
          ×
        </button>
      </div>
      <nav className="history-list">
        {HISTORY.map((item, index) => (
          <button
            className={`history-item ${index === 0 ? "is-active" : ""}`}
            key={item.id}
            type="button"
          >
            <ChatIcon />
            <span>
              <strong>{item.title}</strong>
              <small>{item.time}</small>
            </span>
          </button>
        ))}
      </nav>
      <p className="sidebar-note">会话上下文采用滑动窗口，仅保留最近有效轮次。</p>
    </aside>
  );
}
