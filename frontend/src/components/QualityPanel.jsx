import { CheckIcon, ThumbsDownIcon, ThumbsUpIcon, WarningIcon } from "./Icons.jsx";

const METRICS = [
  ["faithfulness", "忠实度"],
  ["answer_relevancy", "相关性"],
  ["context_precision", "精确率"],
  ["context_recall", "召回率"],
];

function Metric({ label, value, target }) {
  const score = typeof value === "number" ? value : null;
  const low = score !== null && target && score < target;
  return (
    <div className="quality-metric">
      <div><span>{label}</span><strong className={low ? "is-low" : ""}>{score === null ? "—" : score.toFixed(2)}</strong></div>
      <div className="meter"><span className={low ? "is-low" : ""} style={{ width: `${(score || 0) * 100}%` }} /></div>
      <div className="meter-scale"><span>0</span><span>0.5</span><span>1</span></div>
    </div>
  );
}

export default function QualityPanel({ evaluation, evaluationPending, guard, route, retryCount, feedback, onFeedback }) {
  return (
    <aside className="quality-panel" aria-label="质量评估">
      <div className="quality-title">
        <div>
          <h2>质量评估</h2>
          <span>RAGAS-light</span>
        </div>
        <span className={`quality-state ${evaluation?.target_met ? "is-good" : ""}`}>
          {evaluationPending ? "评估中" : evaluation ? "评估完成" : "等待问答"}
        </span>
      </div>

      <div className="metrics">
        {METRICS.map(([key, label]) => (
          <Metric
            key={key}
            label={label}
            value={evaluation?.[key]}
            target={key === "faithfulness" ? (evaluation?.faithfulness_target || 0.9) : null}
          />
        ))}
      </div>

      <div className="quality-section">
        <h3>幻觉检测</h3>
        <div className={`guard-state ${guard?.score >= 0.9 ? "is-good" : guard ? "is-warning" : ""}`}>
          {guard?.score >= 0.9 ? <CheckIcon /> : <WarningIcon />}
          <span>
            {guard ? `忠实度门禁 ${Number(guard.score).toFixed(2)}` : "等待生成"}
            {retryCount ? <small>已重试 {retryCount} 次</small> : null}
          </span>
        </div>
      </div>

      <div className="quality-section">
        <h3>路由状态</h3>
        <div className="route-summary">
          <strong>{route?.label || "尚未路由"}</strong>
          <span>{route?.reason || "提交问题后自动选择三级策略"}</span>
        </div>
      </div>

      <div className="quality-section feedback-section">
        <h3>本次回答反馈</h3>
        <div className="feedback-actions">
          <button className={feedback === "helpful" ? "is-selected" : ""} type="button" onClick={() => onFeedback("helpful")} disabled={!evaluation}>
            <ThumbsUpIcon />有帮助
          </button>
          <button className={feedback === "needs_improvement" ? "is-selected" : ""} type="button" onClick={() => onFeedback("needs_improvement")} disabled={!evaluation}>
            <ThumbsDownIcon />需改进
          </button>
        </div>
        <p>反馈与四维评分将用于后续知识库和提示词优化。</p>
      </div>
    </aside>
  );
}
