import { CheckIcon } from "./Icons.jsx";

const STEPS = [
  { mode: "precise_hit", label: "精准命中", hint: "知识库检索" },
  { mode: "reference_generation", label: "参考生成", hint: "有边界生成" },
  { mode: "ai_inference", label: "AI 推断", hint: "证据不足时启用" },
];

export default function RouteProgress({ route, stage }) {
  const activeIndex = route ? STEPS.findIndex((step) => step.mode === route.mode) : -1;
  return (
    <div className="route-progress" aria-label="三级 RAG 路由">
      {STEPS.map((step, index) => {
        const isActive = index === activeIndex;
        const isPassed = activeIndex > index || (index === 0 && stage === "retrieve");
        return (
          <div className={`route-step ${isActive ? "is-active" : ""} ${isPassed ? "is-passed" : ""}`} key={step.mode}>
            <span className="route-node">{isPassed ? <CheckIcon /> : index + 1}</span>
            <span className="route-copy">
              <strong>{step.label}</strong>
              <small>{step.hint}</small>
            </span>
          </div>
        );
      })}
    </div>
  );
}
