from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any, AsyncIterator, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from rice_agent.config import settings
from rice_agent.evaluation.grounding import FaithfulnessGuard, message_text
from rice_agent.rag.hybrid_retriever import HybridRetriever
from rice_agent.rag.query_expansion import QueryEnhancer
from rice_agent.rag.router import RouteMode, decide_route


class RagAgentState(TypedDict):
    question: str
    history: list[dict[str, str]]
    image_path: str | None
    detection: dict[str, Any] | None
    query_plan: dict[str, Any]
    contexts: list[dict[str, Any]]
    route: dict[str, Any]
    answer: str
    retry_count: int
    retry_feedback: str
    faithfulness_gate: dict[str, Any]
    forced_fallback: bool
    timings: dict[str, float]
    started_at: float


Retriever = Callable[..., Any]
Detector = Callable[..., Any]


def _now_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _trim(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…"


class LangGraphRiceRagAgent:
    """检测、检索、三级路由、生成、幻觉门禁与重试的状态图。"""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        detector: Detector | None = None,
        llm: Any | None = None,
        guard: FaithfulnessGuard | None = None,
        query_enhancer: QueryEnhancer | None = None,
    ) -> None:
        self.retriever = retriever or self._default_retriever
        self.detector = detector or self._default_detector
        self.llm = llm if llm is not None else self._build_llm()
        self.generation_llm = (
            self.llm.with_config({"tags": ["nostream", "answer-draft"]})
            if self.llm is not None
            else None
        )
        self.guard = guard or FaithfulnessGuard(self.llm)
        self.query_enhancer = query_enhancer or QueryEnhancer(self.llm)
        self.graph = self._build_graph()

    @staticmethod
    def _build_llm() -> Any | None:
        if not settings.deepseek_api_key:
            return None
        return ChatDeepSeek(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
            max_retries=2,
        )

    @staticmethod
    async def _default_retriever(
        question: str,
        disease_code: str | None,
        k: int,
        *,
        query_plan: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            from rice_agent.services.rag_store import RiceKnowledgeStore

            if not hasattr(LangGraphRiceRagAgent, "_hybrid_retriever"):
                store = RiceKnowledgeStore()
                LangGraphRiceRagAgent._hybrid_retriever = HybridRetriever(store)
            return await LangGraphRiceRagAgent._hybrid_retriever.search(
                question,
                disease_code,
                k,
                query_plan=query_plan,
            )
        except ModuleNotFoundError:
            from rice_agent.services.fallback_search import keyword_search

            return keyword_search(question, disease_code, k)

    @staticmethod
    def _default_detector(image_path: str) -> dict[str, Any]:
        from rice_agent.agent.tools import get_detector

        return get_detector().detect(
            image_path=image_path,
            confidence_threshold=settings.yolo_confidence,
            iou_threshold=settings.yolo_iou,
            save_annotated=True,
        )

    @staticmethod
    async def _call(service: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(service):
            return await service(*args, **kwargs)
        result = await asyncio.to_thread(service, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _build_graph(self) -> Any:
        builder = StateGraph(RagAgentState)
        builder.add_node("detect", self._detect_node)
        builder.add_node("enhance_query", self._enhance_query_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("route", self._route_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("guard", self._guard_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_edge(START, "detect")
        builder.add_edge("detect", "enhance_query")
        builder.add_edge("enhance_query", "retrieve")
        builder.add_edge("retrieve", "route")
        builder.add_edge("route", "generate")
        builder.add_edge("generate", "guard")
        builder.add_conditional_edges(
            "guard",
            self._after_guard,
            {"retry": "generate", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    async def _detect_node(self, state: RagAgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        if not state["image_path"]:
            writer({"event": "status", "stage": "detect", "state": "skipped"})
            return {"detection": None}

        writer({"event": "status", "stage": "detect", "state": "running"})
        started = time.perf_counter()
        try:
            result = await self._call(self.detector, state["image_path"])
        except Exception as exc:
            result = {
                "success": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        timings = {**state["timings"], "detection_ms": _now_ms(started)}
        writer(
            {
                "event": "detection",
                "stage": "detect",
                "state": "completed" if result.get("success") else "failed",
                "data": self._public_detection(result),
            }
        )
        return {"detection": result, "timings": timings}

    async def _enhance_query_node(
        self,
        state: RagAgentState,
    ) -> dict[str, Any]:
        writer = get_stream_writer()
        writer(
            {
                "event": "status",
                "stage": "query_enhancement",
                "state": "running",
            }
        )
        started = time.perf_counter()
        plan = await self.query_enhancer.enhance(
            state["question"],
            state["history"],
        )
        timings = {
            **state["timings"],
            "query_enhancement_ms": _now_ms(started),
        }
        data = plan.to_dict()
        writer(
            {
                "event": "query_enhancement",
                "stage": "query_enhancement",
                "state": "completed",
                "rewritten_query": data["rewritten_query"],
                "multi_queries": data["multi_queries"],
                "hyde_used": data["hyde_used"],
                "hyde_preview": _trim(data["hyde_document"], 160),
                "method": data["method"],
            }
        )
        return {"query_plan": data, "timings": timings}

    async def _call_retriever(
        self,
        question: str,
        disease_code: str | None,
        k: int,
        query_plan: dict[str, Any],
    ) -> Any:
        kwargs: dict[str, Any] = {}
        try:
            parameters = inspect.signature(self.retriever).parameters.values()
            if any(
                parameter.name == "query_plan"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            ):
                kwargs["query_plan"] = query_plan
        except (TypeError, ValueError):
            pass
        return await self._call(
            self.retriever,
            question,
            disease_code,
            k,
            **kwargs,
        )

    async def _retrieve_node(self, state: RagAgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"event": "status", "stage": "retrieve", "state": "running"})
        started = time.perf_counter()
        disease_codes = self._candidate_codes(state.get("detection"))
        search_codes: list[str | None] = disease_codes or [None]

        tasks = [
            self._call_retriever(
                state["question"],
                code,
                settings.rag_final_top_k,
                state["query_plan"],
            )
            for code in search_codes[:3]
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        contexts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in gathered:
            if isinstance(result, BaseException) or not isinstance(result, list):
                continue
            for item in result:
                if not isinstance(item, dict):
                    continue
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                key = str(metadata.get("chunk_id") or item.get("content") or "")
                if key and key in seen:
                    continue
                seen.add(key)
                contexts.append(item)

        contexts.sort(
            key=lambda item: float(item.get("relevance_score", 0.0) or 0.0),
            reverse=True,
        )
        contexts = contexts[: max(3, min(5, settings.rag_final_top_k))]
        timings = {**state["timings"], "retrieval_ms": _now_ms(started)}
        traces = [
            item.get("retrieval_trace")
            for item in contexts
            if isinstance(item.get("retrieval_trace"), dict)
        ]
        writer(
            {
                "event": "retrieval",
                "stage": "retrieve",
                "state": "completed",
                "count": len(contexts),
                "top_score": round(
                    float(contexts[0].get("relevance_score", 0.0)), 4
                )
                if contexts
                else 0.0,
                "sources": self._sources(contexts),
                "hybrid_trace": traces[0] if traces else {},
            }
        )
        return {"contexts": contexts, "timings": timings}

    async def _route_node(self, state: RagAgentState) -> dict[str, Any]:
        decision = decide_route(
            state["contexts"],
            precise_threshold=settings.rag_precise_threshold,
            reference_threshold=settings.rag_reference_threshold,
            min_precise_chunks=settings.rag_min_precise_chunks,
        )
        route = decision.to_dict()
        get_stream_writer()(
            {
                "event": "route",
                "stage": "route",
                "state": "completed",
                "data": route,
            }
        )
        return {"route": route}

    async def _generate_node(self, state: RagAgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer(
            {
                "event": "status",
                "stage": "generate",
                "state": "retrying" if state["retry_count"] else "running",
                "retry_count": state["retry_count"],
            }
        )
        started = time.perf_counter()
        answer = await self._draft_answer(state)
        timings = {
            **state["timings"],
            "generation_ms": round(
                state["timings"].get("generation_ms", 0.0) + _now_ms(started),
                2,
            ),
        }
        return {"answer": answer, "timings": timings}

    async def _guard_node(self, state: RagAgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"event": "status", "stage": "guard", "state": "running"})
        started = time.perf_counter()
        context_texts = [str(item.get("content", "")) for item in state["contexts"]]
        result = await self.guard.check(state["answer"], context_texts)
        timings = {
            **state["timings"],
            "guard_ms": round(
                state["timings"].get("guard_ms", 0.0) + _now_ms(started),
                2,
            ),
        }
        target_met = result.score >= settings.hallucination_target
        writer(
            {
                "event": "guard",
                "stage": "guard",
                "state": "passed" if target_met else "failed",
                "score": result.score,
                "target": settings.hallucination_target,
                "retry_count": state["retry_count"],
                "method": result.method,
            }
        )

        if target_met:
            return {
                "faithfulness_gate": result.to_dict(),
                "retry_feedback": "",
                "timings": timings,
            }

        if state["retry_count"] < settings.hallucination_max_retries:
            feedback = (
                "上一版未通过忠实度门禁。删除或改写以下无证据声明："
                + "；".join(result.unsupported_claims[:5])
            )
            return {
                "faithfulness_gate": result.to_dict(),
                "retry_count": state["retry_count"] + 1,
                "retry_feedback": feedback,
                "timings": timings,
            }

        fallback = self._safe_fallback(state)
        fallback_result = await self.guard.check(fallback, context_texts)
        writer(
            {
                "event": "guard",
                "stage": "guard",
                "state": "safe_fallback",
                "score": fallback_result.score,
                "target": settings.hallucination_target,
                "retry_count": state["retry_count"],
                "method": fallback_result.method,
            }
        )
        return {
            "answer": fallback,
            "faithfulness_gate": fallback_result.to_dict(),
            "forced_fallback": True,
            "retry_feedback": "",
            "timings": timings,
        }

    @staticmethod
    def _after_guard(state: RagAgentState) -> str:
        score = float(state.get("faithfulness_gate", {}).get("score", 0.0))
        if (
            score < settings.hallucination_target
            and not state.get("forced_fallback", False)
            and state["retry_count"] <= settings.hallucination_max_retries
        ):
            return "retry"
        return "finalize"

    async def _finalize_node(self, state: RagAgentState) -> dict[str, Any]:
        writer = get_stream_writer()
        answer = state["answer"].strip()
        writer(
            {
                "event": "answer_start",
                "route": state["route"],
                "sources": self._sources(state["contexts"]),
            }
        )
        for index in range(0, len(answer), 18):
            writer({"event": "answer_delta", "text": answer[index : index + 18]})
            await asyncio.sleep(0)

        timings = {
            **state["timings"],
            "total_ms": _now_ms(state["started_at"]),
        }
        writer(
            {
                "event": "answer_end",
                "answer": answer,
                "route": state["route"],
                "sources": self._sources(state["contexts"]),
                "guard": state["faithfulness_gate"],
                "retry_count": state["retry_count"],
                "forced_fallback": state["forced_fallback"],
                "timings": timings,
            }
        )
        return {"answer": answer, "timings": timings}

    async def _draft_answer(self, state: RagAgentState) -> str:
        if self.generation_llm is None:
            return self._extractive_answer(state)

        route = RouteMode(state["route"]["mode"])
        route_rules = {
            RouteMode.PRECISE_HIT: (
                "仅使用检索证据回答；每个事实段落附 [序号] 引用。"
                "不得加入证据中没有的事实。"
            ),
            RouteMode.REFERENCE_GENERATION: (
                "以检索证据为参考，只写证据能支持的事实；对缺失信息明确说证据不足。"
                "每个事实段落附 [序号] 引用。"
            ),
            RouteMode.AI_INFERENCE: (
                "知识库未可靠命中。不要给出具体病名、药剂或确定性事实。"
                "只可给出标记为‘推断’的观察路径，并建议补充照片或咨询植保人员。"
            ),
        }[route]

        contexts = "\n\n".join(
            f"[{index}] 来源={item.get('metadata', {}).get('source', '未知')}\n"
            f"{_trim(str(item.get('content', '')), 1800)}"
            for index, item in enumerate(state["contexts"], 1)
        )
        detection = json.dumps(
            self._public_detection(state.get("detection")),
            ensure_ascii=False,
            default=str,
        )
        messages: list[Any] = [
            SystemMessage(
                content=(
                    "你是水稻病虫害 RAG 助手。回答必须安全、克制、可追溯。"
                    "禁止编造药剂名称、剂量、安全间隔期和当地登记信息。"
                    "检测结果只能称为模型筛查，不能称为确诊。"
                    f"当前路由：{state['route']['label']}。{route_rules}"
                    "末尾固定写：‘结果仅供辅助判断，不替代专业诊断。’"
                )
            )
        ]
        for item in state["history"]:
            content = _trim(str(item.get("content", "")), 3000)
            if item.get("role") == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        payload = {
            "question": state["question"],
            "detection": detection,
            "retrieved_contexts": contexts or "无可靠检索证据",
            "retry_feedback": state["retry_feedback"],
            "required_structure": ["判断", "依据", "建议", "参考来源"],
        }
        messages.append(HumanMessage(content=json.dumps(payload, ensure_ascii=False)))
        response = await self.generation_llm.ainvoke(messages)
        answer = message_text(response)
        return answer or self._safe_fallback(state)

    def _safe_fallback(self, state: RagAgentState) -> str:
        if not state["contexts"]:
            return (
                "### 判断\n当前知识库没有检索到足够可靠的依据，因此无法给出具体病害判断。\n\n"
                "### 建议\n请补充清晰的叶片正反面、病斑近照和整株照片，并说明生育期、"
                "田间分布和近期天气；也可联系当地植保人员复核。\n\n"
                "结果仅供辅助判断，不替代专业诊断。"
            )

        excerpts: list[str] = []
        for index, item in enumerate(state["contexts"][:3], 1):
            source = item.get("metadata", {}).get("source", "未知来源")
            content = _trim(str(item.get("content", "")).replace("\n", " "), 320)
            excerpts.append(f"- [{index}] {content}（{source}）")
        return (
            "### 判断\n生成答案未通过忠实度门禁，已切换为知识库原文摘录，"
            "不再补充推断。\n\n### 依据\n"
            + "\n".join(excerpts)
            + "\n\n### 建议\n请结合田间症状与当地植保人员意见复核。\n\n"
            "结果仅供辅助判断，不替代专业诊断。"
        )

    def _extractive_answer(self, state: RagAgentState) -> str:
        if not state["contexts"]:
            return self._safe_fallback(state)
        excerpts: list[str] = []
        for index, item in enumerate(state["contexts"][:3], 1):
            source = item.get("metadata", {}).get("source", "未知来源")
            content = _trim(str(item.get("content", "")).replace("\n", " "), 320)
            excerpts.append(f"- [{index}] {content}（{source}）")
        return (
            "### 判断\n当前未配置生成模型，以下为知识库检索原文摘录。\n\n"
            "### 依据\n"
            + "\n".join(excerpts)
            + "\n\n### 建议\n请结合田间症状与当地植保人员意见复核。\n\n"
            "结果仅供辅助判断，不替代专业诊断。"
        )

    @staticmethod
    def _candidate_codes(detection: dict[str, Any] | None) -> list[str]:
        if not detection or not detection.get("success"):
            return []
        summaries = detection.get("class_summary", [])
        abnormal = [
            item
            for item in summaries
            if item.get("kind") in {"disease", "pest"}
        ]
        selected = abnormal[:3] or summaries[:1]
        return [str(item["disease_code"]) for item in selected if item.get("disease_code")]

    @staticmethod
    def _public_detection(detection: dict[str, Any] | None) -> dict[str, Any] | None:
        if detection is None:
            return None
        allowed = {
            "success",
            "error",
            "message",
            "annotated_image_path",
            "detection_count",
            "primary_result",
            "class_summary",
            "health_conflict",
            "latency_ms",
            "notice",
        }
        return {key: value for key, value in detection.items() if key in allowed}

    @staticmethod
    def _sources(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for index, item in enumerate(contexts, 1):
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            sources.append(
                {
                    "index": index,
                    "source": metadata.get("source", "未知来源"),
                    "disease_code": metadata.get("disease_code"),
                    "chunk_id": metadata.get("chunk_id"),
                    "relevance_score": round(
                        float(item.get("relevance_score", 0.0) or 0.0), 4
                    ),
                    "rrf_score": round(
                        float(item.get("rrf_score", 0.0) or 0.0), 8
                    ),
                    "rerank_score": round(
                        float(item.get("rerank_score", 0.0) or 0.0), 4
                    ),
                    "rerank_method": item.get("rerank_method"),
                    "retrieval_channels": item.get("retrieval_channels", []),
                }
            )
        return sources

    async def stream(
        self,
        *,
        question: str,
        history: list[dict[str, str]] | None = None,
        image_path: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        state: RagAgentState = {
            "question": question.strip(),
            "history": history or [],
            "image_path": image_path,
            "detection": None,
            "query_plan": {},
            "contexts": [],
            "route": {},
            "answer": "",
            "retry_count": 0,
            "retry_feedback": "",
            "faithfulness_gate": {},
            "forced_fallback": False,
            "timings": {},
            "started_at": time.perf_counter(),
        }
        async for part in self.graph.astream(
            state,
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            if part["type"] == "custom":
                event = part["data"]
                if isinstance(event, dict):
                    yield event
            elif part["type"] == "updates":
                for update in part["data"].values():
                    if isinstance(update, dict):
                        state.update(update)

        yield {
            "event": "graph_complete",
            "answer": state["answer"],
            "contexts": state["contexts"],
            "query_plan": state["query_plan"],
            "route": state["route"],
            "guard": state["faithfulness_gate"],
            "retry_count": state["retry_count"],
            "forced_fallback": state["forced_fallback"],
            "timings": state["timings"],
            "detection": self._public_detection(state["detection"]),
        }
