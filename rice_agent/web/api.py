from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from rice_agent.config import settings
from rice_agent.evaluation.ragas_light import EvaluationInput, RagasLightEvaluator
from rice_agent.evaluation.store import JsonlQualityStore
from rice_agent.rag.graph_agent import LangGraphRiceRagAgent
from rice_agent.rag.memory import ConversationTurn, SlidingWindowConversationStore


logger = logging.getLogger("ricecare.web")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

MIME_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
FRONTEND_DIST = settings.project_root / "frontend" / "dist"


@dataclass(slots=True)
class Runtime:
    agent: LangGraphRiceRagAgent
    evaluator: RagasLightEvaluator
    conversations: SlidingWindowConversationStore
    quality_store: JsonlQualityStore


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    agent = LangGraphRiceRagAgent()
    return Runtime(
        agent=agent,
        evaluator=RagasLightEvaluator(
            llm=agent.llm,
            faithfulness_target=settings.hallucination_target,
        ),
        conversations=SlidingWindowConversationStore(
            max_turns=settings.chat_window_turns,
            max_chars=settings.chat_max_history_chars,
        ),
        quality_store=JsonlQualityStore(
            evaluation_path=settings.evaluation_log_path,
            feedback_path=settings.feedback_log_path,
        ),
    )


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    turn_id: str = Field(min_length=8, max_length=128)
    value: str = Field(pattern="^(helpful|needs_improvement)$")
    comment: str = Field(default="", max_length=2000)
    corrected_answer: str = Field(default="", max_length=12000)


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = {**data, "event": event}
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
    )


def _verify_image(data: bytes) -> None:
    with Image.open(BytesIO(data)) as image:
        image.verify()


async def _save_upload(upload: UploadFile) -> Path:
    content_type = (upload.content_type or "").lower()
    suffix = MIME_SUFFIXES.get(content_type)
    if suffix is None:
        raise HTTPException(status_code=415, detail="仅支持 JPG、PNG、WebP 图片")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"图片不能超过 {settings.max_upload_mb} MB",
        )
    if not data:
        raise HTTPException(status_code=400, detail="上传图片为空")

    try:
        await asyncio.to_thread(_verify_image, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="图片内容无法解析") from exc

    target_dir = settings.upload_dir / "web"
    await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
    target = target_dir / f"{uuid4().hex}{suffix}"
    await asyncio.to_thread(target.write_bytes, data)
    return target


def _public_graph_event(event: dict[str, Any]) -> dict[str, Any]:
    detection = event.get("detection")
    if isinstance(detection, dict):
        annotated = detection.pop("annotated_image_path", None)
        if annotated:
            detection["annotated_image_url"] = f"/api/artifacts/{Path(annotated).name}"
    data = event.get("data")
    if isinstance(data, dict):
        annotated = data.pop("annotated_image_path", None)
        if annotated:
            data["annotated_image_url"] = f"/api/artifacts/{Path(annotated).name}"
    return event


app = FastAPI(
    title="稻问 RiceCare",
    version="2.0.0",
    description="LangGraph 多级 RAG + SSE + RAGAS-light 在线质量闭环",
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_configured": bool(settings.deepseek_api_key),
        "evaluation_enabled": settings.evaluation_enabled,
        "faithfulness_target": settings.hallucination_target,
        "routes": ["precise_hit", "reference_generation", "ai_inference"],
    }


@app.post("/api/sessions")
async def create_session() -> dict[str, str]:
    return {"session_id": get_runtime().conversations.create_session()}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str) -> dict[str, bool]:
    get_runtime().conversations.clear(session_id)
    return {"cleared": True}


@app.post("/api/chat/stream")
async def chat_stream(
    question: str = Form(default=""),
    session_id: str = Form(default=""),
    image: UploadFile | None = File(default=None),
) -> StreamingResponse:
    cleaned_question = question.strip()
    if not cleaned_question and image is None:
        raise HTTPException(status_code=400, detail="问题和图片不能同时为空")
    if len(cleaned_question) > 4000:
        raise HTTPException(status_code=400, detail="问题不能超过 4000 字")
    if not cleaned_question:
        cleaned_question = "请分析这张水稻图片，并说明可能表现与复核建议。"

    image_path = await _save_upload(image) if image is not None else None
    runtime = get_runtime()
    resolved_session_id = runtime.conversations.ensure_session(session_id)
    turn_id = uuid4().hex
    history = runtime.conversations.history(resolved_session_id)

    async def generate() -> AsyncIterator[str]:
        final: dict[str, Any] | None = None
        yield _sse(
            "meta",
            {
                "session_id": resolved_session_id,
                "turn_id": turn_id,
                "question": cleaned_question,
                "has_image": image_path is not None,
            },
        )
        try:
            async for event in runtime.agent.stream(
                question=cleaned_question,
                history=history,
                image_path=str(image_path) if image_path else None,
            ):
                event = _public_graph_event(event)
                event_name = str(event.get("event", "message"))
                if event_name == "graph_complete":
                    final = event
                    continue
                yield _sse(event_name, event)

            if final is None:
                raise RuntimeError("LangGraph 未返回最终状态")

            runtime.conversations.append(
                resolved_session_id,
                ConversationTurn(
                    user=cleaned_question,
                    assistant=str(final["answer"]),
                    turn_id=turn_id,
                    metadata={
                        "route": final.get("route"),
                        "guard": final.get("guard"),
                        "timings": final.get("timings"),
                    },
                ),
            )

            if settings.evaluation_enabled:
                yield _sse(
                    "evaluation_pending",
                    {"session_id": resolved_session_id, "turn_id": turn_id},
                )
                sample = EvaluationInput(
                    session_id=resolved_session_id,
                    turn_id=turn_id,
                    question=cleaned_question,
                    answer=str(final["answer"]),
                    contexts=[
                        str(item.get("content", ""))
                        for item in final.get("contexts", [])
                    ],
                    route_mode=str(final.get("route", {}).get("mode", "")),
                )
                evaluation = await asyncio.wait_for(
                    runtime.evaluator.evaluate(sample),
                    timeout=settings.evaluation_timeout_seconds,
                )
                evaluation_payload = evaluation.to_dict()
                await runtime.quality_store.append_evaluation(evaluation_payload)
                logger.info(
                    "quality turn=%s faithfulness=%.3f relevancy=%.3f "
                    "precision=%.3f recall=%.3f target_met=%s",
                    turn_id,
                    evaluation.faithfulness,
                    evaluation.answer_relevancy,
                    evaluation.context_precision,
                    evaluation.context_recall,
                    evaluation.target_met,
                )
                yield _sse("evaluation", evaluation_payload)

            yield _sse(
                "done",
                {
                    "session_id": resolved_session_id,
                    "turn_id": turn_id,
                    "timings": final.get("timings", {}),
                },
            )
        except asyncio.TimeoutError:
            logger.warning("evaluation timeout turn=%s", turn_id)
            yield _sse(
                "evaluation_error",
                {"turn_id": turn_id, "message": "异步评估超时，答案已保留"},
            )
            yield _sse("done", {"session_id": resolved_session_id, "turn_id": turn_id})
        except Exception as exc:
            logger.exception("chat stream failed turn=%s", turn_id)
            yield _sse(
                "error",
                {
                    "session_id": resolved_session_id,
                    "turn_id": turn_id,
                    "message": str(exc),
                    "error": type(exc).__name__,
                },
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict[str, bool]:
    payload = request.model_dump()
    payload["created_at"] = datetime.now(UTC).isoformat()
    await get_runtime().quality_store.append_feedback(payload)
    logger.info(
        "feedback turn=%s value=%s corrected=%s",
        request.turn_id,
        request.value,
        bool(request.corrected_answer),
    )
    return {"saved": True}


@app.get("/api/artifacts/{filename}")
async def artifact(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = settings.output_dir / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path)


if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )


@app.get("/{path:path}", include_in_schema=False, response_model=None)
async def frontend(path: str):
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(
        status_code=503,
        content={
            "message": "前端尚未构建，请先运行 cd frontend && npm install && npm run build",
            "api_docs": "/docs",
        },
    )
