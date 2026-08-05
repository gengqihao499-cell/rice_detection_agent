# 系统架构

## 在线问答链路

```text
React 对话界面
  │  POST multipart/form-data
  ▼
FastAPI /api/chat/stream
  │
  ▼
LangGraph StateGraph
  ├─ detect：有图片时运行 YOLO11L
  ├─ retrieve：按最多 3 个候选类别并行查询 BGE + Chroma
  ├─ route：按相关度选择精准命中 / 参考生成 / AI 推断
  ├─ generate：DeepSeek 依据路由提示和滑动窗口生成草稿
  ├─ guard：声明级忠实度门禁（目标 >= 0.90）
  │    └─ 未达标：携带无证据声明重试，最多 2 次
  └─ finalize：门禁通过后才通过 SSE 发送答案
       │
       ├─ 异步 RAGAS-light 四维评估
       ├─ evaluations.jsonl
       └─ SSE evaluation 事件 → 前端质量面板
```

用户的“有帮助 / 需改进”反馈由 `/api/feedback` 写入 `feedback.jsonl`，可与
`turn_id` 对齐评估记录，作为后续知识库补全、阈值校准和提示词实验的数据集。

## 三级路由

| 模式 | 默认条件 | 生成约束 |
|---|---|---|
| 精准命中 | 至少 1 个块相关度 `>= 0.78` | 只用证据事实并引用 |
| 参考生成 | 最高相关度 `0.45 ~ 0.78` | 有边界生成，缺失信息明确说证据不足 |
| AI 推断 | 最高相关度 `< 0.45` 或无证据 | 不给确定病名，只给观察/补充信息路径 |

阈值通过 `.env` 的 `RAG_PRECISE_THRESHOLD`、`RAG_REFERENCE_THRESHOLD` 和
`RAG_MIN_PRECISE_CHUNKS` 调整。

## 会话管理

FastAPI 内存会话按 `session_id` 保存完成轮次。每次构造提示词时只读取最近
`CHAT_WINDOW_TURNS` 轮，并进一步受 `CHAT_MAX_HISTORY_CHARS` 限制。生产环境若
需要多实例或重启恢复，应替换为 Redis/Postgres checkpointer；当前实现适合单机演示。

## 流式策略

草稿不会立即暴露给用户。系统先完成忠实度门禁与必要重试，再用
`answer_start / answer_delta / answer_end` 事件流式发送已通过门禁的最终答案。
这样牺牲少量首字节时间，换取不会把被判定为幻觉的草稿先显示再撤回。

完整 SSE 事件见 `API.md`，指标语义见 `EVALUATION.md`。
