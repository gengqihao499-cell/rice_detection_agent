# Web API 与 SSE 事件

## 健康检查

`GET /api/health`

返回 LLM 是否配置、评估是否开启、忠实度目标和三级路由名称。不会返回密钥。

## 新建/清理会话

- `POST /api/sessions`
- `DELETE /api/sessions/{session_id}`

## 流式问答

`POST /api/chat/stream`，Content-Type 为 `multipart/form-data`：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `question` | string | 与图片至少一个 | 最大 4000 字 |
| `session_id` | string | 否 | 空值时由后端创建 |
| `image` | file | 否 | JPG/PNG/WebP，默认最大 8 MB |

响应 Content-Type 为 `text/event-stream`。每个事件的 `data` 都是 JSON，并包含
同名 `event` 字段。

| 事件 | 作用 |
|---|---|
| `meta` | 会话与轮次 ID |
| `status` | detect/query_enhancement/retrieve/generate/guard 阶段状态 |
| `detection` | YOLO 公开结果与标注图 URL |
| `query_enhancement` | 独立 Query 改写、Multi-Query、HyDE 状态和预览 |
| `retrieval` | 命中数量、最高分、来源列表与混合检索 trace |
| `route` | 三级路由决策与原因 |
| `guard` | 忠实度门禁分、目标、重试次数 |
| `answer_start` | 最终答案开始 |
| `answer_delta` | 文本片段 |
| `answer_end` | 答案、来源、路由、门禁和耗时 |
| `evaluation_pending` | 异步评估开始 |
| `evaluation` | 四维分数和原因 |
| `evaluation_error` | 评估超时；不撤回已完成答案 |
| `done` | 本轮结束 |
| `error` | 本轮失败 |

## 反馈

`POST /api/feedback`，JSON：

```json
{
  "session_id": "...",
  "turn_id": "...",
  "value": "helpful",
  "comment": "",
  "corrected_answer": ""
}
```

`value` 仅支持 `helpful` 或 `needs_improvement`。
