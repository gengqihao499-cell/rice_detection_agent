# 稻问 RiceCare：水稻病虫害多级 RAG 对话平台

这是一个可本地运行的水稻病虫害辅助问答平台。系统把 YOLO11L 图片检测、
BGE 中文向量、ChromaDB、LangGraph 三级路由、DeepSeek、SSE 流式输出和
RAGAS-light 在线质量评估组合成完整的多轮对话闭环。

> 结果仅供辅助筛查，不构成确诊；涉及农药时请遵循产品标签和当地植保机构意见。

## 已实现能力

- React 对话界面：会话历史、图片上传、流式回答、引用、路由状态和质量面板。
- LangGraph 多级 RAG：`精准命中`、`参考生成`、`AI 推断` 三种策略。
- 本地检索：`BAAI/bge-small-zh-v1.5` + ChromaDB，支持疾病代码过滤。
- 幻觉门禁：声明级忠实度检查，目标 `>= 0.90`，失败后最多重试 2 次；
  仍不合格时自动降级为知识库原文摘录。
- 异步 RAGAS-light：忠实度、相关性、上下文精确率、上下文召回率；
  答案完成后通过同一 SSE 连接回传，不阻塞首个可用答案。
- 质量闭环：评估记录写入 `outputs/evaluations.jsonl`，用户反馈写入
  `outputs/feedback.jsonl`，后端同步输出每轮指标日志。
- 滑动窗口：默认保留最近 6 轮、最多 12000 字，避免上下文无限增长。
- 降级能力：BGE/Chroma 依赖暂不可用时可用可解释关键词检索启动；生产环境仍应
  使用 BGE + Chroma。

## 快速开始

### 1. Python 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写自己的 `DEEPSEEK_API_KEY`。`.env.example` 只能放占位符，
不得提交真实密钥。

### 2. 构建知识库

```powershell
python -m scripts.build_rag_index
```

第一次运行会下载 BGE 模型，并在 `data/rice_chroma_db/` 创建或更新索引。

### 3. 构建对话前端

```powershell
cd frontend
npm install
npm run build
cd ..
```

项目已锁定兼容 Node 20.10 的 Vite 6 版本。

### 4. 启动平台

```powershell
python web_app.py
```

打开 `http://127.0.0.1:8000`。API 文档位于 `http://127.0.0.1:8000/docs`。

开发前端时可分别运行：

```powershell
python web_app.py
cd frontend
npm run dev
```

Vite 会把 `/api` 代理到 FastAPI 的 `8000` 端口。

## SSE 事件顺序

`POST /api/chat/stream` 接收 `question`、`session_id` 和可选 `image`，返回：

```text
meta
  -> status / detection
  -> retrieval
  -> route
  -> guard（可能触发 retry）
  -> answer_start / answer_delta / answer_end
  -> evaluation_pending
  -> evaluation 或 evaluation_error
  -> done
```

完整接口与事件字段见 `docs/API.md`。

## 质量评估边界

- 忠实度采用 RAGAS 的声明支持比例思想，并作为发送前硬门禁。
- 没有人工参考答案时，在线召回率会明确标为 `online_proxy`；它衡量问题所需信息
  是否被上下文覆盖，不冒充标准金答案召回率。
- 有标注评估集时传入 `reference_answer`，召回率切换为 `gold_reference`。
- 在线模块用一次异步 LLM 判分完成四项指标，失败时使用透明的词法回退。

详细定义见 `docs/EVALUATION.md`。

## 响应时间前后对比

运行受控编排基准：

```powershell
python -m scripts.benchmark_latency --trials 5 --output outputs/latency_benchmark.json
```

本次 5 轮结果：

| 指标 | 中位数 |
|---|---:|
| 旧版串行检测 → 三级检索 → 生成 | 407.89 ms |
| LangGraph 并行检索 → 生成 → 忠实度门禁 | 285.60 ms |
| 异步质量事件到达 | 392.27 ms |
| 答案延迟改善 | 29.98% |

这是隔离“串行/并行检索与异步评估”结构收益的受控基准，不代表真实 YOLO、BGE、
DeepSeek 网络耗时。上线前应在目标硬件和网络上追加真实 P50/P95、首字节时间和
整轮评估时间。

## 测试

```powershell
pytest -q
npm --prefix frontend run build
```

模型和检索的手工测试仍可运行：

```powershell
python -m scripts.check_model
python -m scripts.test_rag "叶瘟有哪些典型症状？" --code leaf_blast
python -m scripts.test_detector uploads/test.png
python -m scripts.test_tool_calling
```

旧的命令行、Direct Pipeline 和 Gradio 入口仍保留：`app.py`、`gradio_app.py`。

## 目录重点

```text
frontend/                       React + Vite 对话界面
rice_agent/rag/                 LangGraph、三级路由、滑动窗口
rice_agent/evaluation/          忠实度门禁、RAGAS-light、JSONL 存储
rice_agent/web/api.py           FastAPI、SSE、上传和反馈接口
knowledge/rice_documents/       8 类水稻病虫害知识文档
scripts/benchmark_latency.py    前后响应时间受控基准
web_app.py                      Web 平台启动入口
```

架构和后续工作见 `docs/ARCHITECTURE.md`、`docs/ROADMAP.md`。
