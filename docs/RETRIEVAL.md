# 混合检索、HyDE 与嵌入模型选型

## 在线检索流程

```text
用户 Query + 最近对话
  → 独立问题改写
  → HyDE 假设知识段（只用于检索，不作为答案或证据）
  → 3 条 Multi-Query 扩展
  → 多路并发召回
       ├─ Embedding + Chroma：原 Query / 改写 Query / HyDE / Multi-Query
       └─ BM25：原 Query / 改写 Query / Multi-Query
  → 子块命中后回填父块
  → RRF 融合并按 parent_id 去重
  → Fusion Top-20
  → BAAI/bge-reranker-base CrossEncoder 精排
  → Final Top-4（配置范围限制为 3–5）
```

查询增强只需一次 LLM 调用，同时返回 `rewritten_query`、`hyde_document` 和
`multi_queries`，避免三段串行调用放大延迟。没有配置 LLM 时仍执行确定性的三角度
Multi-Query；HyDE 自动关闭。HyDE 内容永远不会直接发送给用户，也不能作为忠实度证据。

## RRF 公式

对每个候选知识块 `d`：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

项目默认 `k=60`。`rank_i(d)` 是知识块在第 `i` 条召回通道中的名次；未出现在该通道
则不计分。融合时优先使用稳定的 `parent_id` 去重，没有父 ID 时退回 `chunk_id`，最后
才使用规范化文本。这样同一父段的多个高分子块不会重复占据候选位。

## Parent-child chunking

索引建立时先按嵌入模型 tokenizer 计数；如果 tokenizer 暂不可用，使用中英文正则近似
计数并把方法写入索引 manifest，便于发现环境差异。

1. 每篇源文档切成约 500-token 父块，重叠 50 token，父块保存在
   `data/rice_chroma_db/parents.jsonl`，不直接生成向量。
2. 每个父块继续切成约 150-token 子块，重叠 25 token；Chroma 和 BM25 都只索引子块。
3. 子块带稳定的 `child_id` 和 `parent_id`。召回后按 `parent_id` 聚合，保留最多 3 个
   命中子块作为可解释证据，同时把完整父块交给 ReRank 和生成模型。
4. `manifest.json` 记录文档数、父/子块数、最大实际 token、分块参数和 tokenizer。

150-token 子块更容易精确定位症状或防治措施；500-token 父块能恢复同一段落附近的条件、
限制和上下文，减少只命中半句话造成的误读。修改 `.env` 中四项分块参数后，摘要变化会
自动触发重建索引。

## 嵌入模型比较

### 候选模型

| 模型 | 向量维度 | 上下文 | 特点与使用判断 |
|---|---:|---:|---|
| `BAAI/bge-small-zh-v1.5` | 512 | 512 | 体积/延迟最低，适合 CPU 轻量部署；C-MTEB Retrieval 61.77 |
| `BAAI/bge-base-zh-v1.5` | 768 | 512 | 中文精度与成本折中；C-MTEB Retrieval 69.49 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 512 | 中文 v1.5 精度优先；C-MTEB Retrieval 70.46，边际收益较小 |
| `BAAI/bge-m3` | 1024 | 8192 | 100+ 语言、长文本，原生支持 dense/sparse/multi-vector；本项目先比较 dense 输出 |
| `Qwen/Qwen3-Embedding-0.6B` | 32–1024 | 32768 | 100+ 语言、指令感知、MRL 可变维度；资源高于 BGE-small |

BGE v1.5 数字来自 [BAAI 官方模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)，
属于公共基准而非本项目实测。BGE-M3 的维度、长度和多功能能力见
[BGE-M3 官方模型卡](https://huggingface.co/BAAI/bge-m3)；Qwen3 的语言、上下文、
MRL 和指令说明见 [Qwen3-Embedding 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。

### 项目内实测

评测集包含 8 类水稻病虫害和 24 条人工标注 Query。模型公平比较时只编码这 8 类已标注
知识，不把开放论文中尚未标注 `disease_code` 的子块混入排名；相关性按
`relevant_disease_codes` 判断。每次运行都会输出 Hit@1/3/5、MRR@5、维度、加载/编码
耗时和 Query 平均/P95 延迟。

| 模型/链路 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | CPU 平均延迟 |
|---|---:|---:|---:|---:|---:|
| BGE-small + 150-token 子块（当前实测） | 0.8333 | 0.9583 | 0.9583 | 0.8889 | 13.94 ms/Query |
| BGE-small 直接向量编码（旧分块基线） | 0.8750 | 0.9167 | 1.0000 | 0.9167 | 17.17 ms/Query |
| Chroma 向量召回 | 0.8750 | 0.9583 | 1.0000 | 0.9201 | 43.29 ms |
| BM25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.29 ms |
| Multi-Query + Vector + BM25 + RRF + 词法 ReRank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 66.05 ms |

该集合较小且问题包含较明显的领域词，因此 BM25 表现很强；它只能证明当前知识库上的
回归效果，不能外推为线上准确率。上线前应扩展到至少 200 条包含口语、省略、错别字、
跨病害混淆和无答案问题的盲测集。

本机使用 `--local-files-only` 执行五模型配置时，只有 BGE-small 权重完整并得到上表新结果；
BGE-base/large、BGE-M3、Qwen3-0.6B 因本地权重缺失而在
`outputs/embedding_benchmark.json` 中记录 `OSError`。这表示“尚未实测”，不是模型效果差。
下载相应权重后重跑同一命令即可得到可横向比较的 Hit@K、MRR 和延迟。

## 选择结论

默认继续使用 `bge-small-zh-v1.5` 作为“下载后即可 CPU 运行”的安全基线。10MB 知识库
完成后，优先比较 BGE-base、BGE-M3 和 Qwen3-0.6B：base 是中文低风险升级，M3 更适合
中英混合论文和后续原生混合检索，Qwen3 更适合长上下文及 instruction-aware Query。
Qwen3 已保留完整配置和指令模板，但仓库不附带权重；需由使用者首次运行时下载。

选择时不要只看 Hit@5：先要求 Hit@3/MRR 提升，再比较 CPU/GPU P95、索引维度和内存。
如果 Qwen3/BGE-M3 在盲测集没有稳定提升，保留 BGE-small 会更符合轻量部署目标。

最终排序默认使用 `BAAI/bge-reranker-base`。模型首次使用需要下载；加载失败时系统会明确
标记 `lexical_fallback`，使用 Query 覆盖率、原召回分和归一化 RRF 分进行可解释精排，
而不是静默跳过 ReRank。

## 复现命令

```powershell
python -m scripts.benchmark_embeddings
$env:RERANKER_ENABLED="false"
python -m scripts.benchmark_retrieval_pipeline
```

默认候选来自 `.env` 的 `EMBEDDING_BENCHMARK_MODELS`。如果只想查看哪些权重已在本地：

```powershell
python -m scripts.benchmark_embeddings --local-files-only
```

指定 Qwen3 实际评测：

```powershell
python -m scripts.benchmark_embeddings --models Qwen/Qwen3-Embedding-0.6B
```

模型下载或显存不足不会中止整组评测，失败模型会在 JSON 报告中保留异常类型和说明。

## 关键配置

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `RAG_CANDIDATE_TOP_K` | 8 | 每条召回通道候选数 |
| `RAG_FUSION_TOP_K` | 20 | RRF 后送入 ReRank 的数量 |
| `RAG_FINAL_TOP_K` | 4 | 最终上下文数量，运行时限制 3–5 |
| `RRF_K` | 60 | RRF 平滑常数 |
| `MULTI_QUERY_COUNT` | 3 | 扩展 Query 数 |
| `HYDE_ENABLED` | true | 是否把假设答案用于向量检索 |
| `RERANKER_MODEL_NAME` | BAAI/bge-reranker-base | CrossEncoder 模型 |
| `EMBEDDING_LOCAL_FILES_ONLY` | false | 首次下载后设 true 可阻止 Hugging Face 联网检查 |
| `RAG_PARENT_CHUNK_TOKENS` | 500 | 返回生成模型的父块目标 token |
| `RAG_PARENT_OVERLAP_TOKENS` | 50 | 父块重叠 token |
| `RAG_CHILD_CHUNK_TOKENS` | 150 | 向量/BM25 索引子块目标 token |
| `RAG_CHILD_OVERLAP_TOKENS` | 25 | 子块重叠 token |
| `RAG_TOKENIZER_MODEL_NAME` | 空 | 留空时复用嵌入模型 tokenizer |
