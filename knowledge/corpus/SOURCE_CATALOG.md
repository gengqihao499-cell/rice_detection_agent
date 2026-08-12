# 可下载语料来源

## PMC Open Access Subset

- 官方说明：https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- 检索与全文接口：https://dev.europepmc.org/RestfulWebService
- 本项目默认查询：标题或摘要包含 `rice` / `Oryza sativa`，同时包含病害、病原、稻瘟、
  白叶枯、虫害、真菌、细菌、病毒、胁迫或管理等主题词，且全文属于 Open Access Subset。
- 下载器仅保留 CC0、CC BY、CC BY-SA；跳过 NC、ND 和无法机器确认许可的文章。
- 每篇文章在 `knowledge/corpus/pmc/manifest.jsonl` 中记录 PMCID、DOI、作者、期刊、
  原文 URL、许可、SHA-256 和本地文件名。

下载后的全文和逐篇 manifest 属于运行数据，不提交 Git；`.gitkeep` 和本来源说明会提交。
