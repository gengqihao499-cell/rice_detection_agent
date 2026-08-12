import json

import pytest
from langchain_core.messages import AIMessage

from rice_agent.rag.query_expansion import QueryEnhancer


class FakeQueryLlm:
    def with_config(self, config: dict) -> "FakeQueryLlm":
        return self

    async def ainvoke(self, messages: list) -> AIMessage:
        return AIMessage(
            content=json.dumps(
                {
                    "rewritten_query": "水稻叶片梭形灰白病斑的病害鉴别",
                    "hyde_document": "典型知识条目会描述梭形病斑、灰白中心和红褐边缘。",
                    "multi_queries": [
                        "水稻梭形病斑症状",
                        "叶瘟易发条件",
                        "叶瘟与胡麻斑鉴别",
                    ],
                },
                ensure_ascii=False,
            )
        )


@pytest.mark.asyncio
async def test_query_enhancer_builds_hyde_and_multi_queries() -> None:
    plan = await QueryEnhancer(FakeQueryLlm()).enhance(
        "这个梭形斑是什么？",
        [{"role": "user", "content": "我的水稻叶片中间发白"}],
    )
    assert plan.method == "llm_hyde_multi_query"
    assert plan.hyde_document
    assert len(plan.multi_queries) == 3
    assert plan.rewritten_query.startswith("水稻")
    assert len(plan.vector_queries) >= 5


@pytest.mark.asyncio
async def test_query_enhancer_has_deterministic_no_llm_fallback() -> None:
    plan = await QueryEnhancer(None).enhance("叶片上有褐斑")
    assert plan.rewritten_query == "叶片上有褐斑"
    assert plan.hyde_document == ""
    assert len(plan.multi_queries) == 3
