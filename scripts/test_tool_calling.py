from __future__ import annotations

from langchain_deepseek import ChatDeepSeek

from rice_agent.agent.tools import search_rice_knowledge
from rice_agent.config import settings


if __name__ == "__main__":
    if not settings.deepseek_api_key:
        raise SystemExit("请先配置DEEPSEEK_API_KEY")

    llm = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    ).bind_tools([search_rice_knowledge])

    response = llm.invoke(
        "调用知识检索工具查询稻瘟病的典型症状。"
    )

    print("content:", response.content)
    print("tool_calls:", response.tool_calls)
