from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_deepseek import ChatDeepSeek

from rice_agent.agent.prompts import SYSTEM_PROMPT
from rice_agent.agent.tools import TOOLS, TOOL_MAP
from rice_agent.config import settings


def extract_message_text(message: AIMessage) -> str:
    content = message.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: list[str] = []

        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)

        return "\n".join(texts).strip()

    return str(content).strip()


def normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        parsed = json.loads(arguments)

        if not isinstance(parsed, dict):
            raise ValueError("工具参数必须是JSON对象")

        return parsed

    raise ValueError(
        f"不支持的工具参数类型：{type(arguments).__name__}"
    )


class RiceDiseaseAgent:
    """原生 Tool Calling Agentic Loop。"""

    def __init__(
        self,
        max_steps: int | None = None,
        verbose: bool = True,
    ) -> None:
        if not settings.deepseek_api_key:
            raise RuntimeError(
                "缺少DEEPSEEK_API_KEY，请复制.env.example为.env并配置"
            )

        self.max_steps = max_steps or settings.max_agent_steps
        self.verbose = verbose

        self.llm = ChatDeepSeek(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
            max_retries=2,
        )
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT.strip())
        ]

    def reset(self) -> None:
        self.messages = [
            SystemMessage(content=SYSTEM_PROMPT.strip())
        ]

    def _tool_message(
        self,
        tool_call: dict[str, Any],
    ) -> ToolMessage:
        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(
            tool_call.get("id")
            or f"call_{uuid4().hex}"
        )

        try:
            arguments = normalize_tool_arguments(
                tool_call.get("args", {})
            )
        except Exception as exc:
            payload = {
                "success": False,
                "error": "invalid_tool_arguments",
                "message": str(exc),
            }
            return ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                tool_call_id=tool_call_id,
            )

        selected_tool = TOOL_MAP.get(tool_name)

        if selected_tool is None:
            payload = {
                "success": False,
                "error": "unknown_tool",
                "message": f"不存在工具：{tool_name}",
                "allowed_tools": sorted(TOOL_MAP),
            }
            return ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                tool_call_id=tool_call_id,
            )

        if self.verbose:
            print(f"\n[调用工具] {tool_name}")
            print(
                json.dumps(
                    arguments,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        try:
            tool_result = selected_tool.invoke(arguments)
            payload = {
                "success": True,
                "tool_name": tool_name,
                "result": tool_result,
            }
        except Exception as exc:
            payload = {
                "success": False,
                "tool_name": tool_name,
                "error": type(exc).__name__,
                "message": str(exc),
            }

        content = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )

        if len(content) > settings.max_tool_result_chars:
            content = (
                content[:settings.max_tool_result_chars]
                + "\n...[工具结果过长，已截断]"
            )

        if self.verbose:
            print("[工具结果]")
            print(content)

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
        )

    def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))
        executed_calls: set[str] = set()

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n========== Agent Step {step} ==========")

            ai_message: AIMessage = self.llm_with_tools.invoke(
                self.messages
            )
            self.messages.append(ai_message)
            tool_calls = ai_message.tool_calls or []

            if not tool_calls:
                final_answer = extract_message_text(ai_message)

                if not final_answer:
                    return "模型没有返回文本，也没有请求调用工具。"

                return final_answer

            for tool_call in tool_calls:
                signature = json.dumps(
                    {
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )

                if signature in executed_calls:
                    duplicate = {
                        "success": False,
                        "error": "duplicate_tool_call",
                        "message": "已经执行过完全相同的工具调用。",
                    }
                    self.messages.append(
                        ToolMessage(
                            content=json.dumps(
                                duplicate,
                                ensure_ascii=False,
                            ),
                            tool_call_id=str(
                                tool_call.get("id")
                                or f"call_{uuid4().hex}"
                            ),
                        )
                    )
                    continue

                executed_calls.add(signature)
                self.messages.append(
                    self._tool_message(tool_call)
                )

        return (
            f"Agent已经执行{self.max_steps}步，"
            "但仍未生成最终答案。"
        )
