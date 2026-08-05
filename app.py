from __future__ import annotations

import argparse

from rice_agent.agent.loop import RiceDiseaseAgent
from rice_agent.direct_pipeline import analyze_image_direct


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="水稻病虫害检测与RAG Agent"
    )
    parser.add_argument("--image", help="本地图片路径")
    parser.add_argument(
        "--question",
        default="请检测这张图片并说明可能的病虫害、典型表现和基础管理建议。",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="不用模型工具调用，采用确定性YOLO→RAG→LLM流程",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.direct:
        if not args.image:
            raise SystemExit("--direct模式必须提供--image")

        result = analyze_image_direct(args.image, args.question)
        print(result["answer"])
        print("\n标注图：", result.get("detection", {}).get(
            "annotated_image_path"
        ))
        return

    agent = RiceDiseaseAgent(verbose=True)

    if args.image:
        prompt = f"{args.question}\n图片路径：{args.image}"
        print("\nAgent：", agent.chat(prompt))
        return

    print("水稻病虫害Agent已启动。输入exit退出，reset清空对话。")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("对话已重置。")
            continue

        print("\nAgent：", agent.chat(user_input))


if __name__ == "__main__":
    main()
