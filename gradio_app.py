from __future__ import annotations

import json

import gradio as gr

from rice_agent.direct_pipeline import analyze_image_direct


def run_analysis(
    image_path: str | None,
    question: str,
):
    if not image_path:
        return None, "请先上传图片。", "{}"

    result = analyze_image_direct(
        image_path=image_path,
        question=question.strip()
        or "请分析可能的病虫害并给出基础管理建议。",
    )
    detection = result.get("detection", {})

    return (
        detection.get("annotated_image_path"),
        result.get("answer", ""),
        json.dumps(
            {
                "detection": detection,
                "evidence": result.get("evidence", []),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )


with gr.Blocks(title="水稻病虫害智能检测") as demo:
    gr.Markdown(
        """
        # 水稻病虫害智能检测与 RAG 问答

        上传水稻图片后，系统会调用公开 YOLO11L 模型，
        再检索本地知识库生成辅助说明。结果不替代专业植保诊断。
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="filepath",
            label="上传水稻图片",
        )
        annotated_output = gr.Image(
            type="filepath",
            label="检测结果图",
        )

    question_input = gr.Textbox(
        label="问题",
        value="请说明可能的病虫害、典型表现和基础管理建议。",
        lines=3,
    )
    analyze_button = gr.Button("开始分析", variant="primary")
    answer_output = gr.Markdown(label="分析结果")
    json_output = gr.Code(
        label="结构化结果",
        language="json",
    )

    analyze_button.click(
        fn=run_analysis,
        inputs=[image_input, question_input],
        outputs=[annotated_output, answer_output, json_output],
    )


if __name__ == "__main__":
    demo.launch()
