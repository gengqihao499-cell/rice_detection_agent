import pytest

from rice_agent.evaluation.ragas_light import EvaluationInput, RagasLightEvaluator


@pytest.mark.asyncio
async def test_lexical_evaluator_returns_four_scores() -> None:
    evaluator = RagasLightEvaluator(llm=None, faithfulness_target=0.9)
    result = await evaluator.evaluate(
        EvaluationInput(
            session_id="session-1234",
            turn_id="turn-123456",
            question="稻瘟病叶片有什么症状？",
            answer="稻瘟病叶片可出现梭形病斑。",
            contexts=["稻瘟病叶片常见梭形病斑，中央灰白，边缘褐色。"],
        )
    )
    assert 0 <= result.faithfulness <= 1
    assert 0 <= result.answer_relevancy <= 1
    assert 0 <= result.context_precision <= 1
    assert 0 <= result.context_recall <= 1
    assert result.reference_mode == "online_proxy"
    assert result.method == "lexical_fallback"


@pytest.mark.asyncio
async def test_reference_answer_enables_gold_recall_mode() -> None:
    evaluator = RagasLightEvaluator(llm=None)
    result = await evaluator.evaluate(
        EvaluationInput(
            session_id="session-1234",
            turn_id="turn-123456",
            question="症状？",
            answer="叶片有梭形斑。",
            contexts=["叶片有梭形病斑。"],
            reference_answer="叶片出现梭形病斑。",
        )
    )
    assert result.reference_mode == "gold_reference"
