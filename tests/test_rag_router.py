import pytest

from rice_agent.rag.router import RouteMode, decide_route


def test_precise_route_for_high_score() -> None:
    decision = decide_route(
        [{"relevance_score": 0.92}],
        precise_threshold=0.78,
        reference_threshold=0.45,
    )
    assert decision.mode is RouteMode.PRECISE_HIT
    assert decision.qualified_chunks == 1


def test_reference_route_for_medium_score() -> None:
    decision = decide_route(
        [{"relevance_score": 0.62}],
        precise_threshold=0.78,
        reference_threshold=0.45,
    )
    assert decision.mode is RouteMode.REFERENCE_GENERATION


def test_inference_route_when_context_is_weak() -> None:
    decision = decide_route(
        [{"relevance_score": 0.22}],
        precise_threshold=0.78,
        reference_threshold=0.45,
    )
    assert decision.mode is RouteMode.AI_INFERENCE


def test_invalid_thresholds_fail_fast() -> None:
    with pytest.raises(ValueError):
        decide_route([], precise_threshold=0.4, reference_threshold=0.6)
