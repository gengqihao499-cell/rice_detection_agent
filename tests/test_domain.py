from rice_agent.domain import CLASS_METADATA, CODE_METADATA


def test_eight_classes_are_defined() -> None:
    assert set(CLASS_METADATA) == set(range(8))


def test_codes_are_unique() -> None:
    codes = [
        value["code"]
        for value in CLASS_METADATA.values()
    ]
    assert len(codes) == len(set(codes))
    assert set(codes) == set(CODE_METADATA)
