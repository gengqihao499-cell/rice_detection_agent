from __future__ import annotations

from typing import Any


CLASS_METADATA: dict[int, dict[str, Any]] = {
    0: {
        "code": "bacterial_leaf_blight",
        "display_name_zh": "水稻白叶枯病",
        "kind": "disease",
        "expected_token": "Bacterial_Leaf_Blight",
    },
    1: {
        "code": "brown_spot",
        "display_name_zh": "水稻胡麻斑病",
        "kind": "disease",
        "expected_token": "Brown_Spot",
    },
    2: {
        "code": "healthy_leaf",
        "display_name_zh": "健康水稻",
        "kind": "healthy",
        "expected_token": "HealthyLeaf",
    },
    3: {
        "code": "leaf_blast",
        "display_name_zh": "稻瘟病（叶瘟）",
        "kind": "disease",
        "expected_token": "Leaf_Blast",
    },
    4: {
        "code": "leaf_scald",
        "display_name_zh": "水稻叶部病害 Leaf Scald（沿用模型原标签）",
        "kind": "disease",
        "expected_token": "Leaf_Scald",
    },
    5: {
        "code": "narrow_brown_leaf_spot",
        "display_name_zh": "水稻窄褐斑病",
        "kind": "disease",
        "expected_token": "Narrow_Brown_Leaf_Spot",
    },
    6: {
        "code": "neck_blast",
        "display_name_zh": "水稻穗颈瘟",
        "kind": "disease",
        "expected_token": "Neck_Blast",
    },
    7: {
        "code": "rice_hispa",
        "display_name_zh": "水稻 Rice Hispa 虫害",
        "kind": "pest",
        "expected_token": "Rice_Hispa",
    },
}


CODE_METADATA = {
    item["code"]: {"class_id": class_id, **item}
    for class_id, item in CLASS_METADATA.items()
}


def metadata_for_class(
    class_id: int,
    raw_name: str,
) -> dict[str, Any]:
    known = CLASS_METADATA.get(class_id)

    if known is not None:
        return {"class_id": class_id, "raw_class_name": raw_name, **known}

    return {
        "class_id": class_id,
        "raw_class_name": raw_name,
        "code": f"class_{class_id}",
        "display_name_zh": raw_name,
        "kind": "unknown",
        "expected_token": "",
    }


def validate_model_names(
    model_names: dict[int, str] | list[str],
) -> list[str]:
    if isinstance(model_names, list):
        normalized = {index: value for index, value in enumerate(model_names)}
    else:
        normalized = {int(key): str(value) for key, value in model_names.items()}

    warnings: list[str] = []

    for class_id, expected in CLASS_METADATA.items():
        actual = normalized.get(class_id)

        if actual is None:
            warnings.append(f"模型缺少类别ID {class_id}")
            continue

        token = expected["expected_token"]

        if token.lower() not in actual.lower():
            warnings.append(
                f"类别ID {class_id}名称异常：实际={actual!r}，"
                f"期望包含={token!r}"
            )

    if len(normalized) != len(CLASS_METADATA):
        warnings.append(
            f"模型类别数量为{len(normalized)}，项目映射数量为"
            f"{len(CLASS_METADATA)}"
        )

    return warnings
