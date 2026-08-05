from rice_agent.config import settings
from rice_agent.domain import CODE_METADATA


def test_knowledge_files_cover_model_classes() -> None:
    actual = {
        path.stem
        for path in settings.knowledge_dir.glob("*.md")
    }
    assert actual == set(CODE_METADATA)
