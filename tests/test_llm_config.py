from langchain_deepseek import ChatDeepSeek


def test_current_deepseek_constructor_arguments() -> None:
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key="sk-test-only",
        base_url="https://api.deepseek.com",
        temperature=0,
        max_retries=0,
    )
    assert model.model_name == "deepseek-chat"
