from rice_agent.rag.memory import ConversationTurn, SlidingWindowConversationStore


def test_sliding_window_keeps_recent_turns() -> None:
    store = SlidingWindowConversationStore(max_turns=2, max_chars=1000)
    session_id = store.create_session()
    for index in range(4):
        store.append(
            session_id,
            ConversationTurn(user=f"q{index}", assistant=f"a{index}"),
        )

    history = store.history(session_id)
    assert [item["content"] for item in history] == ["q2", "a2", "q3", "a3"]


def test_sliding_window_honors_character_budget() -> None:
    store = SlidingWindowConversationStore(max_turns=5, max_chars=256)
    session_id = store.create_session()
    store.append(session_id, ConversationTurn(user="old" * 80, assistant="old" * 80))
    store.append(session_id, ConversationTurn(user="new", assistant="answer"))
    history = store.history(session_id)
    assert history == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]
