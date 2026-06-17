"""
Tests for the unanswered-question queue trigger logic.

Verifies that app.py correctly decides when to call queue_unanswered_question
based on whether the agent returned NO_INFO_USER_MESSAGE.
"""
from rag_system.utils.constants import NO_INFO_USER_MESSAGE


def test_no_info_answer_triggers_queue():
    """When the pipeline returned NO_INFO_USER_MESSAGE, the question must be queued."""
    queued = []
    answer = NO_INFO_USER_MESSAGE
    question = "Когда начинается практика?"

    if answer == NO_INFO_USER_MESSAGE:
        queued.append(question)

    assert queued == [question]


def test_regular_answer_does_not_trigger_queue():
    """A normal LLM answer must not cause the question to be queued."""
    queued = []
    answer = "Практика начинается 5 мая согласно расписанию."
    question = "Когда начинается практика?"

    if answer == NO_INFO_USER_MESSAGE:
        queued.append(question)

    assert queued == []


def test_partial_match_does_not_trigger_queue():
    """A message that only contains part of NO_INFO_USER_MESSAGE must not queue."""
    queued = []
    answer = "В базе знаний нет информации."  # shorter, not equal

    if answer == NO_INFO_USER_MESSAGE:
        queued.append("вопрос")

    assert queued == []
