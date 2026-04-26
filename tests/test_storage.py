from app.storage import (
    answer_unanswered_question,
    authenticate_user,
    get_chat_messages,
    get_user_by_id,
    initialize_storage,
    list_answered_questions,
    list_unanswered_questions,
    list_users,
    list_user_chats,
    mark_answered_question_seen,
    queue_unanswered_question,
    reject_unanswered_question,
    register_student,
    save_question_answer,
)


def test_initialize_storage_seeds_default_users(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))

    admin = authenticate_user("admin", "admin123", str(state_path))
    student = authenticate_user("student", "student123", str(state_path))
    student2 = authenticate_user("student2", "student234", str(state_path))

    assert admin is not None
    assert admin["role"] == "admin"
    assert student is not None
    assert student["role"] == "student"
    assert student2 is not None
    assert student2["role"] == "student"


def test_authenticate_user_rejects_wrong_password(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))

    assert authenticate_user("student", "wrong-password", str(state_path)) is None


def test_save_question_answer_persists_chat_and_messages(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))
    student = authenticate_user("student", "student123", str(state_path))

    chat_id = save_question_answer(
        student["id"],
        "Как попасть в OpenEdu?",
        "Нужно написать на почту openedu.",
        ["Источник 1"],
        state_path=str(state_path),
    )

    chats = list_user_chats(student["id"], str(state_path))
    messages = get_chat_messages(chat_id, student["id"], str(state_path))

    assert len(chats) == 1
    assert chats[0]["id"] == chat_id
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Как попасть в OpenEdu?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["sources"] == ["Источник 1"]


def test_get_chat_messages_returns_only_owner_chat(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))
    student = authenticate_user("student", "student123", str(state_path))
    admin = authenticate_user("admin", "admin123", str(state_path))
    chat_id = save_question_answer(
        student["id"],
        "Что по практике?",
        "Напишите ответственному.",
        [],
        state_path=str(state_path),
    )

    assert get_chat_messages(chat_id, admin["id"], str(state_path)) == []


def test_get_user_by_id_returns_seeded_user(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))
    student = authenticate_user("student", "student123", str(state_path))

    loaded = get_user_by_id(student["id"], str(state_path))

    assert loaded["username"] == "student"


def test_each_student_has_own_chat_history(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))
    student1 = authenticate_user("student", "student123", str(state_path))
    student2 = authenticate_user("student2", "student234", str(state_path))

    save_question_answer(
        student1["id"],
        "Вопрос первого студента",
        "Ответ первому студенту",
        ["Источник 1"],
        state_path=str(state_path),
    )
    save_question_answer(
        student2["id"],
        "Вопрос второго студента",
        "Ответ второму студенту",
        ["Источник 2"],
        state_path=str(state_path),
    )

    chats1 = list_user_chats(student1["id"], str(state_path))
    chats2 = list_user_chats(student2["id"], str(state_path))
    messages1 = get_chat_messages(chats1[0]["id"], student1["id"], str(state_path))
    messages2 = get_chat_messages(chats2[0]["id"], student2["id"], str(state_path))

    assert len(chats1) == 1
    assert len(chats2) == 1
    assert messages1[0]["content"] == "Вопрос первого студента"
    assert messages2[0]["content"] == "Вопрос второго студента"


def test_register_student_creates_student_role(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))

    new_user = register_student("newstudent", "strong123", "Новый студент", str(state_path))
    users = list_users(str(state_path))

    assert new_user["role"] == "student"
    assert any(user["username"] == "newstudent" for user in users)


def test_register_student_rejects_duplicate_username(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))

    try:
        register_student("student", "strong123", "Дубликат", str(state_path))
    except ValueError as exc:
        assert "уже существует" in str(exc)
    else:
        raise AssertionError("Expected duplicate username validation error")


def test_queue_unanswered_question_makes_it_visible_to_admin(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))
    student = authenticate_user("student", "student123", str(state_path))
    chat_id = save_question_answer(
        student["id"],
        "Когда каникулы в следующем семестре?",
        "В базе знаний нет информации.",
        [],
        state_path=str(state_path),
    )

    question_id = queue_unanswered_question(
        student["id"],
        "Когда каникулы в следующем семестре?",
        chat_id=chat_id,
        state_path=str(state_path),
    )
    pending = list_unanswered_questions(str(state_path))

    assert question_id is not None
    assert len(pending) == 1
    assert pending[0]["student_username"] == "student"
    assert pending[0]["chat_id"] == chat_id


def test_answer_unanswered_question_hides_it_from_admin_queue_and_creates_student_update(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))
    student = authenticate_user("student", "student123", str(state_path))
    admin = authenticate_user("admin", "admin123", str(state_path))
    chat_id = save_question_answer(
        student["id"],
        "Нет ответа в базе",
        "В базе знаний нет информации.",
        [],
        state_path=str(state_path),
    )
    question_id = queue_unanswered_question(
        student["id"],
        "Нет ответа в базе",
        chat_id=chat_id,
        state_path=str(state_path),
    )

    marked = answer_unanswered_question(
        question_id,
        admin["id"],
        "Это новый ответ администратора.",
        str(state_path),
    )
    pending = list_unanswered_questions(str(state_path))
    updates = list_answered_questions(student["id"], str(state_path))
    messages = get_chat_messages(chat_id, student["id"], str(state_path))

    assert marked is True
    assert pending == []
    assert len(updates) == 1
    assert updates[0]["answer"] == "Это новый ответ администратора."
    assert len(messages) == 2


def test_reject_unanswered_question_hides_it_from_admin_queue(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))
    student = authenticate_user("student", "student123", str(state_path))
    admin = authenticate_user("admin", "admin123", str(state_path))
    question_id = queue_unanswered_question(
        student["id"],
        "Не по теме",
        state_path=str(state_path),
    )

    marked = reject_unanswered_question(question_id, admin["id"], str(state_path))
    pending = list_unanswered_questions(str(state_path))

    assert marked is True
    assert pending == []


def test_mark_answered_question_seen_hides_student_update(tmp_path):
    state_path = tmp_path / "assistant_state.json"

    initialize_storage(str(state_path))
    student = authenticate_user("student", "student123", str(state_path))
    admin = authenticate_user("admin", "admin123", str(state_path))
    question_id = queue_unanswered_question(
        student["id"],
        "Когда будут оценки?",
        state_path=str(state_path),
    )
    answer_unanswered_question(
        question_id,
        admin["id"],
        "Оценки появятся позже.",
        str(state_path),
    )

    before = list_answered_questions(student["id"], str(state_path))
    marked = mark_answered_question_seen(question_id, student["id"], str(state_path))
    after = list_answered_questions(student["id"], str(state_path))

    assert len(before) == 1
    assert marked is True
    assert after == []
