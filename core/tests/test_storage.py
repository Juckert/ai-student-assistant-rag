import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import (
    answer_unanswered_question,
    authenticate_user,
    get_chat_messages,
    get_user_by_id,
    init_db as initialize_storage,
    list_answered_questions,
    list_unanswered_questions,
    list_users,
    list_user_chats,
    mark_answered_question_seen,
    queue_unanswered_question,
    reject_unanswered_question,
    register_student,
    save_question_answer,
    Base,
    User,
    Chat,
    Message,
    UnansweredQuestion,
    DEFAULT_USERS,
    hash_password,
)

TEST_DATABASE_URL = "postgresql://ai_assistant:ai_assistant_pass@localhost:5432/ai_assistant_test_db"


@pytest.fixture(autouse=True)
def setup_test_db():
    test_engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine)
    with TestSession() as session:
        for user_data in DEFAULT_USERS:
            session.add(User(
                username=user_data["username"],
                password_hash=hash_password(user_data["password"]),
                role=user_data["role"],
                display_name=user_data["display_name"],
            ))
        session.commit()
    import app.database as db_module
    db_module.engine = test_engine
    db_module.SessionLocal = TestSession
    yield
    Base.metadata.drop_all(bind=test_engine)


def test_initialize_storage_seeds_default_users():
    admin = authenticate_user("admin", "admin123")
    student = authenticate_user("student", "student123")
    student2 = authenticate_user("student2", "student234")

    assert admin is not None
    assert admin["role"] == "admin"
    assert student is not None
    assert student["role"] == "student"
    assert student2 is not None
    assert student2["role"] == "student"


def test_authenticate_user_rejects_wrong_password():
    assert authenticate_user("student", "wrong-password") is None


def test_save_question_answer_persists_chat_and_messages():
    student = authenticate_user("student", "student123")

    chat_id = save_question_answer(
        student["id"],
        "Как попасть в OpenEdu?",
        "Нужно написать на почту openedu.",
        ["Источник 1"],
    )

    chats = list_user_chats(student["id"])
    messages = get_chat_messages(chat_id, student["id"])

    assert len(chats) == 1
    assert chats[0]["id"] == chat_id
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Как попасть в OpenEdu?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["sources"] == ["Источник 1"]


def test_get_chat_messages_returns_only_owner_chat():
    student = authenticate_user("student", "student123")
    admin = authenticate_user("admin", "admin123")
    chat_id = save_question_answer(
        student["id"],
        "Что по практике?",
        "Напишите ответственному.",
        [],
    )

    assert get_chat_messages(chat_id, admin["id"]) == []


def test_get_user_by_id_returns_seeded_user():
    student = authenticate_user("student", "student123")

    loaded = get_user_by_id(student["id"])

    assert loaded["username"] == "student"


def test_each_student_has_own_chat_history():
    student1 = authenticate_user("student", "student123")
    student2 = authenticate_user("student2", "student234")

    save_question_answer(
        student1["id"],
        "Вопрос первого студента",
        "Ответ первому студенту",
        ["Источник 1"],
    )
    save_question_answer(
        student2["id"],
        "Вопрос второго студента",
        "Ответ второму студенту",
        ["Источник 2"],
    )

    chats1 = list_user_chats(student1["id"])
    chats2 = list_user_chats(student2["id"])
    messages1 = get_chat_messages(chats1[0]["id"], student1["id"])
    messages2 = get_chat_messages(chats2[0]["id"], student2["id"])

    assert len(chats1) == 1
    assert len(chats2) == 1
    assert messages1[0]["content"] == "Вопрос первого студента"
    assert messages2[0]["content"] == "Вопрос второго студента"


def test_register_student_creates_student_role():
    new_user = register_student("newstudent", "strong123", "Новый студент")
    users = list_users()

    assert new_user["role"] == "student"
    assert any(user["username"] == "newstudent" for user in users)


def test_register_student_rejects_duplicate_username():
    try:
        register_student("student", "strong123", "Дубликат")
    except ValueError as exc:
        assert "уже существует" in str(exc)
    else:
        raise AssertionError("Expected duplicate username validation error")


def test_queue_unanswered_question_makes_it_visible_to_admin():
    student = authenticate_user("student", "student123")
    chat_id = save_question_answer(
        student["id"],
        "Когда каникулы в следующем семестре?",
        "В базе знаний нет информации.",
        [],
    )

    question_id = queue_unanswered_question(
        student["id"],
        "Когда каникулы в следующем семестре?",
        chat_id=chat_id,
    )
    pending = list_unanswered_questions()

    assert question_id is not None
    assert len(pending) == 1
    assert pending[0]["student_username"] == "student"
    assert pending[0]["chat_id"] == chat_id


def test_answer_unanswered_question_hides_it_from_admin_queue_and_creates_student_update():
    student = authenticate_user("student", "student123")
    admin = authenticate_user("admin", "admin123")
    chat_id = save_question_answer(
        student["id"],
        "Нет ответа в базе",
        "В базе знаний нет информации.",
        [],
    )
    question_id = queue_unanswered_question(
        student["id"],
        "Нет ответа в базе",
        chat_id=chat_id,
    )

    marked = answer_unanswered_question(
        question_id,
        admin["id"],
        "Это новый ответ администратора.",
    )
    pending = list_unanswered_questions()
    updates = list_answered_questions(student["id"])
    messages = get_chat_messages(chat_id, student["id"])

    assert marked is True
    assert pending == []
    assert len(updates) == 1
    assert updates[0]["answer"] == "Это новый ответ администратора."
    assert len(messages) == 2


def test_reject_unanswered_question_hides_it_from_admin_queue():
    student = authenticate_user("student", "student123")
    admin = authenticate_user("admin", "admin123")
    question_id = queue_unanswered_question(
        student["id"],
        "Не по теме",
    )

    marked = reject_unanswered_question(question_id, admin["id"])
    pending = list_unanswered_questions()

    assert marked is True
    assert pending == []


def test_mark_answered_question_seen_hides_student_update():
    student = authenticate_user("student", "student123")
    admin = authenticate_user("admin", "admin123")
    question_id = queue_unanswered_question(
        student["id"],
        "Когда будут оценки?",
    )
    answer_unanswered_question(
        question_id,
        admin["id"],
        "Оценки появятся позже.",
    )

    before = list_answered_questions(student["id"])
    marked = mark_answered_question_seen(question_id, student["id"])
    after = list_answered_questions(student["id"])

    assert len(before) == 1
    assert marked is True
    assert after == []
