import hashlib
import json
import os
from datetime import datetime, timezone

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
STATE_PATH = os.path.join(STORAGE_DIR, "assistant_state.json")

DEFAULT_USERS = (
    {
        "username": "admin",
        "password": "admin123",
        "role": "admin",
        "display_name": "Администратор",
    },
    {
        "username": "student",
        "password": "student123",
        "role": "student",
        "display_name": "Студент",
    },
    {
        "username": "student2",
        "password": "student234",
        "role": "student",
        "display_name": "Студент 2",
    },
)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def resolve_state_path(state_path=None):
    return state_path or STATE_PATH


def initialize_storage(state_path=None):
    resolved_path = resolve_state_path(state_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    if os.path.exists(resolved_path):
        state = load_state(resolved_path)
    else:
        state = empty_state()

    seed_default_users(state)
    save_state(state, resolved_path)


def empty_state():
    return {
        "next_user_id": 1,
        "next_chat_id": 1,
        "next_message_id": 1,
        "next_unanswered_question_id": 1,
        "users": [],
        "chats": [],
        "unanswered_questions": [],
    }


def load_state(state_path=None):
    resolved_path = resolve_state_path(state_path)

    if not os.path.exists(resolved_path):
        return empty_state()

    with open(resolved_path, "r", encoding="utf-8") as state_file:
        return ensure_state_shape(json.load(state_file))


def save_state(state, state_path=None):
    resolved_path = resolve_state_path(state_path)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    with open(resolved_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2)


def ensure_state_shape(state):
    # Keep older state files compatible when new fields are added later.
    state.setdefault("next_user_id", 1)
    state.setdefault("next_chat_id", 1)
    state.setdefault("next_message_id", 1)
    state.setdefault("next_unanswered_question_id", 1)
    state.setdefault("users", [])
    state.setdefault("chats", [])
    state.setdefault("unanswered_questions", [])

    for item in state["unanswered_questions"]:
        item.setdefault("status", "new")
        item.setdefault("reviewed_at", None)
        item.setdefault("reviewed_by", None)
        item.setdefault("admin_answer", None)
        item.setdefault("student_seen_at", None)

    return state


def seed_default_users(state):
    for user in DEFAULT_USERS:
        existing = next(
            (item for item in state["users"] if item["username"] == user["username"]),
            None,
        )

        if existing is not None:
            continue

        state["users"].append(
            {
                "id": state["next_user_id"],
                "username": user["username"],
                "password_hash": hash_password(user["password"]),
                "role": user["role"],
                "display_name": user["display_name"],
                "created_at": utc_now_iso(),
            }
        )
        state["next_user_id"] += 1


def public_user(user):
    if user is None:
        return None

    # UI code should never receive password hashes.
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "display_name": user["display_name"],
    }


def list_users(state_path=None):
    state = load_state(state_path)
    users = [
        {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "display_name": user["display_name"],
            "created_at": user["created_at"],
        }
        for user in state["users"]
    ]
    users.sort(key=lambda item: (item["role"], item["username"]))
    return users


def authenticate_user(username, password, state_path=None):
    state = load_state(state_path)
    user = next(
        (item for item in state["users"] if item["username"] == username.strip()),
        None,
    )

    if user is None:
        return None

    if user["password_hash"] != hash_password(password):
        return None

    return public_user(user)


def register_student(username, password, display_name, state_path=None):
    normalized_username = username.strip()
    normalized_display_name = display_name.strip()

    if not normalized_username:
        raise ValueError("Введите логин.")

    if len(normalized_username) < 3:
        raise ValueError("Логин должен содержать минимум 3 символа.")

    if not normalized_display_name:
        raise ValueError("Введите имя пользователя.")

    if len(password) < 6:
        raise ValueError("Пароль должен содержать минимум 6 символов.")

    state = load_state(state_path)
    existing = next(
        (item for item in state["users"] if item["username"].lower() == normalized_username.lower()),
        None,
    )

    if existing is not None:
        raise ValueError("Пользователь с таким логином уже существует.")

    user = {
        "id": state["next_user_id"],
        "username": normalized_username,
        "password_hash": hash_password(password),
        "role": "student",
        "display_name": normalized_display_name,
        "created_at": utc_now_iso(),
    }
    state["users"].append(user)
    state["next_user_id"] += 1
    save_state(state, state_path)
    return public_user(user)


def get_user_by_id(user_id, state_path=None):
    state = load_state(state_path)
    user = next((item for item in state["users"] if item["id"] == user_id), None)
    return public_user(user)


def list_user_chats(user_id, state_path=None):
    state = load_state(state_path)
    chats = [
        {
            "id": chat["id"],
            "title": chat["title"],
            "created_at": chat["created_at"],
            "updated_at": chat["updated_at"],
        }
        for chat in state["chats"]
        if chat["user_id"] == user_id
    ]
    chats.sort(key=lambda item: (item["updated_at"], item["id"]), reverse=True)
    return chats


def create_chat(user_id, title, state_path=None):
    state = load_state(state_path)
    chat_id = state["next_chat_id"]
    timestamp = utc_now_iso()

    state["chats"].append(
        {
            "id": chat_id,
            "user_id": user_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [],
        }
    )
    state["next_chat_id"] += 1
    save_state(state, state_path)
    return chat_id


def append_message(chat_id, role, content, sources=None, state_path=None):
    state = load_state(state_path)
    chat = next((item for item in state["chats"] if item["id"] == chat_id), None)

    if chat is None:
        raise ValueError(f"Chat {chat_id} was not found")

    chat["messages"].append(
        {
            "id": state["next_message_id"],
            "role": role,
            "content": content,
            "sources": sources or [],
            "created_at": utc_now_iso(),
        }
    )
    chat["updated_at"] = utc_now_iso()
    state["next_message_id"] += 1
    save_state(state, state_path)


def queue_unanswered_question(user_id, question, chat_id=None, state_path=None):
    normalized_question = " ".join(question.split())
    state = load_state(state_path)

    # Keep one open record per question until the admin marks it as processed.
    existing = next(
        (
            item
            for item in state["unanswered_questions"]
            if item["user_id"] == user_id
            and item.get("chat_id") == chat_id
            and item["question"] == normalized_question
            and item.get("status", "new") == "new"
        ),
        None,
    )

    if existing is not None:
        return existing["id"]

    question_id = state["next_unanswered_question_id"]
    state["unanswered_questions"].append(
        {
            "id": question_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "question": normalized_question,
            "created_at": utc_now_iso(),
            "status": "new",
            "reviewed_at": None,
            "reviewed_by": None,
        }
    )
    state["next_unanswered_question_id"] += 1
    save_state(state, state_path)
    return question_id


def list_unanswered_questions(state_path=None):
    state = load_state(state_path)
    # Join unanswered questions with user info for the admin panel.
    users_by_id = {user["id"]: user for user in state["users"]}
    questions = []

    for item in state["unanswered_questions"]:
        if item.get("status", "new") != "new":
            continue

        user = users_by_id.get(item["user_id"])
        questions.append(
            {
                "id": item["id"],
                "user_id": item["user_id"],
                "student_name": user["display_name"] if user else f"Пользователь {item['user_id']}",
                "student_username": user["username"] if user else "unknown",
                "chat_id": item.get("chat_id"),
                "question": item["question"],
                "created_at": item["created_at"],
            }
        )

    questions.sort(key=lambda item: (item["created_at"], item["id"]), reverse=True)
    return questions


def answer_unanswered_question(question_id, admin_user_id, answer, state_path=None):
    state = load_state(state_path)
    question = next((item for item in state["unanswered_questions"] if item["id"] == question_id), None)

    if question is None:
        return False

    if question.get("status", "new") != "new":
        return False

    question["status"] = "answered"
    question["reviewed_at"] = utc_now_iso()
    question["reviewed_by"] = admin_user_id
    question["admin_answer"] = answer
    question["student_seen_at"] = None

    save_state(state, state_path)
    return True


def list_answered_questions(user_id, state_path=None, unseen_only=True):
    state = load_state(state_path)
    items = []

    for question in state["unanswered_questions"]:
        if question["user_id"] != user_id:
            continue

        if question.get("status") != "answered":
            continue

        if unseen_only and question.get("student_seen_at") is not None:
            continue

        items.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answer": question.get("admin_answer") or "",
                "answered_at": question.get("reviewed_at"),
                "chat_id": question.get("chat_id"),
            }
        )

    items.sort(key=lambda item: (item["answered_at"] or "", item["id"]), reverse=True)
    return items


def mark_answered_question_seen(question_id, user_id, state_path=None):
    state = load_state(state_path)
    question = next(
        (
            item
            for item in state["unanswered_questions"]
            if item["id"] == question_id and item["user_id"] == user_id
        ),
        None,
    )

    if question is None:
        return False

    if question.get("status") != "answered":
        return False

    if question.get("student_seen_at") is not None:
        return False

    question["student_seen_at"] = utc_now_iso()
    save_state(state, state_path)
    return True


def reject_unanswered_question(question_id, admin_user_id, state_path=None):
    state = load_state(state_path)
    question = next((item for item in state["unanswered_questions"] if item["id"] == question_id), None)

    if question is None:
        return False

    if question.get("status", "new") != "new":
        return False

    question["status"] = "rejected"
    question["reviewed_at"] = utc_now_iso()
    question["reviewed_by"] = admin_user_id
    save_state(state, state_path)
    return True


def get_chat_messages(chat_id, user_id, state_path=None):
    state = load_state(state_path)
    chat = next(
        (item for item in state["chats"] if item["id"] == chat_id and item["user_id"] == user_id),
        None,
    )

    if chat is None:
        return []

    return list(chat["messages"])


def save_question_answer(user_id, question, answer, sources, chat_id=None, state_path=None):
    resolved_chat_id = chat_id or create_chat(user_id, build_chat_title(question), state_path=state_path)
    append_message(resolved_chat_id, "user", question, state_path=state_path)
    append_message(resolved_chat_id, "assistant", answer, sources=sources, state_path=state_path)
    return resolved_chat_id


def build_chat_title(question):
    compact = " ".join(question.split())
    return compact[:60] + ("..." if len(compact) > 60 else "")
