import os
import hashlib
from datetime import datetime, timezone
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ai_assistant:ai_assistant_pass@localhost:5432/ai_assistant_db"
)

engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="student")
    display_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(100))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"))
    role = Column(String(10))
    content = Column(Text, nullable=False)
    sources = Column(JSONB, default=[])
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UnansweredQuestion(Base):
    __tablename__ = "unanswered_questions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"))
    question = Column(Text, nullable=False)
    status = Column(String(20), default="new")
    reviewed_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at = Column(DateTime)
    admin_answer = Column(Text)
    student_seen_at = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


DEFAULT_USERS = (
    {"username": "admin", "password": "admin123", "role": "admin", "display_name": "Администратор"},
    {"username": "student", "password": "student123", "role": "student", "display_name": "Студент"},
    {"username": "student2", "password": "student234", "role": "student", "display_name": "Студент 2"},
)


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    with get_db_session() as session:
        for user_data in DEFAULT_USERS:
            existing = session.query(User).filter(User.username == user_data["username"]).first()
            if not existing:
                session.add(User(
                    username=user_data["username"],
                    password_hash=hash_password(user_data["password"]),
                    role=user_data["role"],
                    display_name=user_data["display_name"],
                ))


def list_users():
    with get_db_session() as session:
        users = session.query(User).order_by(User.role, User.username).all()
        return [
            {
                "id": u.id,
                "username": u.username,
                "role": u.role,
                "display_name": u.display_name,
                "created_at": u.created_at.isoformat() if u.created_at else "",
            }
            for u in users
        ]


def authenticate_user(username, password):
    with get_db_session() as session:
        user = session.query(User).filter(User.username == username.strip()).first()
        if not user or user.password_hash != hash_password(password):
            return None
        return {"id": user.id, "username": user.username, "role": user.role, "display_name": user.display_name}


def register_student(username, password, display_name):
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
    with get_db_session() as session:
        existing = session.query(User).filter(User.username == normalized_username).first()
        if existing:
            raise ValueError("Пользователь с таким логином уже существует.")
        user = User(
            username=normalized_username,
            password_hash=hash_password(password),
            role="student",
            display_name=normalized_display_name,
        )
        session.add(user)
        session.flush()
        return {"id": user.id, "username": user.username, "role": user.role, "display_name": user.display_name}


def get_user_by_id(user_id):
    with get_db_session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        return {"id": user.id, "username": user.username, "role": user.role, "display_name": user.display_name}


def list_user_chats(user_id):
    with get_db_session() as session:
        chats = (
            session.query(Chat)
            .filter(Chat.user_id == user_id)
            .order_by(Chat.updated_at.desc(), Chat.id.desc())
            .all()
        )
        return [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at.isoformat() if c.created_at else "",
                "updated_at": c.updated_at.isoformat() if c.updated_at else "",
            }
            for c in chats
        ]


def queue_unanswered_question(user_id, question, chat_id=None):
    normalized_question = " ".join(question.split())
    with get_db_session() as session:
        existing = session.query(UnansweredQuestion).filter(
            UnansweredQuestion.user_id == user_id,
            UnansweredQuestion.chat_id == chat_id,
            UnansweredQuestion.question == normalized_question,
            UnansweredQuestion.status == "new",
        ).first()
        if existing:
            return existing.id
        q = UnansweredQuestion(user_id=user_id, chat_id=chat_id, question=normalized_question)
        session.add(q)
        session.flush()
        return q.id


def list_unanswered_questions():
    with get_db_session() as session:
        questions = (
            session.query(UnansweredQuestion)
            .filter(UnansweredQuestion.status == "new")
            .order_by(UnansweredQuestion.created_at.desc(), UnansweredQuestion.id.desc())
            .all()
        )
        result = []
        for q in questions:
            user = session.query(User).filter(User.id == q.user_id).first()
            result.append({
                "id": q.id,
                "user_id": q.user_id,
                "student_name": user.display_name if user else f"Пользователь {q.user_id}",
                "student_username": user.username if user else "unknown",
                "chat_id": q.chat_id,
                "question": q.question,
                "created_at": q.created_at.isoformat() if q.created_at else "",
            })
        return result


def answer_unanswered_question(question_id, admin_user_id, answer):
    with get_db_session() as session:
        q = session.query(UnansweredQuestion).filter(
            UnansweredQuestion.id == question_id,
            UnansweredQuestion.status == "new",
        ).first()
        if not q:
            return False
        q.status = "answered"
        q.reviewed_by = admin_user_id
        q.reviewed_at = datetime.now(timezone.utc)
        q.admin_answer = answer
        q.student_seen_at = None
        return True


def list_answered_questions(user_id):
    with get_db_session() as session:
        questions = session.query(UnansweredQuestion).filter(
            UnansweredQuestion.user_id == user_id,
            UnansweredQuestion.status == "answered",
            UnansweredQuestion.student_seen_at == None,  # noqa: E711
        ).order_by(UnansweredQuestion.reviewed_at.desc(), UnansweredQuestion.id.desc()).all()
        return [
            {
                "id": q.id,
                "question": q.question,
                "answer": q.admin_answer or "",
                "answered_at": q.reviewed_at.isoformat() if q.reviewed_at else "",
                "chat_id": q.chat_id,
            }
            for q in questions
        ]


def mark_answered_question_seen(question_id, user_id):
    with get_db_session() as session:
        q = session.query(UnansweredQuestion).filter(
            UnansweredQuestion.id == question_id,
            UnansweredQuestion.user_id == user_id,
            UnansweredQuestion.status == "answered",
            UnansweredQuestion.student_seen_at == None,  # noqa: E711
        ).first()
        if not q:
            return False
        q.student_seen_at = datetime.now(timezone.utc)
        return True


def reject_unanswered_question(question_id, admin_user_id):
    with get_db_session() as session:
        q = session.query(UnansweredQuestion).filter(
            UnansweredQuestion.id == question_id,
            UnansweredQuestion.status == "new",
        ).first()
        if not q:
            return False
        q.status = "rejected"
        q.reviewed_by = admin_user_id
        q.reviewed_at = datetime.now(timezone.utc)
        return True


def get_chat_messages(chat_id, user_id):
    with get_db_session() as session:
        chat = session.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
        if not chat:
            return []
        messages = (
            session.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.created_at)
            .all()
        )
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources or [],
                "created_at": m.created_at.isoformat() if m.created_at else "",
            }
            for m in messages
        ]


def save_question_answer(user_id, question, answer, sources, chat_id=None):
    with get_db_session() as session:
        if not chat_id:
            chat = Chat(user_id=user_id, title=_build_chat_title(question))
            session.add(chat)
            session.flush()
            chat_id = chat.id
        else:
            chat = session.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                raise ValueError(f"Chat {chat_id} was not found")

        now = datetime.now(timezone.utc)
        session.add(Message(chat_id=chat_id, role="user", content=question, sources=[]))
        session.add(Message(chat_id=chat_id, role="assistant", content=answer, sources=sources or []))
        chat.updated_at = now
        return chat_id


def _build_chat_title(question):
    compact = " ".join(question.split())
    return compact[:60] + ("..." if len(compact) > 60 else "")
