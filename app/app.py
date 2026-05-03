import os
import sys
from datetime import datetime

import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from storage import (
    answer_unanswered_question,
    authenticate_user,
    get_chat_messages,
    get_user_by_id,
    initialize_storage,
    list_answered_questions,
    list_users,
    list_user_chats,
    list_unanswered_questions,
    mark_answered_question_seen,
    queue_unanswered_question,
    reject_unanswered_question,
    register_student,
    save_question_answer,
)
from knowledge_base import append_manual_qa
from backend_client import (
    is_backend_available,
    get_kb_index_id,
    get_or_create_kb_index,
    upload_files_to_index,
    build_kb_index,
    create_backend_session,
    link_index_to_session,
    rag_chat,
)

st.set_page_config(page_title="ИИ-ассистент студента", layout="wide")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".csv")

# Phrases that indicate the RAG found no relevant information
_NO_INFO_PHRASES = (
    "i don't know",
    "i do not know",
    "no relevant information",
    "cannot find",
    "не нашёл",
    "не найдено",
    "нет информации",
    "не могу найти",
    "не содержит информации",
)


def _is_no_info_answer(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in _NO_INFO_PHRASES)


def get_supported_data_files():
    if not os.path.exists(DATA_DIR):
        return []
    files = []
    for name in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, name)
        if os.path.isfile(path) and name.lower().endswith(SUPPORTED_EXTENSIONS):
            files.append(path)
    return sorted(files)


def format_timestamp(value):
    try:
        timestamp = datetime.fromisoformat(value)
        return timestamp.astimezone().strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError):
        return value


def ensure_runtime_state():
    initialize_storage()

    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None

    if "selected_chat_id" not in st.session_state:
        st.session_state.selected_chat_id = None

    # One backend session per Streamlit session (used for RAG routing context)
    if "backend_session_id" not in st.session_state:
        st.session_state.backend_session_id = None

    # Cached KB index ID to avoid repeated API lookups
    if "kb_index_id" not in st.session_state:
        st.session_state.kb_index_id = None


def _ensure_backend_session(title: str = "Новый диалог") -> bool:
    """
    Create a backend session and link the shared KB index to it.
    Stores session_id and index_id in session_state. Returns True on success.
    """
    if st.session_state.backend_session_id:
        return True

    sid = create_backend_session(title)
    if not sid:
        return False
    st.session_state.backend_session_id = sid

    kb_id = st.session_state.kb_index_id or get_or_create_kb_index()
    if kb_id:
        link_index_to_session(sid, kb_id)
        st.session_state.kb_index_id = kb_id

    return True


def get_current_user():
    auth_user = st.session_state.get("auth_user")
    if not auth_user:
        return None
    fresh_user = get_user_by_id(auth_user["id"])
    if fresh_user is None:
        st.session_state.auth_user = None
        st.session_state.selected_chat_id = None
        return None
    st.session_state.auth_user = fresh_user
    return fresh_user


def render_login_screen():
    st.title("🎓 ИИ-ассистент студента")
    login_tab, register_tab = st.tabs(["🔐 Вход", "📝 Регистрация студента"])

    with login_tab:
        st.subheader("Вход в систему")
        with st.form("login_form"):
            username = st.text_input("Логин")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", use_container_width=True)

        if submitted:
            user = authenticate_user(username, password)
            if user is None:
                st.error("Неверный логин или пароль.")
            else:
                st.session_state.auth_user = user
                st.session_state.selected_chat_id = None
                st.rerun()

        with st.expander("👥 Демо-аккаунты"):
            st.write("Администратор: `admin / admin123`")
            st.write("Студент: `student / student123`")
            st.write("Студент 2: `student2 / student234`")

    with register_tab:
        st.subheader("Создать аккаунт студента")
        with st.form("register_form"):
            display_name = st.text_input("Имя")
            new_username = st.text_input("Логин")
            new_password = st.text_input("Пароль", type="password")
            confirm_password = st.text_input("Повторите пароль", type="password")
            register_submitted = st.form_submit_button("Зарегистрироваться", use_container_width=True)

        if register_submitted:
            if new_password != confirm_password:
                st.error("Пароли не совпадают.")
            else:
                try:
                    user = register_student(new_username, new_password, display_name)
                    st.session_state.auth_user = user
                    st.session_state.selected_chat_id = None
                    st.success("Аккаунт создан. Выполняю вход...")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def render_sidebar(user):
    st.sidebar.title("👤 Профиль")
    st.sidebar.write(f"Пользователь: **{user['display_name']}**")
    st.sidebar.write(f"Роль: **{role_label(user['role'])}**")

    if user["role"] == "admin":
        backend_ok = is_backend_available()
        if backend_ok:
            kb_id = st.session_state.kb_index_id or get_kb_index_id()
            if kb_id:
                st.sidebar.success("База знаний: индекс готов")
            else:
                st.sidebar.warning("База знаний пока не создана")
        else:
            st.sidebar.error("Backend недоступен (порт 8000)")

    if st.sidebar.button("🚪 Выйти", use_container_width=True):
        st.session_state.auth_user = None
        st.session_state.selected_chat_id = None
        st.session_state.backend_session_id = None
        st.rerun()


def role_label(role):
    return "Администратор" if role == "admin" else "Студент"


def render_sources(sources):
    if not sources:
        return
    with st.expander("📚 Источники"):
        for source in sources:
            if isinstance(source, dict):
                text = source.get("text") or source.get("content") or str(source)
                score = source.get("score")
                label = f"{text[:300]}..." if len(text) > 300 else text
                if score is not None:
                    label = f"[релевантность: {score:.2f}] {label}"
                st.write(label)
            else:
                st.write(str(source)[:300])


def render_student_welcome():
    st.markdown("### 👋 Добро пожаловать")
    st.write(
        "Система автоматически отвечает на наиболее массовые и важные вопросы "
        "студентов программ искусственного интеллекта."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Как пользоваться**")
        st.markdown(
            "- Напишите вопрос простыми словами, как задали бы его преподавателю, куратору или одногруппникам.\n"
            "- Ассистент найдёт подходящую информацию в базе знаний и подготовит ответ.\n"
            "- История диалогов сохраняется в вашем аккаунте, поэтому к ней можно вернуться позже."
        )

    with col2:
        st.markdown("**Что важно знать**")
        st.markdown(
            "- Ответы формируются на основе базы знаний.\n"
            "- Если точной информации в базе нет, система честно об этом скажет.\n"
            "- Вы всегда можете открыть прошлые диалоги в боковой панели."
        )

    st.info("💬 Начните новый диалог: введите вопрос в поле ниже.")


def render_student_updates(user):
    answered_questions = list_answered_questions(user["id"])
    if not answered_questions:
        return

    st.subheader("📬 Появились новые ответы")
    for item in answered_questions:
        with st.expander(item["question"], expanded=True):
            st.write(f"Вопрос: **{item['question']}**")
            st.write(f"Ответ администратора: **{item['answer']}**")
            if item["answered_at"]:
                st.write(f"Время обновления: **{format_timestamp(item['answered_at'])}**")
            if st.button("Понятно", key=f"mark_answer_seen_{item['id']}", use_container_width=True):
                mark_answered_question_seen(item["id"], user["id"])
                st.rerun()


def render_student_chat(user):
    st.title("🎓 ИИ-ассистент студента")
    st.caption("💬 Задайте вопрос, и система найдёт ответ в базе знаний и сохранит историю диалога.")

    forwarded_notice = st.session_state.pop("unanswered_forwarded_notice", None)
    if forwarded_notice:
        st.info("📨 Мы не нашли точного ответа и уже передали этот вопрос администратору.")

    chats = list_user_chats(user["id"])
    chat_ids = [0] + [chat["id"] for chat in chats]
    selected_chat_id = st.session_state.get("selected_chat_id")

    if selected_chat_id not in [chat["id"] for chat in chats]:
        selected_chat_id = 0

    selected_index = 0
    if selected_chat_id:
        selected_index = chat_ids.index(selected_chat_id)

    chosen_chat = st.sidebar.radio(
        "💬 История диалогов",
        chat_ids,
        index=selected_index,
        format_func=lambda chat_id: format_chat_label(chat_id, chats),
    )

    st.session_state.selected_chat_id = None if chosen_chat == 0 else chosen_chat

    if st.sidebar.button("➕ Новый диалог", use_container_width=True):
        st.session_state.selected_chat_id = None
        # Reset backend session so the new chat gets a fresh context
        st.session_state.backend_session_id = None
        st.rerun()

    if st.session_state.selected_chat_id is None:
        render_student_updates(user)

    messages = []
    if st.session_state.selected_chat_id is not None:
        messages = get_chat_messages(st.session_state.selected_chat_id, user["id"])

    if not messages:
        render_student_welcome()
    else:
        for message in messages:
            with st.chat_message("user" if message["role"] == "user" else "assistant"):
                st.write(message["content"])
                if message["role"] == "assistant":
                    render_sources(message.get("sources", []))

    question = st.chat_input("💬 Напишите ваш вопрос")
    if not question:
        return

    # Check backend availability before attempting a query
    if not is_backend_available():
        st.error("Backend недоступен. Убедитесь, что сервер запущен на порту 8000.")
        return

    # Ensure we have a backend session with the KB index linked
    if not _ensure_backend_session(title=question[:60]):
        st.error("Не удалось создать сессию на backend. Проверьте, запущен ли сервер.")
        return

    with st.spinner("🔎 Ищу ответ в базе знаний..."):
        try:
            answer, source_docs = rag_chat(st.session_state.backend_session_id, question)

            no_info = _is_no_info_answer(answer)
            if no_info:
                display_answer = (
                    "Точного ответа в базе знаний не нашлось. "
                    "Вопрос передан администратору."
                )
                sources_to_save: list = []
            else:
                display_answer = answer
                sources_to_save = source_docs

            chat_id = save_question_answer(
                user["id"],
                question,
                display_answer,
                sources_to_save,
                chat_id=st.session_state.selected_chat_id,
            )

            if no_info:
                queue_unanswered_question(user["id"], question, chat_id=chat_id)
                st.session_state.unanswered_forwarded_notice = True

            st.session_state.selected_chat_id = chat_id
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def format_chat_label(chat_id, chats):
    if chat_id == 0:
        return "Новый диалог"
    for chat in chats:
        if chat["id"] == chat_id:
            return chat["title"]
    return f"Диалог {chat_id}"


def render_admin_panel(user):
    st.title("🛠️ Панель администратора")
    st.caption("Загрузка и обновление базы знаний")

    users = list_users()
    student_count = len([u for u in users if u["role"] == "student"])
    unanswered_questions = list_unanswered_questions()

    kb_id = st.session_state.kb_index_id or get_kb_index_id()
    st.session_state.kb_index_id = kb_id

    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] { min-height: 120px; }
        div[data-testid="stMetricLabel"] { min-height: 48px; align-items: start; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)
    metrics = [
        ("Backend", "Доступен" if is_backend_available() else "Недоступен"),
        ("Индекс базы знаний", "Готов" if kb_id else "Не создан"),
        ("Файлов в data", len(get_supported_data_files())),
        ("Студентов", student_count),
        ("Вопросов без ответа", len(unanswered_questions)),
    ]

    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.metric(label, value)

    st.subheader("📂 Загрузка файла")
    file = st.file_uploader("Выберите файл", type=["pdf", "txt", "csv"])
    existing_files = get_supported_data_files()

    if existing_files:
        build_options = ["Все файлы из папки data"] + existing_files
        selected_existing_file = st.selectbox(
            "Или выберите файл из папки data",
            build_options,
            format_func=lambda path: (
                path if path == "Все файлы из папки data" else os.path.basename(path)
            ),
        )
    else:
        selected_existing_file = None
        st.info("В папке data пока нет поддерживаемых файлов.")

    if file:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, file.name)
        with open(path, "wb") as target_file:
            target_file.write(file.getbuffer())

        if st.button("Построить индекс по загруженному файлу", use_container_width=True):
            _build_index([path], success_message="База знаний успешно обновлена.")

    elif selected_existing_file and st.button("Построить индекс из data", use_container_width=True):
        source_files = (
            existing_files
            if selected_existing_file == "Все файлы из папки data"
            else [selected_existing_file]
        )
        _build_index(source_files, success_message="База знаний обновлена из папки data.")

    st.subheader("📨 Вопросы без ответа")
    if not unanswered_questions:
        st.success("Сейчас нет новых вопросов, которые нужно добавить в базу знаний.")
    else:
        for item in unanswered_questions:
            with st.expander(item["question"]):
                st.write(f"Логин: `{item['student_username']}`")
                st.write(f"Вопрос: **{item['question']}**")
                st.write(f"Время: **{format_timestamp(item['created_at'])}**")
                admin_answer = st.text_area(
                    "Ответ",
                    key=f"admin_answer_{item['id']}",
                    height=120,
                    placeholder="Введите ответ администратора",
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        "Сохранить ответ в базу",
                        key=f"answer_unanswered_{item['id']}",
                        use_container_width=True,
                    ):
                        normalized_answer = " ".join(admin_answer.split())
                        if not normalized_answer:
                            st.error("Введите ответ перед сохранением.")
                        else:
                            append_manual_qa(item["question"], normalized_answer)
                            rebuilt = _build_index(
                                get_supported_data_files(),
                                success_message="База знаний обновлена, ответ добавлен.",
                            )
                            if rebuilt:
                                answer_unanswered_question(item["id"], user["id"], normalized_answer)
                                st.success("Ответ сохранён и добавлен в базу знаний.")
                                st.rerun()

                with col2:
                    if st.button(
                        "Отклонить как не по теме",
                        key=f"reject_unanswered_{item['id']}",
                        use_container_width=True,
                    ):
                        reject_unanswered_question(item["id"], user["id"])
                        st.success("Вопрос отклонён.")
                        st.rerun()

    st.subheader("👥 Пользователи")
    st.dataframe(
        [
            {
                "ID": u["id"],
                "Логин": u["username"],
                "Имя": u["display_name"],
                "Роль": role_label(u["role"]),
                "Создан": u["created_at"],
            }
            for u in users
        ],
        use_container_width=True,
        hide_index=True,
    )


def _build_index(source_files: list[str], success_message: str) -> bool:
    """Upload files to the backend KB index and trigger RAG indexing. Returns True on success."""
    if not is_backend_available():
        st.error("Backend недоступен. Убедитесь, что сервер запущен на порту 8000.")
        return False

    st.info("Загрузка файлов в базу знаний...")

    try:
        kb_id = get_or_create_kb_index()
        if not kb_id:
            st.error("Не удалось создать индекс на backend.")
            return False

        files_payload: list[tuple[str, bytes]] = []
        for path in source_files:
            with open(path, "rb") as f:
                files_payload.append((os.path.basename(path), f.read()))

        uploaded = upload_files_to_index(kb_id, files_payload)
        if not uploaded:
            st.error("Не удалось загрузить файлы на backend.")
            return False

        st.info(f"Загружено файлов: {len(uploaded)}. Запускаю индексирование...")
        ok, msg = build_kb_index(kb_id)

        if ok:
            st.session_state.kb_index_id = kb_id
            # Reset backend session so the next student query picks up the new index
            st.session_state.backend_session_id = None
            st.success(success_message)
            return True

        st.error(f"Ошибка индексирования: {msg}")
        return False

    except Exception as exc:
        st.error(str(exc))
        return False


def main():
    ensure_runtime_state()
    user = get_current_user()

    if user is None:
        render_login_screen()
        return

    render_sidebar(user)

    if user["role"] == "admin":
        render_admin_panel(user)
    else:
        render_student_chat(user)


main()
