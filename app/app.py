import os
import sys
from datetime import datetime

import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    # Streamlit runs this file as a script, so add the project root for local imports.
    sys.path.insert(0, ROOT_DIR)

from agent.agent import MODEL as DEFAULT_OLLAMA_MODEL
from agent.agent import NO_INFO_USER_MESSAGE, generate_answer, is_no_info_answer
from knowledge_base import append_manual_qa
from database import (
    answer_unanswered_question,
    authenticate_user,
    get_chat_messages,
    get_user_by_id,
    init_db as initialize_storage,
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
from rag.ingest import ingest, load_db, save_db
from rag.retrieve import search

st.set_page_config(page_title="ИИ-ассистент студента", layout="wide")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".csv")


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

    index, docs = load_db()

    # Session state keeps these objects across Streamlit reruns.
    if "index" not in st.session_state:
        st.session_state.index = index

    if "docs" not in st.session_state:
        st.session_state.docs = docs

    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None

    if "selected_chat_id" not in st.session_state:
        st.session_state.selected_chat_id = None


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
        st.sidebar.write(f"🤖 Модель: `{DEFAULT_OLLAMA_MODEL}`")

        if st.session_state.docs:
            st.sidebar.success(f"База знаний загружена: {len(st.session_state.docs)} фрагментов")
        else:
            st.sidebar.warning("База знаний пока не загружена")

    if st.sidebar.button("🚪 Выйти", use_container_width=True):
        st.session_state.auth_user = None
        st.session_state.selected_chat_id = None
        st.rerun()


def role_label(role):
    return "Администратор" if role == "admin" else "Студент"


def render_sources(sources):
    if not sources:
        return

    with st.expander("📚 Источники"):
        for source in sources:
            st.write(source)


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

            if st.button(
                "Понятно",
                key=f"mark_answer_seen_{item['id']}",
                use_container_width=True,
            ):
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
                # "No info" answers should not show unrelated retrieved chunks.
                if message["role"] == "assistant" and not is_no_info_answer(message["content"]):
                    render_sources(message["sources"])

    question = st.chat_input("💬 Напишите ваш вопрос")

    if not question:
        return

    if st.session_state.index is None:
        st.warning("📭 База знаний пуста. Сначала администратор должен собрать индекс.")
        return

    with st.spinner("🔎 Ищу ответ в базе знаний..."):
        try:
            chunks = search(
                question,
                st.session_state.index,
                st.session_state.docs,
                k=2,
            )
            answer = generate_answer(question, chunks)
            sources = chunks

            if is_no_info_answer(answer):
                # Save a user-friendly answer, but keep the original question for admin review.
                answer = NO_INFO_USER_MESSAGE
                sources = []

            chat_id = save_question_answer(
                user["id"],
                question,
                answer,
                sources,
                chat_id=st.session_state.selected_chat_id,
            )
            if is_no_info_answer(answer):
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
    student_count = len([user for user in users if user["role"] == "student"])
    unanswered_questions = list_unanswered_questions()

    st.markdown(
        """
        <style>
        /* Streamlit metric cards need a fixed height to stay aligned. */
        div[data-testid="stMetric"] {
            min-height: 120px;
        }

        div[data-testid="stMetricLabel"] {
            min-height: 48px;
            align-items: start;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)
    metrics = [
        ("Фрагментов в базе", len(st.session_state.docs) if st.session_state.docs else 0),
        ("Файлов в data", len(get_supported_data_files())),
        ("Всего пользователей", len(users)),
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
            build_index(path, success_message="База знаний успешно обновлена.")

    elif selected_existing_file and st.button("Построить индекс из data", use_container_width=True):
        source_files = (
            existing_files
            if selected_existing_file == "Все файлы из папки data"
            else selected_existing_file
        )
        build_index(source_files, success_message="База знаний обновлена из папки data.")

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
                            rebuilt = build_index(
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
                "ID": user["id"],
                "Логин": user["username"],
                "Имя": user["display_name"],
                "Роль": role_label(user["role"]),
                "Создан": user["created_at"],
            }
            for user in users
        ],
        use_container_width=True,
        hide_index=True,
    )


def build_index(source_files, success_message):
    st.info("Обработка документов...")

    try:
        index, docs = ingest(source_files)
        save_db(index, docs)
        st.session_state.index = index
        st.session_state.docs = docs
        st.success(success_message)
        return True
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
