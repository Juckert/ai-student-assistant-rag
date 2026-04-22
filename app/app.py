import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agent import generate_answer
from rag.ingest import ingest, load_db, save_db
from rag.retrieve import search

st.set_page_config(page_title="AI Student Assistant", layout="wide")
st.title("AI Student Assistant")

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


index, docs = load_db()

if "index" not in st.session_state:
    st.session_state.index = index

if "docs" not in st.session_state:
    st.session_state.docs = docs


mode = st.sidebar.selectbox("Mode", ["Student", "Admin"])


if mode == "Student":
    st.header("Ask a question")

    question = st.text_input("Your question")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif st.session_state.index is None:
            st.warning("Knowledge base is empty. Upload documents first.")
        else:
            st.info("Searching...")

            try:
                chunks = search(
                    question,
                    st.session_state.index,
                    st.session_state.docs,
                    k=2,
                )

                answer = generate_answer(question, chunks)

                st.success("Answer")
                st.write(answer)

                st.subheader("Sources")
                for chunk in chunks:
                    st.write(chunk)
            except Exception as exc:
                st.error(str(exc))

elif mode == "Admin":
    st.header("Upload knowledge base")

    file = st.file_uploader("Upload file", type=["pdf", "txt", "csv"])
    existing_files = get_supported_data_files()

    if existing_files:
        build_options = ["All files in data folder"] + existing_files
        selected_existing_file = st.selectbox(
            "Or choose a file from the data folder",
            build_options,
            format_func=lambda path: (
                path if path == "All files in data folder" else os.path.basename(path)
            ),
        )
    else:
        selected_existing_file = None
        st.info("No supported files found in the data folder yet.")

    if file:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, file.name)

        with open(path, "wb") as target_file:
            target_file.write(file.getbuffer())

        if st.button("Build index"):
            st.info("Processing document...")

            try:
                index, docs = ingest(path)

                save_db(index, docs)

                st.session_state.index = index
                st.session_state.docs = docs

                st.success("Knowledge base updated successfully.")
            except Exception as exc:
                st.error(str(exc))

    elif selected_existing_file and st.button("Build index from data folder"):
        st.info("Processing document...")

        try:
            source_files = (
                existing_files
                if selected_existing_file == "All files in data folder"
                else selected_existing_file
            )
            index, docs = ingest(source_files)

            save_db(index, docs)

            st.session_state.index = index
            st.session_state.docs = docs

            st.success("Knowledge base updated from the data folder.")
        except Exception as exc:
            st.error(str(exc))
