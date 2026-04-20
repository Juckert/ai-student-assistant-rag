# streamlit run app/app.py
import sys
import os
import streamlit as st

# PATH FIX
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.ingest import ingest, save_db, load_db
from rag.retrieve import search
from agent.agent import generate_answer


# STREAMLIT CACHE MODEL
@st.cache_resource
def load_model_cached():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/multilingual-e5-base")


st.set_page_config(page_title="AI Student Assistant", layout="wide")
st.title("🎓 AI Student Assistant")


# LOAD DB
index, docs = load_db()

if "index" not in st.session_state:
    st.session_state.index = index

if "docs" not in st.session_state:
    st.session_state.docs = docs


# MODE
mode = st.sidebar.selectbox("Mode", ["👨‍🎓 Student", "👨‍🏫 Admin"])


# STUDENT MODE
if mode == "👨‍🎓 Student":

    st.header("Ask a question")

    question = st.text_input("Your question")

    if st.button("Ask"):

        if st.session_state.index is None:
            st.warning("Knowledge base is empty. Upload documents first.")
        else:
            st.info("Searching...")

            chunks = search(
                question,
                st.session_state.index,
                st.session_state.docs,
                k=2
            )

            answer = generate_answer(question, chunks)

            st.success("Answer:")
            st.write(answer)

            st.subheader("📚 Sources")
            for c in chunks:
                st.write(c)


# ADMIN MODE
elif mode == "👨‍🏫 Admin":

    st.header("Upload knowledge base")

    file = st.file_uploader("Upload file", type=["pdf", "txt"])

    if file:

        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", file.name)

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        if st.button("Build index"):

            st.info("Processing document...")

            index, docs = ingest(path)

            save_db(index, docs)

            st.session_state.index = index
            st.session_state.docs = docs

            st.success("Knowledge base updated successfully!")