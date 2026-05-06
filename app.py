import tempfile
from pathlib import Path

import streamlit as st

from rag import ingest_docs, rag_chain, retrieve_docs, ask

st.set_page_config(
    page_title="Simple QA Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  

if "chain" not in st.session_state:
    st.session_state.chain = None

if "docs" not in st.session_state:
    st.session_state.docs = retrieve_docs()

# Build chain if docs already exist
if st.session_state.chain is None and st.session_state.docs:
    st.session_state.chain = rag_chain()

# Sidebar
with st.sidebar:
    st.header("Personal RAG Assistant")
    st.subheader("Upload Documents")

    uploaded_file = st.file_uploader(
        "Drop a file to add it to your knowledge base",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        with st.spinner(f"Uploading **{uploaded_file.name}**..."):
            # Save to temp file so LangChain loaders can read it
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                result = ingest_docs(tmp_path, source_name=uploaded_file.name)
                st.success(
                    f"**{result['file']}** — "
                    f"{result['pages']} page(s), {result['chunks']} chunks"
                )
                st.session_state.docs = retrieve_docs()
                st.session_state.chain = rag_chain()
            except Exception as e:
                st.error(f"Error: {e}")

    # List all the uploaded documents
    st.markdown("---")
    st.subheader("Your Documents")
    if st.session_state.docs:
        for f in st.session_state.docs:
            st.write(f"- {f}")
    else:
        st.caption("No documents yet. Upload one above.")

    # Clear chat
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# Main body of the UI
st.title("Your Personal Knowledge Base")
st.caption("Ask anything about your uploaded documents")

# Chat history display
if not st.session_state.chat_history:
    if st.session_state.docs:
        st.info("Your documents are ready. Ask a question below to get started.")
    else:
        st.info("Upload a PDF or TXT file in the sidebar to build your knowledge base.")
else:
    for role, content, sources in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)
            if role == "assistant" and sources:
                st.caption("Sources: " + ", ".join(sources))

# Input
question = st.chat_input("What does the document say about...?")

if question and question.strip():
    if not st.session_state.docs:
        st.warning("Upload at least one document first.")
    elif st.session_state.chain is None:
        st.error("RAG chain not initialized. Try uploading a document.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = ask(
                    st.session_state.chain,
                    question.strip(),
                    st.session_state.chat_history,
                )
                st.session_state.chat_history.append(("user", question.strip(), []))
                st.session_state.chat_history.append(
                    ("assistant", response["answer"], response["sources"])
                )
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
