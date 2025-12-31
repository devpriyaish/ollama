"""
Production-Ready Streamlit Frontend for LangGraph RAG Chatbot
----------------------------------------------------------------
"""

# ==========================
# Standard Library Imports
# ==========================
import uuid

# ==========================
# Third-Party Imports
# ==========================
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ==========================
# Local Application Imports
# ==========================
from backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ==========================
# Utility Helpers
# ==========================
def generate_thread_id():
    return uuid.uuid4()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Create a new thread + clear chat history."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["thread_titles"][str(thread_id)] = "New Chat"
    add_thread(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ==========================
# Session Initialization
# ==========================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# NEW: store titles for each thread
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ==========================
# Sidebar UI
# ==========================
st.sidebar.title("Multi-Utility Chatbot")

# ---- Display current conversation title instead of thread_id ----
current_title = st.session_state["thread_titles"].get(thread_key, "New Chat")
st.sidebar.header(current_title)

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    last_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{last_doc['filename']}` "
        f"({last_doc['chunks']} chunks | {last_doc['documents']} pages)"
    )
else:
    st.sidebar.info("No PDF uploaded yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name not in thread_docs:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete")

# ---- Show list of past conversations using titles ----
st.sidebar.subheader("Past Conversations")
if threads:
    for t_id in threads:
        tid_str = str(t_id)
        title = st.session_state["thread_titles"].get(tid_str, "Chat")

        if st.sidebar.button(title, key=f"thread-{tid_str}"):
            selected_thread = t_id
else:
    st.sidebar.write("No previous chats available.")

# ==========================
# Main Chat Area
# ==========================
st.title("Voyage Estimation AI Agent")

# ---- Render Chat History ----
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])

# ---- User Input ----
user_input = st.chat_input("Start estimating your voyage…")

if user_input:

    # --- If this is the first message of this thread → set the title ---
    if st.session_state["thread_titles"].get(thread_key) in [None, "New Chat"]:
        st.session_state["thread_titles"][thread_key] = user_input[:40]

    # Save & render user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_stream():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_message = st.write_stream(ai_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool completed", state="complete", expanded=False
            )

    # store AI message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # PDF metadata under chat window
    meta = thread_document_metadata(thread_key)
    if meta:
        st.caption(
            f"Indexed PDF: {meta.get('filename')} "
            f"(Chunks: {meta.get('chunks')} | Pages: {meta.get('documents')})"
        )

st.divider()

# ==========================
# Thread Restoration
# ==========================
if selected_thread:
    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(selected_thread)
    restored = []

    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        restored.append({"role": role, "content": m.content})

    st.session_state["message_history"] = restored
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
