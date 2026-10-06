"""Chat UI. Talks to the AskDocs API over HTTP. Started by `python run.py`.

How Streamlit works: the whole script re-runs from top to bottom on every interaction
(a click, an upload, a new chat message). Anything that must survive between re-runs,
like the chat history, is kept in `st.session_state`.

The UI holds no models or data itself; it only calls the API. That keeps the UI light and
means the same API could serve a different frontend (a web app, a Slack bot, ...).
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st

# Where the API lives. run.py and docker-compose set this; the default matches a local run.
# 127.0.0.1 rather than localhost: on Windows, localhost tries IPv6 first and stalls ~2 s per request.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# A fictional handbook that ships with the repo, so the app can be tried in one click.
SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "lumen_employee_handbook.md"
SAMPLE_QUESTIONS = [
    "How many days a week can I work remotely?",
    "What should I do if my laptop is stolen?",
    "Do I need approval for a 700 EUR business trip?",
    "How much parental leave do employees get?",
]

# Retrieval modes: API value → label shown to the user.
MODES = {
    "hybrid_rerank": "Hybrid + reranking (recommended)",
    "hybrid": "Hybrid (semantic + keyword)",
    "dense": "Semantic only",
    "bm25": "Keyword only",
}

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="AskDocs", page_icon="📄", layout="wide")


# --- API helpers -----------------------------------------------------------------------------------
def api(method: str, path: str, **kwargs):
    """Call the API and return its JSON, or None after showing a friendly error."""
    try:
        # Long timeout: indexing a big PDF can take a while.
        r = requests.request(method, f"{API_URL}{path}", timeout=600, **kwargs)
    except requests.ConnectionError:
        st.title("📄 AskDocs")
        st.warning(f"The AskDocs API isn't reachable at {API_URL}. Start the app with `python run.py`.")
        st.stop()  # nothing below can work without the API
    if not r.ok:
        # FastAPI puts error messages in {"detail": "..."}; fall back to the raw body otherwise.
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(detail)
        return None
    return r.json()


def upload(files: list[tuple[str, bytes]]) -> None:
    """Send files to the API for indexing and report the result of each."""
    with st.spinner("Reading, chunking and indexing..."):
        payload = [("files", (name, data, "application/octet-stream")) for name, data in files]
        results = api("POST", "/documents", files=payload)
    for item in results or []:
        if item["warning"]:
            st.toast(f"{item['source']}: {item['warning']}", icon="⚠️")
        else:
            st.toast(f"Indexed {item['source']} ({item['chunks']} chunks)", icon="✅")


def iter_events(response: requests.Response):
    """Split a streamed NDJSON response into events as soon as each line arrives."""
    buffer = b""
    for piece in response.iter_content(chunk_size=None):  # None = hand over data as it arrives
        buffer += piece
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line.strip():
                yield json.loads(line)


# --- Rendering helpers --------------------------------------------------------------------------
def escape_markdown(text: str) -> str:
    """Show source text literally: a chunk starting with '# Title' must not render as a giant heading."""
    return re.sub(r"([\\`*_#\[\]<>|~])", r"\\\1", text)


def render_sources(sources: list[dict], cited: list[int]) -> None:
    """Collapsible list of the passages the answer was based on, with a relevance bar for each."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} retrieved, {len(cited)} cited)"):
        for i, s in enumerate(sources, 1):
            location = f"{s['source']}, page {s['page']}" if s["page"] else s["source"]
            st.markdown(f"**[{i}] {location}**" + ("  ✅ cited" if i in cited else ""))
            st.progress(min(max(s["relevance"], 0.0), 1.0), text=f"relevance {s['relevance']:.0%}")
            preview = s["text"] if len(s["text"]) <= 700 else s["text"][:700] + "..."
            st.caption(escape_markdown(preview))


def render_timing(timings: dict, mode: str) -> None:
    if timings:
        st.caption(
            f"Retrieval {timings['retrieval']:.0f} ms · generation {timings['generation'] / 1000:.1f} s · {MODES[mode]}"
        )


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []), msg.get("cited", []))
            render_timing(msg.get("timings", {}), msg.get("mode", "hybrid_rerank"))


def answer_question(question: str, mode: str) -> dict | None:
    """Stream an answer into the chat. Returns the finished message for the history."""
    with st.chat_message("assistant"):
        body = st.empty()
        body.caption("Searching your documents...")
        text, sources, done = "", [], {}
        try:
            with requests.post(
                f"{API_URL}/ask/stream", json={"question": question, "mode": mode}, stream=True, timeout=600
            ) as r:
                r.raise_for_status()
                for event in iter_events(r):
                    if event["type"] == "sources":
                        sources = event["sources"]
                        body.caption(f"Found {len(sources)} relevant passages, writing the answer...")
                    elif event["type"] == "token":
                        text += event["text"]
                        body.markdown(text + " ▌")  # the cursor shows the answer is still being written
                    elif event["type"] == "done":
                        done = event
                    elif event["type"] == "error":
                        body.error(event["message"])
                        return None
        except requests.RequestException as e:
            body.error(f"Something went wrong while answering: {e}")
            return None
        body.markdown(text)
        cited = done.get("cited", [])
        render_sources(sources, cited)
        render_timing(done.get("timings_ms", {}), mode)
    return {"role": "assistant", "content": text, "sources": sources, "cited": cited,
            "timings": done.get("timings_ms", {}), "mode": mode}


# --- Page --------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

info = api("GET", "/info") or {}
docs: dict[str, int] = info.get("documents", {})

with st.sidebar:
    st.header("📄 AskDocs")

    st.subheader("Documents")
    # A form with clear_on_submit empties the picker after indexing, so files aren't re-sent by accident.
    with st.form("upload", clear_on_submit=True, border=False):
        picked = st.file_uploader(
            "Add PDF, TXT or Markdown files", type=["pdf", "txt", "md"], accept_multiple_files=True
        )
        if st.form_submit_button("Index documents", type="primary", width="stretch") and picked:
            upload([(f.name, f.getvalue()) for f in picked])
            st.rerun()

    if not docs and SAMPLE_DOC.exists():
        if st.button("Load sample document", width="stretch", help="A fictional employee handbook"):
            upload([(SAMPLE_DOC.name, SAMPLE_DOC.read_bytes())])
            st.rerun()

    for name, n_chunks in docs.items():
        col1, col2 = st.columns([6, 1], vertical_alignment="center")
        col1.markdown(f"**{name}**  \n{n_chunks} chunks")
        # Each button needs a unique key; quote() makes the file name safe inside a URL.
        if col2.button("✕", key=f"del-{name}", help=f"Remove {name}"):
            api("DELETE", f"/documents/{quote(name, safe='')}")
            st.rerun()

    st.divider()
    st.subheader("Settings")
    mode = st.selectbox("Retrieval mode", list(MODES), format_func=MODES.get)
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    online = info.get("llm_online", False)
    st.caption(
        f"{'🟢' if online else '🔴'} LLM: {info.get('llm_model', '?')}  \n"
        f"Embeddings: {info.get('embed_model', '?')}  \n"
        f"Reranker: {info.get('rerank_model') or 'off'}  \n"
        f"Device: {info.get('device', '?')}  \n"
        f"Index: {len(docs)} documents, {sum(docs.values())} chunks"
    )
    if info and not online:
        st.warning("The language model is offline. Start Ollama (or check `.env`) to get answers.")

st.title("📄 AskDocs")
st.caption("Ask questions about your documents. Answers are grounded in the indexed files and cite their sources.")

# Before the first question: point to the next step, and offer example questions for the sample document.
pending = st.session_state.pop("pending_question", None)
if not st.session_state.messages and not pending:
    if not docs:
        st.info("Add a document in the sidebar, or load the sample document to try it out.")
    elif SAMPLE_DOC.name in docs:
        st.write("Example questions:")
        cols = st.columns(2)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"sample-{i}", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()

# Redraw the conversation (the script re-runs on every interaction).
for msg in st.session_state.messages:
    render_message(msg)

question = st.chat_input("Ask a question about your documents", disabled=not docs) or pending
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])
    reply = answer_question(question, mode)
    if reply:
        st.session_state.messages.append(reply)
