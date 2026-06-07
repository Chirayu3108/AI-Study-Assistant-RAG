import os
import tempfile
import streamlit as st
from src.rag_pipeline import RAGPipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Assistant",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 0.97rem;
        line-height: 1.7;
        color: #1a1a1a;   /* force dark text so it's readable on any Streamlit theme */
    }
    .source-card {
        background: #fafafa;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #1a1a1a;   /* same fix for source cards */
    }
    .confidence-high { color: #2e7d32; font-weight: 600; }
    .confidence-mid  { color: #f57c00; font-weight: 600; }
    .confidence-low  { color: #c62828; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Initialise the RAG pipeline once per session ──────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_pipeline():
    """Cache the pipeline so it isn't rebuilt on every interaction."""
    return RAGPipeline(persist_directory="data/vector_store")


pipeline = load_pipeline()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">AI Study Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your study material and ask anything about it.</p>', unsafe_allow_html=True)

# ── Layout: sidebar + main ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Documents")
    st.write("Supported: PDF, TXT, CSV, DOCX, XLSX, JSON")

    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv", "docx", "xlsx", "json"],
    )

    if uploaded_files:
        if st.button("Index Documents", use_container_width=True):
            progress = st.progress(0)
            total_chunks = 0

            for i, uploaded_file in enumerate(uploaded_files):
                # write the uploaded file to a temp location so our loaders can read it
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunks = pipeline.ingest_file(tmp_path)
                    total_chunks += chunks

                os.unlink(tmp_path)  # clean up temp file
                progress.progress((i + 1) / len(uploaded_files))

            st.success(f"Indexed {len(uploaded_files)} file(s) → {total_chunks} chunks stored.")

    st.divider()

    # show how many chunks are stored right now
    doc_count = pipeline.get_document_count()
    st.metric("Chunks in Knowledge Base", doc_count)

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq LLaMA · Streamlit")


# ── Main area: Q&A ────────────────────────────────────────────────────────────
st.subheader("Ask a Question")

# keep chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask something about your study material...")

if query:
    if pipeline.get_document_count() == 0:
        st.warning("No documents indexed yet. Upload some files from the sidebar first.")
    else:
        with st.spinner("Thinking..."):
            result = pipeline.ask(query, top_k=5)

        # save to history
        st.session_state.chat_history.append({"query": query, "result": result})

# render chat history (latest first)
for item in reversed(st.session_state.chat_history):
    q = item["query"]
    r = item["result"]

    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer-box">{r["answer"]}</div>', unsafe_allow_html=True)

        # confidence badge
        conf = r["confidence"]
        if conf >= 0.4:
            badge = f'<span class="confidence-high">High confidence ({conf})</span>'
        elif conf >= 0.2:
            badge = f'<span class="confidence-mid">Medium confidence ({conf})</span>'
        else:
            badge = f'<span class="confidence-low">Low confidence ({conf}) — answer may not be reliable</span>'
        st.markdown(badge, unsafe_allow_html=True)

        # collapsible sources section
        if r["sources"]:
            with st.expander("Sources"):
                for src in r["sources"]:
                    st.markdown(f"""
<div class="source-card">
    <strong> {src['file']}</strong> &nbsp;·&nbsp; Page: {src['page']} &nbsp;·&nbsp; Score: {src['score']}<br>
    <em>{src['preview']}</em>
</div>
""", unsafe_allow_html=True)
