import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import time
import pickle

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SOURCE", page_icon="📚", layout="wide")

# ── Persistent memory paths ───────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
FILES_CACHE_PATH = os.path.join(DATA_DIR, "uploaded_files.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Custom CSS — ChatGPT × Adobe Firefly aesthetic ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background with subtle Firefly gradient */
.stApp {
    background: linear-gradient(135deg, #0d0d0f 0%, #12101a 50%, #0d0f1a 100%);
    color: #ececec;
}

/* ── Main title ── */
h1 {
    background: linear-gradient(90deg, #a259ff, #ff6bcb, #ff9f43);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600;
    letter-spacing: -0.5px;
}

/* ── Divider ── */
hr {
    border-color: #2a2a3a !important;
}

/* ── Dialog / modal backdrop (used by upload/analyzing dialogs) ── */
div[data-testid="stDialog"] {
    backdrop-filter: blur(7px) !important;
    background: rgba(6, 8, 16, 0.55) !important;
}

/* ── Dialog / modal (chat dialog only) ── */
div[data-testid="stDialog"]:has([data-testid="stChatInput"]) > div {
    background: #1a1a2e !important;
    border: 1px solid #2e2e4a !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 40px rgba(162, 89, 255, 0.15) !important;
    color: #ececec !important;
}

div[data-testid="stDialog"]:has([data-testid="stChatInput"]) h1,
div[data-testid="stDialog"]:has([data-testid="stChatInput"]) h2,
div[data-testid="stDialog"]:has([data-testid="stChatInput"]) h3,
div[data-testid="stDialog"]:has([data-testid="stChatInput"]) p,
div[data-testid="stDialog"]:has([data-testid="stChatInput"]) label,
div[data-testid="stDialog"]:has([data-testid="stChatInput"]) span {
    color: #ececec !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #a259ff, #ff6bcb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 2px 16px rgba(162, 89, 255, 0.3) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: #2a2a3a !important;
    color: #666 !important;
    box-shadow: none !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background: #1e1e2e !important;
    border: 1px solid #3a3a5a !important;
    border-radius: 10px !important;
    color: #ececec !important;
}
.stTextInput > div > div > input:focus {
    border-color: #a259ff !important;
    box-shadow: 0 0 0 2px rgba(162, 89, 255, 0.2) !important;
}

/* ── Sliders ── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #a259ff, #ff6bcb) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1a1a2e !important;
    border: 2px dashed #3a3a5a !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #a259ff !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #a259ff, #ff6bcb, #ff9f43) !important;
    border-radius: 99px !important;
}

/* ── Info / success / error alerts ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}
[data-baseweb="notification"][kind="positive"] {
    background: rgba(76, 175, 80, 0.12) !important;
    border-left: 3px solid #4caf50 !important;
}
[data-baseweb="notification"][kind="negative"] {
    background: rgba(244, 67, 54, 0.12) !important;
    border-left: 3px solid #f44336 !important;
}
[data-baseweb="notification"][kind="info"] {
    background: rgba(162, 89, 255, 0.1) !important;
    border-left: 3px solid #a259ff !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #16162a !important;
    border-radius: 14px !important;
    border: 1px solid #252540 !important;
    margin-bottom: 6px !important;
    padding: 12px 16px !important;
}
/* User bubble accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #1e1040, #1a1a3a) !important;
    border-color: #a259ff44 !important;
}

/* Keep dialog sizing natural; only style chat visuals below. */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 4px 0 !important;
    margin-bottom: 4px !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    padding-left: 0 !important;
    margin-left: 0 !important;
}
[data-testid="stChatInput"] > div {
    padding-left: 0 !important;
    margin-left: 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #1a1a2e !important;
    border: 1px solid #3a3a5a !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #a259ff !important;
    box-shadow: 0 0 0 2px rgba(162, 89, 255, 0.2) !important;
}

/* Make chat input box more rounded */
[data-testid="stChatInput"] > div {
    border-radius: 32px !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
}

/* ── Full-page chat layout (Step 3) ── */
.chat-shell {
    max-width: 900px;
    margin: 0 auto;
}
.chat-top {
    max-width: 900px;
    margin: 0 auto 10px auto;
}
[data-testid="stChatMessage"] {
    max-width: 900px;
    margin-left: auto !important;
    margin-right: auto !important;
}
[data-testid="stChatInput"] {
    position: fixed !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    bottom: 18px !important;
    width: min(900px, calc(100vw - 34px)) !important;
    z-index: 1000 !important;
}
[data-testid="stChatInput"] > div {
    background: rgba(34, 34, 42, 0.92) !important;
    border: 1px solid #34364a !important;
    border-radius: 28px !important;
    backdrop-filter: blur(6px);
    padding: 8px 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border: none !important;
    box-shadow: none !important;
}
.main .block-container {
    padding-bottom: 132px !important;
}

.composer-file-label {
    position: fixed;
    left: 50%;
    transform: translateX(-50%);
    bottom: 104px;
    width: min(900px, calc(100vw - 34px));
    color: #a9acc4;
    font-size: 0.84rem;
    z-index: 999;
    pointer-events: none;
}

.sample-hint-wrap {
    min-height: 46vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ── Home (step 1) layout ── */
.home-source-title {
    text-align: center;
    font-size: clamp(5rem, 15vw, 9.5rem) !important;
    margin: 0.35rem 0 0.25rem 0;
    line-height: 1.03;
    font-weight: 800;
    letter-spacing: 0.02em;
    background: linear-gradient(90deg, #a259ff, #ff6bcb, #ff9f43);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.home-powered {
    text-align: center;
    color: #9b9bb4;
    margin: 0 0 0.8rem 0;
    font-size: 0.98rem;
}
.home-divider {
    border-color: #2a2a3a !important;
    margin: 0.35rem 0 0.6rem 0 !important;
}

# ── Auto-load previous saved knowledge base ───────────────────────────────────
if st.session_state.vectorstore is None and os.path.exists(FAISS_INDEX_PATH):
    try:
        embeddings = OllamaEmbeddings(model=st.session_state.embedding_model)
        st.session_state.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        if os.path.exists(FILES_CACHE_PATH):
            with open(FILES_CACHE_PATH, "rb") as f:
                st.session_state.uploaded_files_data = pickle.load(f)

        st.session_state.step = 3
    except Exception:
        pass       


.home-welcome {
    text-align: center;
    color: #888;
    padding: 0.55rem 0 0.2rem 0;
}

def build_vectorstore():
    all_docs = []
    for name, data in st.session_state.uploaded_files_data:
        all_docs.extend(load_file_bytes(name, data))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model=st.session_state.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save FAISS permanently
    vectorstore.save_local(FAISS_INDEX_PATH)
    st.success(f"Saved FAISS to: {FAISS_INDEX_PATH}")

    # Save uploaded file metadata permanently
    with open(FILES_CACHE_PATH, "wb") as f:
        pickle.dump(st.session_state.uploaded_files_data, f)
            
    st.success(f"Saved uploaded file cache to: {FILES_CACHE_PATH}")

    st.session_state.vectorstore = vectorstore
    st.session_state.chat_history = []
            
.home-powered-corner {
    position: fixed;
    left: 14px;
    bottom: 16px;
    color: #9b9bb4;
    font-size: 0.92rem;
    z-index: 40;
    letter-spacing: 0.01em;
}

#sample-hint-text {
    color: #9ea3bf;
    font-size: 1.04rem;
    text-align: center;
    opacity: 0.9;
    max-width: 760px;
    line-height: 1.5;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #1a1a2e !important;
    border-radius: 10px !important;
    color: #a0a0c0 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #a259ff !important;
}

/* ── Caption / small text ── */
.stCaption, small {
    color: #8080a0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #12101a; }
::-webkit-scrollbar-thumb { background: #3a3a5a; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #a259ff; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "step": 1,           # 1 = upload, 2 = analysing, 3 = chat
    "vectorstore": None,
    "chat_history": [],
    "model_name": "llama3.1",
    "embedding_model": "nomic-embed-text",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "top_k": 4,
    "error": None,
    "uploaded_files_data": [],   # list of (name, bytes) — serialisable across reruns
    "show_disclaimer": False,
    "show_upload": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper ────────────────────────────────────────────────────────────────────
def load_file_bytes(name: str, data: bytes) -> list:
    suffix = os.path.splitext(name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix in (".docx", ".doc"):
        loader = Docx2txtLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")
    docs = loader.load()
    os.unlink(tmp_path)
    return docs


def build_vectorstore():
    all_docs = []
    for name, data in st.session_state.uploaded_files_data:
        all_docs.extend(load_file_bytes(name, data))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    )
    chunks = splitter.split_documents(all_docs)
    embeddings = OllamaEmbeddings(model=st.session_state.embedding_model)
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state.chat_history = []


# Remove CSS for .left-glitch-rail, .right-glitch-rail, .left-glitch-title, .right-glitch-title, .left-glitch-track, .right-glitch-track, .left-glitch-line, .right-glitch-line, @keyframes leftRailScroll, @keyframes rightRailScroll, @keyframes glitchFlicker, @keyframes glitchFadeOut

# ═══════════════════════════════════════════════════════════════════════════════
# DIALOG 0 — Experimental disclaimer
# ═══════════════════════════════════════════════════════════════════════════════
@st.dialog("Experimental Notice", width="small", dismissible=False)
def dialog_experimental_notice():
    st.markdown("This is an experimental project. Made for the Live session on Saturday (25-04-2026), similar project as I made for the assignment. Tweeked a bit and added a bit more features.")
    if st.button("Confirm", type="primary", use_container_width=True):
        st.session_state.show_disclaimer = False
        st.session_state.show_upload = True
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# DIALOG 1 — Upload files
# ═══════════════════════════════════════════════════════════════════════════════
@st.dialog("📂 Upload your documents", width="large", dismissible=False)
def dialog_upload():
    st.markdown("Upload one or more files to build your knowledge base.")

    # Settings inside the dialog
    with st.expander("⚙️ Preferrences", expanded=False):
        st.session_state.model_name = st.text_input(
            "Ollama model", value=st.session_state.model_name,
            help="e.g. llama3, mistral, phi3"
        )
        st.session_state.embedding_model = st.text_input(
            "Embedding model", value=st.session_state.embedding_model,
            help="Pull with: ollama pull nomic-embed-text"
        )
        st.session_state.chunk_size = st.slider("Chunk size", 256, 2048, st.session_state.chunk_size, 64)
        st.session_state.chunk_overlap = st.slider("Chunk overlap", 0, 512, st.session_state.chunk_overlap, 16)
        st.session_state.top_k = st.slider("Top-K chunks", 1, 10, st.session_state.top_k)

    uploaded = st.file_uploader(
        "Choose files", type=["pdf", "txt", "docx", "doc"], accept_multiple_files=True
    )

    if uploaded:
        st.success(f"{len(uploaded)} file(s) selected: {', '.join(f.name for f in uploaded)}")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Analyse ➜", type="primary", disabled=not uploaded):
            st.session_state.uploaded_files_data = [(f.name, f.read()) for f in uploaded]
            st.session_state.step = 2
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# DIALOG 2 — Analysing
# ═══════════════════════════════════════════════════════════════════════════════
@st.dialog("🔍 Analysing documents…", width="large", dismissible=False)
def dialog_analysing():
    st.markdown("Please remain patient, while your documents are being analysed.")
    names = [n for n, _ in st.session_state.uploaded_files_data]
    st.info(f"Processing: {', '.join(names)}")

    progress = st.progress(0, text="Loading documents…")
    status = st.empty()

    try:
        progress.progress(20, text="Loading vectors…")
        time.sleep(0.3)

        progress.progress(50, text="Splitting into chunks…")
        status.caption("Chunking text…")
        time.sleep(0.2)

        progress.progress(70, text="Creating embeddings (may take a moment)…")
        status.caption("Calling Ollama's embeddings…")

        build_vectorstore()

        progress.progress(100, text="Done!")
        status.success("✅ Knowledge base built successfully!")
        time.sleep(0.8)

        st.session_state.step = 3
        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(
            f"**Error:** {e}\n\n"
            f"Make sure Ollama is running (`ollama serve`) and the embedding model "
            f"**{st.session_state.embedding_model}** is pulled "
            f"(`ollama pull {st.session_state.embedding_model}`)."
        )
        if st.button("← Go back"):
            st.session_state.step = 1
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Full-page Chat
# ═══════════════════════════════════════════════════════════════════════════════
def render_chat_page():
    names = [n for n, _ in st.session_state.uploaded_files_data]
    # primary_name = names[0] if names else "your file"

    # File name above composer (left side)
    # st.markdown(
    #     f"<div class='composer-file-label'>📄 {primary_name}</div>",
    #     unsafe_allow_html=True,
    # )

    # ── Message history ───────────────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if entry.get("sources"):
                with st.expander("📄 Source chunks"):
                    for chunk in entry["sources"]:
                        st.markdown(f"**{chunk['label']}**")
                        st.text(chunk["text"])

    # anchor for auto-scroll
    st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)

    if len(st.session_state.chat_history) == 0:
        sample_questions = [
            "What is the main idea?",
            "Summarize in 5 bullet points.",
            "What are the key terms or concepts?",
            "Give me the most important section.",
            "What questions can I ask to better understand this?",
            "List facts and figures found in this file.",
            "Explain this file in simple language.",
            "Brief the document I provided, all bullet points + key details from the document I provided.",
        ]
        js_hints = "[" + ",".join([f'\"{q}\"' for q in sample_questions]) + "]"
        components.html(
            f"""
            <div class="sample-hint-wrap">
                <div id="sample-hint-text"></div>
            </div>
            <style>
                .sample-hint-wrap {{
                    min-height: 46vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                #sample-hint-text {{
                    color: #9ea3bf;
                    font-family: 'Light Deltha', 'Segoe UI Light', 'Segoe UI', sans-serif;
                    font-size: clamp(1.9rem, 4vw, 3rem);
                    font-weight: 300;
                    letter-spacing: 0.02em;
                    text-align: center;
                    opacity: 0;
                    max-width: 760px;
                    line-height: 1.5;
                    transition: opacity 0.45s ease-in-out;
                    padding: 0 10px;
                }}
                #sample-hint-text.visible {{
                    opacity: 0.95;
                }}
            </style>
            <script>
                (function() {{
                    const hints = {js_hints};
                    const target = document.getElementById('sample-hint-text');
                    if (!target || !hints || !hints.length) return;

                    let previous = -1;

                    function randomIndex() {{
                        if (hints.length === 1) return 0;
                        let idx = Math.floor(Math.random() * hints.length);
                        while (idx === previous) {{
                            idx = Math.floor(Math.random() * hints.length);
                        }}
                        previous = idx;
                        return idx;
                    }}

                    function showNextHint() {{
                        target.classList.remove('visible');
                        setTimeout(() => {{
                            const i = randomIndex();
                            target.textContent = 'Try asking: ' + hints[i];
                            target.classList.add('visible');
                        }}, 450);
                    }}

                    showNextHint();
                    const intervalId = setInterval(showNextHint, 4000);

                    // Hide and stop hints immediately when user starts typing
                    const parentInput = window.parent?.document?.querySelector('[data-testid="stChatInput"] textarea');
                    const hideHints = () => {{
                        clearInterval(intervalId);
                        const wrap = document.querySelector('.sample-hint-wrap');
                        if (wrap) wrap.style.display = 'none';
                    }};

                    if (parentInput) {{
                        parentInput.addEventListener('input', hideHints, {{ once: true }});
                    }}
                }})();
            </script>
            """,
            height=320,
            scrolling=False,
        )

    question = st.chat_input("Ask...")

    # Remove the middle gray box, keep only outer and textarea styling
    st.markdown("""
        <script>
        // Remove hint wrapper in the main document as a fallback
        const chatInput = document.querySelector('[data-testid="stChatInput"] textarea');
        if (chatInput) {
            chatInput.addEventListener('input', function() {
                const hintWrap = document.querySelector('.sample-hint-wrap');
                if (hintWrap) {
                    hintWrap.remove();
                }
            });
        }
        </script>
        <style>
        [data-testid="stChatInput"] textarea {
            border-radius: 18px !important;
            background: #23232e !important;
            border: 1.5px solid #34364a !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if question:
        # Also fade out on submit (for robustness)
        st.markdown(
            """
            <script>
            (function() {
                const leftRail = document.querySelector('.left-glitch-rail');
                const rightRail = document.querySelector('.right-glitch-rail');
                if (leftRail) leftRail.classList.add('fade-out');
                if (rightRail) rightRail.classList.add('fade-out');
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating…"):
                try:
                    prompt_template = PromptTemplate(
                        input_variables=["context", "question"],
                        template=(
                            "You are a helpful assistant. Use ONLY the context below to answer "
                            "the question. If the answer is not in the context, say so.\n\n"
                            "Context:\n{context}\n\n"
                            "Question: {question}\n\n"
                            "Answer:"
                        ),
                    )
                    llm = OllamaLLM(model=st.session_state.model_name, temperature=0)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": st.session_state.top_k}
                        ),
                        chain_type_kwargs={"prompt": prompt_template},
                        return_source_documents=True,
                    )
                    result = qa_chain.invoke({"query": question})
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])

                    st.write(answer)

                    sources = []
                    with st.expander("📄 Source chunks"):
                        for i, doc in enumerate(source_docs, 1):
                            src = doc.metadata.get("source", "unknown")
                            page = doc.metadata.get("page", "")
                            label = f"Chunk {i} — {os.path.basename(src)}"
                            if page != "":
                                label += f", page {int(page) + 1}"
                            st.markdown(f"**{label}**")
                            st.text(doc.page_content[:500])
                            sources.append({"label": label, "text": doc.page_content[:500]})

                    st.session_state.chat_history.append(
                        {"question": question, "answer": answer, "sources": sources}
                    )

                except Exception as e:
                    st.error(
                        f"LLM error: {e}\n\n"
                        f"Make sure Ollama is running and model **{st.session_state.model_name}** is pulled.\n"
                        f"`ollama pull {st.session_state.model_name}`"
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE — backdrop + dialog router
# ═══════════════════════════════════════════════════════════════════════════════
step = st.session_state.step

if step == 1:
    _, hero_col, _ = st.columns([1, 3.2, 1])
    with hero_col:
        st.markdown(
            """
            <h1 class='home-source-title' style='font-size: clamp(5rem, 15vw, 9.5rem); font-weight: 800; text-align:center; margin: 0.35rem 0 0.25rem 0; position: relative; left: 42px;'>
                SOURCE
            </h1>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='home-divider' />", unsafe_allow_html=True)
elif step == 2:
    st.title("SOURCE")
    st.caption("Powered by Ollama · LangChain · FAISS")
    st.markdown("---")

if step == 1:
    st.markdown(
        """
        <style>
        html, body, #root {
            overflow: hidden !important;
            height: 100dvh !important;
            max-height: 100dvh !important;
            overscroll-behavior: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .stApp {
            height: 100dvh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
        }
        [data-testid="stAppViewContainer"] {
            overflow: hidden !important;
            height: 100dvh !important;
            max-height: 100dvh !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            overflow: hidden !important;
            height: 100dvh !important;
            max-height: 100dvh !important;
        }
        section.main {
            overflow: hidden !important;
            height: 100dvh !important;
            max-height: 100dvh !important;
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .main .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            overflow: hidden !important;
            height: 100dvh !important;
            max-height: 100dvh !important;
            box-sizing: border-box !important;
        }
        * {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
        section.main::-webkit-scrollbar,
        [data-testid="stAppViewContainer"]::-webkit-scrollbar,
        html::-webkit-scrollbar,
        body::-webkit-scrollbar,
        *::-webkit-scrollbar {
            width: 0 !important;
            height: 0 !important;
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        (function () {
            const rootWin = window.parent || window;
            if (rootWin.__sourceHomeScrollLocked) return;
            rootWin.__sourceHomeScrollLocked = true;

            const prevent = (e) => {
                e.preventDefault();
                e.stopPropagation();
                return false;
            };

            try {
                rootWin.scrollTo(0, 0);
                if (rootWin.document && rootWin.document.body) {
                    rootWin.document.body.style.setProperty("overflow", "hidden", "important");
                    rootWin.document.body.style.setProperty("height", "100vh", "important");
                    rootWin.document.body.style.setProperty("max-height", "100vh", "important");
                }
                if (rootWin.document && rootWin.document.documentElement) {
                    rootWin.document.documentElement.style.setProperty("overflow", "hidden", "important");
                    rootWin.document.documentElement.style.setProperty("height", "100vh", "important");
                    rootWin.document.documentElement.style.setProperty("max-height", "100vh", "important");
                }

                const containers = rootWin.document.querySelectorAll(
                    "section.main, [data-testid='stAppViewContainer'], .stApp"
                );
                containers.forEach((el) => {
                    el.style.setProperty("overflow", "hidden", "important");
                    el.style.setProperty("height", "100vh", "important");
                    el.style.setProperty("max-height", "100vh", "important");
                    el.style.setProperty("scrollbar-width", "none", "important");
                    el.scrollTop = 0;
                });

                const parentStyle = rootWin.document.createElement("style");
                parentStyle.textContent = "::-webkit-scrollbar{width:0 !important;height:0 !important;display:none !important;}";
                rootWin.document.head.appendChild(parentStyle);
            } catch (err) {
                // ignore cross-frame access issues
            }

            rootWin.addEventListener("wheel", prevent, { passive: false });
            rootWin.addEventListener("touchmove", prevent, { passive: false });
            rootWin.addEventListener("scroll", () => {
                try {
                    rootWin.scrollTo(0, 0);
                    const containers = rootWin.document.querySelectorAll(
                        "section.main, [data-testid='stAppViewContainer'], .stApp"
                    );
                    containers.forEach((el) => {
                        el.scrollTop = 0;
                    });
                } catch (err) {
                    // ignore cross-frame access issues
                }
            }, { passive: false });
            rootWin.addEventListener("keydown", (e) => {
                const blocked = ["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "];
                if (blocked.includes(e.key)) {
                    prevent(e);
                }
            }, { passive: false });
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )
    _, hero_col, _ = st.columns([5.25, 4, 4])
    with hero_col:
       
        st.markdown("<div class='home-powered-corner'>Powered by Ollama · LangChain · FAISS ·</div>", unsafe_allow_html=True)
        _, center_col, _ = st.columns([0.25, 7, 5])
        with center_col:
            clicked_get_started = st.button("Get Started", type="primary", use_container_width=True)

    if clicked_get_started:
        st.session_state.show_disclaimer = True
        st.rerun()

    if st.session_state.show_disclaimer:
        dialog_experimental_notice()
    elif st.session_state.show_upload:
        dialog_upload()

elif step == 2:
    st.markdown(
        "<div style='text-align:center; padding: 3rem 0; color: #888;'>"
        "<h3>🔍 Analysing your documents…</h3>"
        "</div>",
        unsafe_allow_html=True,
    )
    dialog_analysing()

elif step == 3:
    render_chat_page()

