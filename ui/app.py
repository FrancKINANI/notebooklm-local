"""
Streamlit interface for LocalNotebook RAG.

Interactive notebook-style UI for:
- Uploading and ingesting documents
- Asking questions with source citations
- Comparing model outputs (LFM2.5 vs Llama 3.1)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LocalNotebook",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Main theme */
    .main { background-color: #0e1117; }

    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.85em;
    }
    .source-card:hover {
        border-color: #4a9eff;
        box-shadow: 0 0 15px rgba(74, 158, 255, 0.15);
    }

    /* Metrics row */
    .metric-box {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .metric-value {
        font-size: 1.4em;
        font-weight: 700;
        color: #4a9eff;
    }
    .metric-label {
        font-size: 0.75em;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Header gradient */
    .header-gradient {
        background: linear-gradient(90deg, #4a9eff, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 800;
    }

    /* Chat messages */
    .stChatMessage { border-radius: 12px; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helper functions ─────────────────────────────────────────────────────────


def check_api_health() -> dict | None:
    """Check if the API is available."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def query_api(question: str, model: str, top_k: int) -> dict | None:
    """Send a query to the API."""
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question, "model": model, "top_k": top_k},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error: {r.status_code} — {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Is the server running?")
        return None


def ingest_api(source_path: str) -> dict | None:
    """Trigger document ingestion via the API."""
    try:
        r = requests.post(
            f"{API_URL}/ingest",
            json={"source_path": source_path},
            timeout=300,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        return None


def send_session_eval(model: str, feedbacks: list) -> bool:
    """Send session feedbacks and trigger evaluation."""
    try:
        r = requests.post(
            f"{API_URL}/session_eval",
            json={"model_key": model, "feedbacks": feedbacks},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<p class="header-gradient">📚 LocalNotebook</p>', unsafe_allow_html=True
    )
    st.caption("Privacy-first RAG • Local LLMs • No cloud")

    st.divider()

    # Model selection
    model = st.selectbox(
        "🤖 Model",
        options=["llama3.1", "lfm2.5"],
        format_func=lambda x: {
            "llama3:8b": "llama3:8b (Transformer)",
            "lfm2.5": "LFM 2.5 1.2B (Hybrid SSM)",
        }[x],
    )

    top_k = st.slider("🔍 Sources (top-k)", min_value=1, max_value=15, value=5)

    st.divider()

    # Document ingestion
    st.subheader("📄 Ingest Documents")
    source_input = st.text_input(
        "Source path",
        value="data/raw/",
        placeholder="Path to file or directory",
    )
    if st.button("🚀 Ingest", use_container_width=True):
        with st.spinner("Ingesting..."):
            result = ingest_api(source_input)
            if result and result.get("status") == "success":
                st.success(f"✅ {result.get('message', 'Done!')}")
            else:
                st.error("Ingestion failed")

    st.divider()

    # Session Evaluation
    st.subheader("🏁 Finish Session")
    st.caption("Submit your feedback and run RAGAS evaluation.")
    if st.button("End & Evaluate", use_container_width=True, type="primary"):
        feedbacks = st.session_state.get("feedbacks", [])
        if send_session_eval(model, feedbacks):
            st.success("🎯 Evaluation triggered! Check MLflow in a minute.")
            st.session_state.feedbacks = []  # Reset feedbacks
        else:
            st.error("Failed to trigger evaluation.")

    st.divider()

    # Health status
    health = check_api_health()
    if health:
        col1, col2 = st.columns(2)
        with col1:
            status = "🟢" if health.get("ollama_available") else "🔴"
            st.metric("Ollama", status)
        with col2:
            st.metric("Docs", health.get("vectorstore_count", 0))
    else:
        st.warning("⚠️ API not available")
        st.caption(f"Expected at: {API_URL}")

    st.divider()

    # Evaluation results
    st.subheader("📊 Latest Evaluation")
    metrics_path = Path("metrics/ragas_results.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                eval_metrics = json.load(f)

            st.caption(f"Model: {eval_metrics.get('model', 'N/A')}")

            cols = st.columns(2)
            with cols[0]:
                st.metric("Faithfulness", f"{eval_metrics.get('faithfulness', 0):.2f}")
                st.metric(
                    "Context Prec.", f"{eval_metrics.get('context_precision', 0):.2f}"
                )
            with cols[1]:
                st.metric("Relevancy", f"{eval_metrics.get('answer_relevancy', 0):.2f}")
                st.metric(
                    "Context Recall", f"{eval_metrics.get('context_recall', 0):.2f}"
                )

            st.caption(f"Based on {eval_metrics.get('num_samples', 0)} samples")
        except Exception:
            st.error("Could not load metrics")
    else:
        st.info("No evaluation results yet. Run ingestion to trigger background eval.")


# ── Main chat interface ─────────────────────────────────────────────────────

st.markdown('<p class="header-gradient">Ask your documents</p>', unsafe_allow_html=True)
st.caption(
    "Answers are grounded in your ingested sources — fully local, fully private."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Sources and metrics... (keep existing code)
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{src.get("filename", "unknown")}</b> '
                        f'(chunk {src.get("chunk_index", "?")})'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Feedback buttons for assistant messages
        if msg["role"] == "assistant" and "latency_ms" in msg.get("metrics", {}):
            col1, col2, _ = st.columns([0.1, 0.1, 0.8])
            with col1:
                if st.button("👍", key=f"up_{i}"):
                    # Save feedback
                    st.session_state.feedbacks.append(
                        {
                            "question": st.session_state.messages[i - 1]["content"],
                            "answer": msg["content"],
                            "is_positive": True,
                            "model_key": model,
                        }
                    )
                    st.toast("Thanks for the feedback!", icon="✅")
            with col2:
                if st.button("👎", key=f"down_{i}"):
                    st.session_state.feedbacks.append(
                        {
                            "question": st.session_state.messages[i - 1]["content"],
                            "answer": msg["content"],
                            "is_positive": False,
                            "model_key": model,
                        }
                    )
                    st.toast("Feedback recorded.", icon="📝")

        if msg.get("metrics"):
            m = msg["metrics"]
            cols = st.columns(3)
            with cols[0]:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">'
                    f'{m.get("latency_ms", 0):.0f}ms</div>'
                    f'<div class="metric-label">Latency</div></div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">'
                    f'{m.get("tokens_per_second", 0):.1f}</div>'
                    f'<div class="metric-label">Tokens/s</div></div>',
                    unsafe_allow_html=True,
                )
            with cols[2]:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">'
                    f'{m.get("num_sources", 0)}</div>'
                    f'<div class="metric-label">Sources</div></div>',
                    unsafe_allow_html=True,
                )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_api(prompt, model, top_k)

        if result:
            st.markdown(result["answer"])

            # Show sources
            if result.get("sources"):
                with st.expander(
                    f"📎 {len(result['sources'])} source(s)", expanded=False
                ):
                    for src in result["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'📄 <b>{src.get("filename", "unknown")}</b> '
                            f'(chunk {src.get("chunk_index", "?")})'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Show metrics
            cols = st.columns(3)
            with cols[0]:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">'
                    f'{result.get("latency_ms", 0):.0f}ms</div>'
                    f'<div class="metric-label">Latency</div></div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">'
                    f'{result.get("tokens_per_second", 0):.1f}</div>'
                    f'<div class="metric-label">Tokens/s</div></div>',
                    unsafe_allow_html=True,
                )
            with cols[2]:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">'
                    f'{result.get("num_sources", 0)}</div>'
                    f'<div class="metric-label">Sources</div></div>',
                    unsafe_allow_html=True,
                )

            # Save to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "metrics": {
                        "latency_ms": result.get("latency_ms", 0),
                        "tokens_per_second": result.get("tokens_per_second", 0),
                        "num_sources": result.get("num_sources", 0),
                    },
                }
            )
        else:
            error_msg = "Sorry, I couldn't process your question. Check that the API and Ollama are running."
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
