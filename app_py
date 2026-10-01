"""
MF-Guide: Streamlit Chatbot UI

Phase 5 — Chat interface with source citations.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from functools import lru_cache

import streamlit as st

from config import Config
from rag_chain import RagChain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAG Chain Initialization (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_rag_chain() -> RagChain:
    """Initialize and cache the RAG chain."""
    config = Config.from_env()
    return RagChain(config)


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------


def render_source_chips(sources: list) -> None:
    """
    Render clickable source URL chips below a bot response.

    Args:
        sources: List of source URLs.
    """
    if not sources:
        return
    st.markdown("**Sources:**")
    cols = st.columns(len(sources))
    for i, url in enumerate(sources):
        # Extract scheme name from URL
        slug = url.split("/")[-1].replace("-", " ").title()
        with cols[i]:
            st.link_button(slug[:30], url)


def render_sidebar(config: Config) -> None:
    """
    Render the sidebar with app info and controls.

    Args:
        config: Application configuration.
    """
    with st.sidebar:
        st.title("About")
        st.markdown(
            """
        **MF-Guide** is a RAG-powered chatbot for HDFC mutual funds.

        It answers questions using data from 5 HDFC schemes:
        - Large Cap
        - Flexi Cap
        - ELSS
        - Small Cap
        - Balanced Advantage
        """
        )
        st.divider()
        st.markdown(f"**Embedding Model:** `{config.EMBEDDING_MODEL}`")
        st.markdown(f"**LLM:** `{config.LLM_MODEL}`")
        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def process_query(prompt: str) -> None:
    """
    Process a user query: generate response and update chat history.

    Args:
        prompt: User's question.
    """
    rag = get_rag_chain()

    # Get conversation history (last N messages for memory window)
    config = Config.from_env()
    history = st.session_state.messages[-config.MEMORY_WINDOW:]

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag.query(prompt, history=history)
                st.markdown(response.answer)
                if response.sources:
                    render_source_chips(response.sources)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                response = None

    # Save to history
    if response:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
            }
        )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(page_title="MF-Guide", page_icon="📈", layout="wide")
    st.title("MF-Guide: HDFC Mutual Fund Chatbot")
    st.caption(
        "Ask questions about HDFC mutual fund schemes. "
        "Answers are grounded in public data from Groww.in."
    )

    config = Config.from_env()

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested prompts
    suggestions = [
        "What is the expense ratio of HDFC ELSS?",
        "Compare risk levels of HDFC Large Cap vs Small Cap",
        "What is the minimum SIP for HDFC Equity Fund?",
        "What are the tax benefits of HDFC ELSS?",
    ]

    # Render suggestion buttons
    cols = st.columns(len(suggestions))
    for i, q in enumerate(suggestions):
        with cols[i]:
            if st.button(q, key=f"suggest_{i}", use_container_width=True):
                st.session_state.messages.append(
                    {"role": "user", "content": q, "sources": []}
                )
                st.session_state["trigger_query"] = q
                st.rerun()

    st.divider()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_source_chips(msg["sources"])

    # Chat input
    if prompt := st.chat_input("Ask about HDFC mutual funds..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        process_query(prompt)

    # Handle suggestion button clicks
    if "trigger_query" in st.session_state:
        q = st.session_state.pop("trigger_query")
        process_query(q)

    render_sidebar(config)


if __name__ == "__main__":
    main()
