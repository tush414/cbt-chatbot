##################################################################
#  streamlit_app.py — CBT Chatbot Streamlit Frontend             #
#                                                                #
#  Features:                                                     #
#  - RAG store build / load on startup                           #
#  - Model selection: OpenAI models only                         #
#  - Full multi-turn CBT chat with interactive technique steps   #
#  - Sidebar: session summary, mood history, technique log       #
#  - Satisfaction ratings per response                           #
##################################################################

import os
import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ─────────────────
st.set_page_config(
    page_title="CBT Therapeutic Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global font & background ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a78bfa;
}

/* ── Chat messages ── */
.stChatMessage {
    border-radius: 12px;
    padding: 4px 0;
}

/* ── User bubble ── */
[data-testid="stChatMessage"][data-is-user="true"] .stChatMessageContent {
    background-color: #1e2a4a;
    border: 1px solid #2d4070;
    border-radius: 12px;
    color: #c9d4f0;
}

/* ── Bot bubble ── */
[data-testid="stChatMessage"][data-is-user="false"] .stChatMessageContent {
    background-color: #1a1f2e;
    border: 1px solid #2a3045;
    border-radius: 12px;
    color: #e0e0e0;
}

/* ── Step progress badge ── */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
    letter-spacing: 0.03em;
}

/* ── Crisis banner ── */
.crisis-banner {
    background-color: #3b1010;
    border: 1px solid #c53030;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #fed7d7;
    font-size: 0.9rem;
}

/* ── Mood chip ── */
.mood-chip {
    display: inline-block;
    background-color: #1e2a4a;
    border: 1px solid #3a4a7a;
    color: #93c5fd;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 12px;
    margin: 2px;
}

/* ── Cards ── */
.info-card {
    background-color: #1a1f2e;
    border: 1px solid #2a3045;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 0.85rem;
    color: #c0c8e0;
}

/* ── Technique active indicator ── */
.technique-active {
    background: linear-gradient(135deg, #1a2e1a, #1a3020);
    border: 1px solid #2d6a4f;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
    color: #81e6a0;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    background-color: #1a1f2e !important;
    border: 1px solid #2a3045 !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
}

/* ── Text input ── */
.stChatInputContainer {
    background-color: #1a1f2e;
    border-top: 1px solid #2a3045;
}

/* ── Satisfaction stars ── */
.sat-row { display: flex; gap: 6px; margin: 4px 0; flex-wrap: wrap; }
.sat-btn {
    cursor: pointer;
    font-size: 1.2rem;
    background: none;
    border: none;
    padding: 2px;
    line-height: 1;
}

/* ── Divider ── */
hr { border-color: #2a3045; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    background-color: #1a1f2e;
    border-radius: 8px 8px 0 0;
    color: #a0aec0;
}
.stTabs [aria-selected="true"] {
    background-color: #2d3748 !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  IMPORTS (after page config)
# ══════════════════════════════════════════════════════════════
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Import your existing modules
from medical_rag import load_medical_rag_store, build_medical_rag_store
from chatbot import (
    build_cbt_graph,
    CBTState,
    TECHNIQUE_STEPS,
    TECHNIQUE_DESCRIPTIONS,
    CRISIS_RESPONSE,
    TECHNIQUE_HOMEWORK,
    MOOD_EMPATHY_PROFILES,
    SINGLE_TURN_TECHNIQUES,
)


# ══════════════════════════════════════════════════════════════
#  MODEL CONFIGURATION — OpenAI only
# ══════════════════════════════════════════════════════════════
MODEL_OPTIONS = {
    "🟢 GPT-4o Mini (fast)":    "gpt-4o-mini",
    "🔵 GPT-4o (most capable)": "gpt-4o",
    "⚡ GPT-3.5 Turbo (legacy)": "gpt-3.5-turbo",
}

RETRIEVAL_OPTIONS = {
    "Similarity (default)": "similarity",
    "MMR (diversity-aware)": "mmr",
    "Hybrid":               "hybrid",
}

PERSIST_PATH = "medical_chroma"
MOOD_EMOJI = {
    "anxious":     "😰",
    "depressed":   "😔",
    "angry":       "😤",
    "overwhelmed": "😵",
    "hopeful":     "🌟",
    "confused":    "🤔",
    "neutral":     "😐",
    "grieving":    "💔",
    "lonely":      "🫂",
}


# ══════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "chat_history":        [],
        "session_log":         [],
        "homework":            "None assigned yet",
        "session_number":      1,
        "turn_number":         0,
        "active_technique":    "",
        "active_step_index":   0,
        "step_answers":        [],
        "rag_loaded":          False,
        "rag_stores":          None,
        "selected_model_key":  list(MODEL_OPTIONS.keys())[0],
        "retrieval_strategy":  "similarity",
        "satisfaction_scores": [],
        "mood_history":        [],
        "technique_history":   [],
        "show_welcome":        True,
        "pending_rating":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
#  LLM FACTORY — OpenAI only
# ══════════════════════════════════════════════════════════════
def get_llm(model_key: str) -> ChatOpenAI:
    model_id = MODEL_OPTIONS[model_key]
    return ChatOpenAI(model=model_id, temperature=0.4)


# ══════════════════════════════════════════════════════════════
#  RAG STORE MANAGEMENT
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_or_build_rag(persist_path: str):
    """Try loading existing store; build if not found. Cached across reruns."""
    stores = load_medical_rag_store(persist_path)
    if stores is None:
        stores = build_medical_rag_store(persist_path)
    return stores


# ══════════════════════════════════════════════════════════════
#  GRAPH BUILDER — cached per model key + retrieval strategy
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_graph(model_key: str, retrieval_strategy: str, persist_path: str):
    """Build and cache the LangGraph for the chosen model & strategy."""
    import chatbot as cb
    cb.RETRIEVAL_STRATEGY = retrieval_strategy
    cb.LLM = get_llm(model_key)

    stores = load_or_build_rag(persist_path)
    return build_cbt_graph(stores["medical"], stores["cbt"])


# ══════════════════════════════════════════════════════════════
#  CHAT FUNCTION
# ══════════════════════════════════════════════════════════════
def run_chat_turn(user_message: str) -> str:
    st.session_state.turn_number += 1

    graph = get_graph(
        st.session_state.selected_model_key,
        st.session_state.retrieval_strategy,
        PERSIST_PATH,
    )

    initial_state: CBTState = {
        "user_query":          user_message,
        "crisis_detected":     False,
        "mood":                "neutral",
        "selected_technique":  "",
        "technique_rationale": "",
        "medical_context":     [],
        "cbt_context":         [],
        "final_response":      "",
        "session_log":         st.session_state.session_log.copy(),
        "session_number":      st.session_state.session_number,
        "turn_number":         st.session_state.turn_number,
        "homework":            st.session_state.homework,
        "chat_history":        st.session_state.chat_history.copy(),
        "satisfaction_score":  -1,
        "active_technique":    st.session_state.active_technique,
        "active_step_index":   st.session_state.active_step_index,
        "step_answers":        st.session_state.step_answers.copy(),
    }

    final_state = graph.invoke(initial_state)
    response    = final_state["final_response"]

    # Persist state back
    st.session_state.chat_history.append({"role": "user",      "content": user_message})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.session_log       = final_state.get("session_log",       st.session_state.session_log)
    st.session_state.homework          = final_state.get("homework",          st.session_state.homework)
    st.session_state.active_technique  = final_state.get("active_technique",  "")
    st.session_state.active_step_index = final_state.get("active_step_index", 0)
    st.session_state.step_answers      = final_state.get("step_answers",      [])

    detected_mood = final_state.get("mood", "neutral")
    if detected_mood:
        st.session_state.mood_history.append(detected_mood)
    detected_tech = final_state.get("selected_technique", "")
    if detected_tech:
        st.session_state.technique_history.append(detected_tech)

    return response


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 CBT Chatbot")
        st.markdown("*Powered by LangGraph + RAG*")
        st.divider()

        # ── Model & strategy selection ─────────────────────────
        st.markdown("### ⚙️ Configuration")

        new_model = st.selectbox(
            "OpenAI Model",
            options=list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model_key),
            help="All models require OPENAI_API_KEY",
        )
        if new_model != st.session_state.selected_model_key:
            st.session_state.selected_model_key = new_model
            get_graph.clear()
            st.rerun()

        new_strategy = st.selectbox(
            "Retrieval Strategy",
            options=list(RETRIEVAL_OPTIONS.keys()),
            index=list(RETRIEVAL_OPTIONS.values()).index(st.session_state.retrieval_strategy),
            help="How documents are retrieved from ChromaDB",
        )
        st.session_state.retrieval_strategy = RETRIEVAL_OPTIONS[new_strategy]

        # ── RAG Store Management ────────────────────────────────
        st.divider()
        st.markdown("### 📚 RAG Store")

        rag_exists = (
            os.path.exists(PERSIST_PATH) and bool(os.listdir(PERSIST_PATH))
            if os.path.exists(PERSIST_PATH) else False
        )

        if rag_exists:
            st.success("✅ RAG store found", icon="✅")
        else:
            st.warning("⚠️ RAG store not found", icon="⚠️")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload", use_container_width=True, help="Reload existing store"):
                load_or_build_rag.clear()
                get_graph.clear()
                st.toast("RAG store reloaded!", icon="✅")
        with col2:
            if st.button("🔨 Rebuild", use_container_width=True, help="Scrape & rebuild from scratch"):
                with st.spinner("Building RAG store… (this takes a few minutes)"):
                    load_or_build_rag.clear()
                    get_graph.clear()
                    build_medical_rag_store(PERSIST_PATH)
                st.toast("RAG store rebuilt!", icon="✅")

        # ── Session info ────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Session Info")

        st.markdown(f"""
<div class="info-card">
  <b>Session</b> #{st.session_state.session_number} &nbsp;|&nbsp;
  <b>Turns</b>: {st.session_state.turn_number}
</div>
""", unsafe_allow_html=True)

        if st.session_state.homework and st.session_state.homework != "None assigned yet":
            st.markdown("**📝 Current Homework**")
            st.markdown(f"""
<div class="info-card">
{st.session_state.homework}
</div>
""", unsafe_allow_html=True)

        # ── Mood history ────────────────────────────────────────
        if st.session_state.mood_history:
            st.markdown("**😊 Mood History**")
            mood_html = " ".join(
                f'<span class="mood-chip">{MOOD_EMOJI.get(m, "❓")} {m}</span>'
                for m in st.session_state.mood_history[-8:]
            )
            st.markdown(mood_html, unsafe_allow_html=True)

        # ── Technique history ───────────────────────────────────
        if st.session_state.technique_history:
            st.markdown("**🛠 Techniques Used**")
            unique_techs = list(dict.fromkeys(st.session_state.technique_history))
            for t in unique_techs:
                st.markdown(f"- {TECHNIQUE_DESCRIPTIONS.get(t, t)}")

        # ── Satisfaction scores ─────────────────────────────────
        if st.session_state.satisfaction_scores:
            avg = sum(st.session_state.satisfaction_scores) / len(st.session_state.satisfaction_scores)
            st.markdown(f"**⭐ Avg Satisfaction**: {avg:.1f}/10")

        # ── Active technique indicator ──────────────────────────
        if st.session_state.active_technique:
            steps   = TECHNIQUE_STEPS[st.session_state.active_technique]
            current = st.session_state.active_step_index
            name    = TECHNIQUE_DESCRIPTIONS.get(st.session_state.active_technique, st.session_state.active_technique)
            st.markdown(f"""
<div class="technique-active">
  🟢 <b>Active:</b> {name}<br>
  Step {current + 1} of {len(steps)}
</div>
""", unsafe_allow_html=True)

        # ── Reset ───────────────────────────────────────────────
        st.divider()
        if st.button("🔁 New Session", use_container_width=True, type="secondary"):
            keys_to_clear = [
                "chat_history", "session_log", "homework", "turn_number",
                "active_technique", "active_step_index", "step_answers",
                "satisfaction_scores", "mood_history", "technique_history",
                "show_welcome",
            ]
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.session_number += 1
            st.rerun()

        # ── Disclaimer ──────────────────────────────────────────
        st.divider()
        st.caption(
            "⚠️ This chatbot is for psychoeducational support only. "
            "It is NOT a substitute for professional mental health care. "
            "In crisis, please call 112 / 911 or a crisis helpline."
        )


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    init_session()
    render_sidebar()

    # ── Header ─────────────────────────────────────────────────
    st.markdown("## 🧠 CBT Therapeutic Chatbot")
    st.markdown(
        "*Evidence-based Cognitive Behavioural Therapy support — "
        "multi-turn, technique-guided, RAG-enhanced*"
    )
    st.divider()

    # ── RAG startup (runs once, cached) ────────────────────────
    if not st.session_state.rag_loaded:
        with st.spinner("🔄 Loading RAG knowledge stores…"):
            try:
                get_graph(
                    st.session_state.selected_model_key,
                    st.session_state.retrieval_strategy,
                    PERSIST_PATH,
                )
                st.session_state.rag_loaded = True
            except Exception as e:
                st.error(f"❌ Failed to load RAG store: {e}")
                st.info("Click **🔨 Rebuild** in the sidebar to build the RAG store from scratch.")
                return

    # ── Welcome message ─────────────────────────────────────────
    if st.session_state.show_welcome:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(
                "Hello, and welcome. 💙 I'm here to support you using "
                "**Cognitive Behavioural Therapy (CBT)** principles.\n\n"
                "I'll listen carefully, detect how you're feeling, and guide you "
                "through an evidence-based CBT technique tailored to this moment.\n\n"
                "**How are you feeling today, and what's on your mind?**"
            )
        st.session_state.show_welcome = False

    # ── Render existing chat history ────────────────────────────
    for i, msg in enumerate(st.session_state.chat_history):
        avatar = "🧑" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            is_last_assistant = (
                msg["role"] == "assistant"
                and i == len(st.session_state.chat_history) - 1
                and not st.session_state.active_technique
            )
            if is_last_assistant:
                _render_rating_row(i)

    # ── Active technique step badge ─────────────────────────────
    if st.session_state.active_technique:
        steps  = TECHNIQUE_STEPS[st.session_state.active_technique]
        step_i = st.session_state.active_step_index
        name   = TECHNIQUE_DESCRIPTIONS.get(st.session_state.active_technique, "")
        label  = steps[step_i]["label"] if step_i < len(steps) else "closing"
        st.markdown(
            f'<div class="step-badge">⚡ {name} — Step {step_i + 1}/{len(steps)}: {label}</div>',
            unsafe_allow_html=True,
        )

    # ── Chat input ──────────────────────────────────────────────
    placeholder = (
        f"Step {st.session_state.active_step_index + 1} — type your response…"
        if st.session_state.active_technique
        else "Share what's on your mind…"
    )

    if user_input := st.chat_input(placeholder):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Thinking…"):
                response = run_chat_turn(user_input)
            st.markdown(response)

            if not st.session_state.active_technique:
                _render_rating_row(len(st.session_state.chat_history) - 1)

        st.rerun()


# ══════════════════════════════════════════════════════════════
#  SATISFACTION RATING ROW
# ══════════════════════════════════════════════════════════════
def _render_rating_row(msg_index: int):
    rating_key = f"rating_{msg_index}"

    if rating_key in st.session_state:
        given = st.session_state[rating_key]
        st.caption(f"⭐ You rated this {given}/10")
        return

    st.caption("How helpful was this response?")

    cols = st.columns(11)
    for score in range(11):
        with cols[score]:
            icon = "😔" if score <= 3 else "😐" if score <= 6 else "😊"
            if st.button(f"{icon}\n{score}", key=f"rate_{msg_index}_{score}", use_container_width=True):
                st.session_state[rating_key] = score
                st.session_state.satisfaction_scores.append(score)
                if st.session_state.session_log:
                    st.session_state.session_log[-1]["satisfaction_score"] = score
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()