##################################################################
#  cbt_chatbot.py — CBT LangGraph Chatbot  (Interactive Edition) #
#                                                               #
#  LangGraph Pipeline                                           #
#  ──────────────────                                           #
#  Every turn first checks: is a technique mid-execution?       #
#                                                               #
#  YES → [technique_step_node] → [session_tracker] → END       #
#                                                               #
#  NO  → [crisis_detection]                                     #
#             │ crisis → END                                    #
#             │ safe ↓                                          #
#        [mood_detector]                                        #
#             ↓                                                 #
#        [cbt_technique_selector]                               #
#             ↓                                                 #
#        [rag_retriever]                                        #
#             ↓                                                 #
#        [cbt_response_generator]  ← starts technique if       #
#             ↓                       interactive               #
#        [session_tracker] → END                                #
##################################################################

from __future__ import annotations

import os
import json
import datetime
from typing import TypedDict, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

from medical_rag import load_medical_rag_store

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  EVALUATION CONFIGURATION
# ══════════════════════════════════════════════════════════════
RETRIEVAL_STRATEGY: str = "similarity"
# Options: "similarity" | "mmr" | "hybrid"

LLM_MODEL: str = "gpt-4o-mini"

LLM = ChatOpenAI(model=LLM_MODEL, temperature=0.4)

# ──────────────────────────────────────────────────────────────
#  Crisis response
# ──────────────────────────────────────────────────────────────
CRISIS_RESPONSE = """
I'm really glad you reached out, and I want you to know that you are not alone. 💙

What you're feeling right now is real, and it matters. Please reach out to a
crisis professional who can give you the proper support you deserve.

**Crisis Helplines — available 24 / 7:**

| Country       | Helpline                    | Contact                    |
|---------------|-----------------------------|----------------------------|
| India         | iCall                       | 9152987821                 |
| India         | Vandrevala Foundation       | 1860-2662-345              |
| India         | AASRA                       | 9820466627                 |
| USA           | 988 Suicide & Crisis Line   | Call or text **988**       |
| UK            | Samaritans                  | **116 123**                |
| International | Befrienders Worldwide       | https://befrienders.org    |

**Please also speak with a licensed therapist or psychiatrist.**
They are trained to help with exactly what you are going through.

If you are in immediate danger, call your local emergency services (112 / 911).

I am an AI assistant — I care about your wellbeing, but I cannot replace
professional help. You matter, and things can get better. 🙏
""".strip()

# ══════════════════════════════════════════════════════════════
#  INTERACTIVE TECHNIQUE STEPS
#  Each technique is a list of step dicts:
#    prompt  — what the bot says / asks for this step
#    label   — short name shown in logs
#    reflect — if True, the LLM reflects on the user's answer
#              before moving to the next step
# ══════════════════════════════════════════════════════════════
TECHNIQUE_STEPS: dict[str, list[dict]] = {

    "grounding_5_4_3_2_1": [
        {
            "label"  : "intro",
            "prompt" : (
                "Let's do the 5-4-3-2-1 grounding exercise together, right now. "
                "It will take about 2 minutes and will help anchor you in the present moment.\n\n"
                "👁️  **Step 1 of 5 — SIGHT**\n"
                "Look around the room slowly. Name **5 things you can see** right now. "
                "Take your time — notice colours, shapes, textures.\n\n"
                "Type them out one by one when you're ready."
            ),
            "reflect": True,
        },
        {
            "label"  : "touch",
            "prompt" : (
                "✋  **Step 2 of 5 — TOUCH**\n"
                "Feel the surfaces around you — your hands on the keyboard, "
                "your feet on the floor, your clothes against your skin.\n\n"
                "Name **4 things you can feel** right now — describe the sensation "
                "(rough, smooth, warm, cool, soft…)."
            ),
            "reflect": True,
        },
        {
            "label"  : "sound",
            "prompt" : (
                "👂  **Step 3 of 5 — SOUND**\n"
                "Close your eyes for a moment if that feels okay. "
                "Listen carefully — near sounds and far ones.\n\n"
                "Name **3 things you can hear** right now."
            ),
            "reflect": True,
        },
        {
            "label"  : "smell",
            "prompt" : (
                "👃  **Step 4 of 5 — SMELL**\n"
                "Take a slow breath in through your nose.\n\n"
                "Name **2 things you can smell** — even faint scents count. "
                "If you can't smell anything specific, describe the quality of the air."
            ),
            "reflect": True,
        },
        {
            "label"  : "taste",
            "prompt" : (
                "👅  **Step 5 of 5 — TASTE**\n"
                "Notice any taste in your mouth right now — even just the neutral "
                "sensation of your tongue.\n\n"
                "Name **1 thing you can taste**."
            ),
            "reflect": True,
        },
        {
            "label"  : "close",
            "prompt" : None,   # generated dynamically from all answers
            "reflect": False,
            "is_closing": True,
        },
    ],

    "breathing_box_4_4_4_4": [
        {
            "label"  : "intro",
            "prompt" : (
                "Let's do box breathing together right now. It takes less than 2 minutes "
                "and physically slows your nervous system down.\n\n"
                "**Round 1 of 3**\n"
                "→ Breathe **IN** slowly… 1… 2… 3… 4…\n"
                "→ **HOLD**… 1… 2… 3… 4…\n"
                "→ Breathe **OUT** slowly… 1… 2… 3… 4…\n"
                "→ **HOLD**… 1… 2… 3… 4…\n\n"
                "Take your time doing it. Type 'done' when you've finished round 1."
            ),
            "reflect": False,
        },
        {
            "label"  : "round_2",
            "prompt" : (
                "**Round 2 of 3**\n"
                "→ IN… 1… 2… 3… 4…\n"
                "→ HOLD… 1… 2… 3… 4…\n"
                "→ OUT… 1… 2… 3… 4…\n"
                "→ HOLD… 1… 2… 3… 4…\n\n"
                "Type 'done' when finished."
            ),
            "reflect": False,
        },
        {
            "label"  : "round_3",
            "prompt" : (
                "**Round 3 of 3 — last one.**\n"
                "→ IN… 1… 2… 3… 4…\n"
                "→ HOLD… 1… 2… 3… 4…\n"
                "→ OUT… 1… 2… 3… 4…\n"
                "→ HOLD… 1… 2… 3… 4…\n\n"
                "Type 'done' when finished."
            ),
            "reflect": False,
        },
        {
            "label"  : "close",
            "prompt" : (
                "Well done. 🌿\n\n"
                "Notice what's happening in your body right now — even a small shift "
                "in tension or heart rate. That's your parasympathetic nervous system "
                "responding. You just physiologically interrupted the anxiety response.\n\n"
                "How are you feeling compared to a few minutes ago? "
                "Rate your calm level from 0–10."
            ),
            "reflect": True,
            "is_closing": True,
        },
    ],

    "thought_record": [
        {
            "label"  : "situation",
            "prompt" : (
                "Let's work through a **CBT Thought Record** together. "
                "This is one of the most powerful tools in CBT — we examine a difficult "
                "thought from multiple angles to find a more balanced perspective.\n\n"
                "**Column 1 — SITUATION**\n"
                "Describe the specific situation that triggered how you're feeling. "
                "Be concrete: What happened? Where were you? Who was involved? When?\n\n"
                "Take your time and write it out."
            ),
            "reflect": False,
        },
        {
            "label"  : "emotions",
            "prompt" : (
                "**Column 2 — EMOTIONS**\n"
                "What emotions did you feel in that moment? "
                "Name each one and rate its intensity from 0–100%.\n\n"
                "For example: *anxious (75%), ashamed (60%), helpless (80%)*\n\n"
                "What were yours?"
            ),
            "reflect": True,
        },
        {
            "label"  : "auto_thought",
            "prompt" : (
                "**Column 3 — AUTOMATIC THOUGHT**\n"
                "What thought flashed through your mind in that moment? "
                "This is often a quick, automatic thought — not a reasoned conclusion.\n\n"
                "It might sound like:\n"
                "  • 'I always mess things up'\n"
                "  • 'They think I'm stupid'\n"
                "  • 'I can't cope with this'\n\n"
                "What was YOUR automatic thought?"
            ),
            "reflect": True,
        },
        {
            "label"  : "evidence_for",
            "prompt" : (
                "**Column 4 — EVIDENCE FOR**\n"
                "What facts or evidence support that automatic thought?\n\n"
                "Be specific — only list actual facts, not feelings or assumptions. "
                "For example: 'I did make a mistake in the report' is evidence. "
                "'People probably think I'm incompetent' is not.\n\n"
                "What evidence supports your thought?"
            ),
            "reflect": True,
        },
        {
            "label"  : "evidence_against",
            "prompt" : (
                "**Column 5 — EVIDENCE AGAINST**\n"
                "Now — what facts or evidence *contradict* that thought?\n\n"
                "Ask yourself:\n"
                "  • What would I tell a close friend if they had this thought?\n"
                "  • Have I handled similar situations before?\n"
                "  • Am I ignoring anything positive?\n"
                "  • Is there another explanation?\n\n"
                "What evidence works against your automatic thought?"
            ),
            "reflect": True,
        },
        {
            "label"  : "balanced_thought",
            "prompt" : (
                "**Column 6 — BALANCED THOUGHT**\n"
                "Taking *both* sides into account — the evidence for AND against — "
                "write a more realistic, balanced version of the original thought.\n\n"
                "This isn't about being positive. It's about being *accurate*.\n\n"
                "For example: *'I made a mistake, AND I have handled difficult situations "
                "before. One mistake doesn't define my ability.'*\n\n"
                "What's your balanced thought?"
            ),
            "reflect": True,
        },
        {
            "label"  : "outcome",
            "prompt" : (
                "**Column 7 — OUTCOME**\n"
                "Go back to the emotions you rated in Column 2.\n\n"
                "Now that you've examined the thought from both sides and found a more "
                "balanced perspective — how would you rate those same emotions now?\n\n"
                "List each emotion and its new intensity (0–100%). Even a small drop "
                "of 10–20% is a meaningful shift."
            ),
            "reflect": True,
            "is_closing": True,
        },
    ],

    "behavioural_activation": [
        {
            "label"  : "intro",
            "prompt" : (
                "Let's work through **Behavioural Activation** together.\n\n"
                "Here's the key insight: when we're low, we wait to *feel* motivated "
                "before acting. But motivation actually comes *after* action, not before. "
                "So the strategy is: act first, even at 10% energy, and let the mood follow.\n\n"
                "**Step 1 — IDENTIFY**\n"
                "Think of ONE activity that used to bring you even a small sense of "
                "pleasure, connection, or accomplishment — even something tiny.\n\n"
                "It doesn't have to feel exciting right now. What comes to mind?"
            ),
            "reflect": True,
        },
        {
            "label"  : "schedule",
            "prompt" : (
                "**Step 2 — SCHEDULE IT**\n"
                "Vague intentions don't happen. We need a specific commitment.\n\n"
                "When *exactly* this week can you do that activity? "
                "Give me a day and a time — even just 10–15 minutes.\n\n"
                "For example: *'Wednesday evening at 7pm'*"
            ),
            "reflect": True,
        },
        {
            "label"  : "barrier",
            "prompt" : (
                "**Step 3 — ANTICIPATE BARRIERS**\n"
                "What might get in the way of you actually doing this?\n\n"
                "Think honestly — tiredness, self-doubt, something coming up? "
                "Name the most likely obstacle."
            ),
            "reflect": True,
        },
        {
            "label"  : "commit",
            "prompt" : (
                "**Step 4 — LOWER THE BAR & COMMIT**\n"
                "The rule is: you only have to *start*. Five minutes counts. "
                "Showing up at all — even reluctantly — is a win.\n\n"
                "Before and after you do it, rate your mood from 0–10. "
                "This collects real evidence about the action-mood link.\n\n"
                "So — what's your commitment? Write it out in one sentence: "
                "*'I will [activity] on [day] at [time], even if I only feel like it 10%.'*"
            ),
            "reflect": True,
            "is_closing": True,
        },
    ],

    "problem_solving": [
        {
            "label"  : "define",
            "prompt" : (
                "Let's work through **Problem-Solving Therapy** step by step.\n\n"
                "**Step 1 — DEFINE THE PROBLEM**\n"
                "Vague problems feel overwhelming. Specific problems feel solvable.\n\n"
                "Write your problem as ONE precise sentence. Not 'everything is a mess' "
                "but something like: *'I have 4 exams in 6 days and don't know where to start.'*\n\n"
                "What is YOUR specific problem?"
            ),
            "reflect": True,
        },
        {
            "label"  : "brainstorm",
            "prompt" : (
                "**Step 2 — BRAINSTORM SOLUTIONS**\n"
                "List every possible solution you can think of — no judging yet. "
                "Quantity over quality at this stage. Even impractical ideas count.\n\n"
                "Aim for at least 4–5 options. What could you potentially do?"
            ),
            "reflect": True,
        },
        {
            "label"  : "evaluate",
            "prompt" : (
                "**Step 3 — EVALUATE & CHOOSE**\n"
                "Look at the solutions you listed. For each one, quickly consider:\n"
                "  • Is it realistic right now?\n"
                "  • What are the pros and cons?\n"
                "  • What resources or support would it need?\n\n"
                "Which option — or combination — feels most doable? Tell me which one "
                "you're choosing and why."
            ),
            "reflect": True,
        },
        {
            "label"  : "first_step",
            "prompt" : (
                "**Step 4 — FIRST ACTION STEP**\n"
                "Break your chosen solution into the single smallest first step — "
                "something that takes less than 10 minutes to start.\n\n"
                "For example: *'Tonight I will write down all my subjects and "
                "estimate how many hours each needs.'*\n\n"
                "What is YOUR smallest first step, and when will you do it?"
            ),
            "reflect": True,
            "is_closing": True,
        },
    ],

    "self_compassion_break": [
        {
            "label"  : "intro",
            "prompt" : (
                "Let's do a **Self-Compassion Break** together. "
                "This takes about 2 minutes. It won't take the pain away, "
                "but it changes how you relate to it.\n\n"
                "Find a comfortable position. Place a hand on your heart "
                "or chest if that feels okay.\n\n"
                "**Step 1 — ACKNOWLEDGE THE PAIN**\n"
                "Say to yourself — silently or out loud:\n"
                "*'This is a moment of real pain. This is hard. What I'm feeling is real.'*\n\n"
                "Don't try to fix it or push it away. Just acknowledge it.\n\n"
                "When you've done that, tell me in your own words — "
                "what is the pain you're acknowledging right now?"
            ),
            "reflect": True,
        },
        {
            "label"  : "common_humanity",
            "prompt" : (
                "**Step 2 — COMMON HUMANITY**\n"
                "Now remind yourself:\n"
                "*'I am not the only one who has felt this way. "
                "Suffering is part of being human. "
                "Many people, right now, are feeling something just like this.'*\n\n"
                "This isn't dismissing your pain — it's recognising you don't carry it alone.\n\n"
                "Does this feel true to you? What comes up when you sit with the idea "
                "that you're not alone in this?"
            ),
            "reflect": True,
        },
        {
            "label"  : "self_kindness",
            "prompt" : (
                "**Step 3 — SELF-KINDNESS**\n"
                "Ask yourself: if someone I deeply cared about was feeling exactly "
                "what I'm feeling right now — what would I say to them?\n\n"
                "Now say that to yourself. You might try:\n"
                "*'May I be gentle with myself right now.'*\n"
                "*'I don't have to have it all together. I'm doing the best I can.'*\n\n"
                "What kind words can you offer yourself in this moment? "
                "Write them out — even if they feel hard to believe right now."
            ),
            "reflect": True,
            "is_closing": True,
        },
    ],

    "socratic_questioning": [
        {
            "label"  : "identify_thought",
            "prompt" : (
                "Let's use **Socratic Questioning** to examine the thought that's "
                "driving how you feel. We're going to slow it down and look at it carefully.\n\n"
                "**Question 1 — IDENTIFY THE THOUGHT**\n"
                "What is the specific thought or belief that's most troubling you right now?\n\n"
                "Try to put it into one clear sentence — the thought itself, not the feeling. "
                "For example: *'I'm going to fail my exams'* or *'Nobody cares about me.'*\n\n"
                "What is YOUR thought?"
            ),
            "reflect": True,
        },
        {
            "label"  : "evidence",
            "prompt" : (
                "**Question 2 — EXAMINE THE EVIDENCE**\n"
                "Let's look at that thought as if we're scientists — not assuming it's "
                "true or false, just checking the evidence.\n\n"
                "  • What actual *facts* support this thought? (Not feelings — facts.)\n"
                "  • What facts *contradict* it?\n"
                "  • Are you certain this thought is true, or is it an interpretation?\n\n"
                "What do you find when you examine the evidence?"
            ),
            "reflect": True,
        },
        {
            "label"  : "alternative",
            "prompt" : (
                "**Question 3 — ALTERNATIVE PERSPECTIVES**\n"
                "Is there another way to see this situation?\n\n"
                "  • What would you say to a close friend who told you this thought?\n"
                "  • Has anyone else seen this situation differently?\n"
                "  • What's a more charitable interpretation?\n\n"
                "What alternative perspective can you find?"
            ),
            "reflect": True,
        },
        {
            "label"  : "perspective",
            "prompt" : (
                "**Question 4 — PERSPECTIVE IN TIME**\n"
                "In 6 months — or even 6 weeks — how significant do you think "
                "this moment will feel?\n\n"
                "  • What will you likely say to yourself looking back?\n"
                "  • Have you survived difficult situations before? What got you through?\n\n"
                "What does your future self say about this?"
            ),
            "reflect": True,
            "is_closing": True,
        },
    ],
}

# Techniques that are NOT interactive (delivered in one response)
SINGLE_TURN_TECHNIQUES = {
    "positive_data_log",
    "behavioural_experiment",
    "decatastrophising",
    "psychoeducation",
}

# ══════════════════════════════════════════════════════════════
#  Shared LangGraph State
# ══════════════════════════════════════════════════════════════
class CBTState(TypedDict):
    user_query:            str
    crisis_detected:       bool
    mood:                  str
    selected_technique:    str
    technique_rationale:   str
    medical_context:       list[str]
    cbt_context:           list[str]
    final_response:        str
    session_log:           list[dict]
    session_number:        int
    turn_number:           int
    homework:              str
    chat_history:          list[dict]
    satisfaction_score:    int           # 0–10, -1 = skipped
    # Interactive technique state
    active_technique:      str           # technique currently mid-execution
    active_step_index:     int           # which step we're on
    step_answers:          list[dict]    # collected user answers per step


# ══════════════════════════════════════════════════════════════
#  NODE 1 — Crisis Detection
# ══════════════════════════════════════════════════════════════
CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "self-harm",
    "self harm", "cut myself", "hurt myself", "want to die", "don't want to live",
    "don't want to be alive", "no reason to live", "take my own life",
    "overdose", "jump off", "hang myself", "not worth living",
    "wish i was dead", "better off dead", "can't go on", "give up on life",
]


def crisis_detection_node(state: CBTState) -> CBTState:
    query_lower = state["user_query"].lower()
    keyword_hit = any(kw in query_lower for kw in CRISIS_KEYWORDS)

    if not keyword_hit:
        prompt = (
            "You are a clinical crisis-detection classifier.\n"
            "Does this message contain ANY indication of suicidal ideation, "
            "self-harm, intent to hurt oneself, or wishing to not be alive?\n\n"
            f"Message: {state['user_query']}\n\n"
            "Answer with one word only: YES or NO."
        )
        result = LLM.invoke([HumanMessage(content=prompt)])
        keyword_hit = "YES" in result.content.upper()

    new_state = dict(state)
    new_state["crisis_detected"] = keyword_hit
    if keyword_hit:
        new_state["final_response"] = CRISIS_RESPONSE
        print("[CrisisDetection] ⚠️  CRISIS DETECTED")
    else:
        print("[CrisisDetection] ✓ No crisis detected")
    return new_state


def crisis_router(state: CBTState) -> Literal["mood_detector", "__end__"]:
    return "__end__" if state["crisis_detected"] else "mood_detector"


# ══════════════════════════════════════════════════════════════
#  NODE 2 — Mood Detector
# ══════════════════════════════════════════════════════════════
MOOD_PROMPT = """\
You are a clinical psychologist trained in CBT.
Analyse the PRIMARY emotional tone of this message.
Choose ONE mood from this list:

  anxious | depressed | angry | overwhelmed | hopeful | confused | neutral | grieving | lonely

Return ONLY the single mood word. No punctuation, no explanation.

Message: {query}
"""


def mood_detector_node(state: CBTState) -> CBTState:
    response = LLM.invoke(
        [HumanMessage(content=MOOD_PROMPT.format(query=state["user_query"]))]
    )
    mood = response.content.strip().lower().split()[0]
    new_state = dict(state)
    new_state["mood"] = mood
    print(f"[MoodDetector] Mood: {mood}")
    return new_state


# ══════════════════════════════════════════════════════════════
#  NODE 3 — CBT Technique Selector
# ══════════════════════════════════════════════════════════════
TECHNIQUE_MATRIX = {
    "anxious":     ("grounding_5_4_3_2_1",    "Grounding anchors you in the present and interrupts anxious thinking loops."),
    "depressed":   ("behavioural_activation",  "Behavioural activation breaks the inactivity-depression cycle."),
    "angry":       ("problem_solving",         "Problem-solving therapy channels anger into constructive action."),
    "overwhelmed": ("problem_solving",         "Breaking the problem into concrete steps reduces overwhelm."),
    "grieving":    ("self_compassion_break",   "The self-compassion break acknowledges pain with kindness."),
    "lonely":      ("behavioural_activation",  "Scheduling social activities gradually rebuilds connection."),
    "hopeful":     ("socratic_questioning",    "Socratic questioning helps build on existing strengths and insights."),
    "confused":    ("problem_solving",         "Problem-solving structures unclear situations into manageable steps."),
    "neutral":     ("psychoeducation",         "Psychoeducation provides relevant information to support understanding."),
}

TECHNIQUE_DESCRIPTIONS = {
    "grounding_5_4_3_2_1":    "5-4-3-2-1 Sensory Grounding",
    "breathing_box_4_4_4_4":  "Box Breathing (4-4-4-4)",
    "thought_record":         "CBT Thought Record (7 Columns)",
    "behavioural_activation": "Behavioural Activation",
    "socratic_questioning":   "Socratic Questioning",
    "behavioural_experiment": "Behavioural Experiment",
    "self_compassion_break":  "Self-Compassion Break",
    "positive_data_log":      "Positive Data Log",
    "decatastrophising":      "Decatastrophising",
    "problem_solving":        "Problem-Solving Therapy",
    "psychoeducation":        "Psychoeducation",
}


def cbt_technique_selector_node(state: CBTState) -> CBTState:
    mood = state.get("mood", "neutral")
    technique, rationale = TECHNIQUE_MATRIX.get(
        mood,
        ("socratic_questioning", "Socratic questioning is broadly applicable.")
    )
    new_state = dict(state)
    new_state["selected_technique"] = technique
    new_state["technique_rationale"] = rationale
    print(f"[TechniqueSelector] {TECHNIQUE_DESCRIPTIONS.get(technique, technique)}")
    return new_state


# ══════════════════════════════════════════════════════════════
#  NODE 4 — RAG Retriever
# ══════════════════════════════════════════════════════════════
def _deduplicate(docs: list) -> list:
    seen, unique = set(), []
    for d in docs:
        key = d.page_content[:150]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _retrieve(query: str, vectorstore: Chroma, k: int = 4, label: str = "") -> list:
    if RETRIEVAL_STRATEGY == "similarity":
        docs = vectorstore.similarity_search(query, k=k)
    elif RETRIEVAL_STRATEGY == "mmr":
        docs = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=k * 5, lambda_mult=0.55
        )
    elif RETRIEVAL_STRATEGY == "hybrid":
        sim_docs = vectorstore.similarity_search(query, k=k)
        kw_docs  = vectorstore.similarity_search(query.split("[")[0].strip(), k=k)
        docs = _deduplicate(sim_docs + kw_docs)
    else:
        raise ValueError(f"Unknown RETRIEVAL_STRATEGY '{RETRIEVAL_STRATEGY}'.")

    if label:
        print(f"  [{label}] {RETRIEVAL_STRATEGY} : {len(docs)} doc(s)")
    return docs


def rag_retriever_node(
    state: CBTState,
    medical_store: Chroma,
    cbt_store: Chroma,
) -> CBTState:
    query     = state["user_query"]
    mood      = state.get("mood", "neutral")
    technique = state.get("selected_technique", "")

    print(f"[RAG] strategy='{RETRIEVAL_STRATEGY}' | Retrieving from medical_general…")
    medical_docs = _retrieve(f"{query} emotional state: {mood}", medical_store, k=3, label="medical")

    print(f"[RAG] strategy='{RETRIEVAL_STRATEGY}' | Retrieving from cbt_knowledge…")
    cbt_docs = _retrieve(f"{query} mood: {mood} technique: {technique}", cbt_store, k=4, label="cbt")

    new_state = dict(state)
    new_state["medical_context"] = [d.page_content for d in medical_docs[:5]]
    new_state["cbt_context"]     = [d.page_content for d in cbt_docs[:6]]
    return new_state


# ══════════════════════════════════════════════════════════════
#  MOOD EMPATHY PROFILES
# ══════════════════════════════════════════════════════════════
MOOD_EMPATHY_PROFILES = {
    "anxious": {
        "tone": "calm, steady, and grounding — like a quiet anchor in a storm",
        "opening": (
            "Acknowledge the physical reality of anxiety first — the racing thoughts, "
            "tight chest, restlessness. Let them feel seen in their body before their mind. "
            "Slow the pace of your language down deliberately."
        ),
        "language": (
            "Use slowing, present-moment language: 'right now, in this moment', "
            "'let's take this one breath at a time', 'you don't have to solve everything at once'. "
            "Normalise anxiety as the brain's threat-protection system working overtime — "
            "not a sign of weakness or failure. Use 'we' to feel collaborative: 'let's try this together'."
        ),
        "what_they_need": (
            "They need to feel that the anxiety makes complete sense given their situation, "
            "that they are not broken, and that there is something concrete they can do right now "
            "to bring their nervous system back down."
        ),
        "avoid": (
            "Never say 'just relax', 'don't worry', 'it's not a big deal', or rush to solutions "
            "before they feel heard. Do not be peppy or artificially upbeat — it feels dismissive."
        ),
        "depth_note": (
            "Anxiety often contains a specific feared story — name it if you can detect it "
            "from their message ('it sounds like part of you is afraid that…'). "
            "Specificity feels far more understood than generic comfort."
        ),
    },
    "depressed": {
        "tone": "gentle, patient, and quietly present — never falsely cheerful or hurried",
        "opening": (
            "Meet them in the heaviness without trying to lift them out of it too quickly. "
            "Depression involves exhaustion — even reading takes effort. Keep sentences shorter. "
            "Acknowledge the weight before anything else."
        ),
        "language": (
            "Use soft, low-demand language that removes pressure: 'even the smallest step counts', "
            "'you don't have to feel motivated to start — that comes after, not before', "
            "'one tiny thing is enough'. Acknowledge how hard it is to engage when depressed — "
            "that reaching out at all is genuinely significant."
        ),
        "what_they_need": (
            "They need to feel that their heaviness is real and valid, not something to be "
            "argued out of or positivity-washed away. They need a sense that movement — "
            "even microscopic — is possible without requiring them to feel better first."
        ),
        "avoid": (
            "Never say 'cheer up', 'think positive', 'others have it worse', "
            "or imply they should feel better by now. Do not give long lists — "
            "it feels overwhelming. Do not be relentlessly upbeat."
        ),
        "depth_note": (
            "Depression often involves a harsh inner critic. If you detect self-criticism or "
            "hopelessness in their message, name it gently: 'it sounds like part of you is "
            "being very hard on yourself right now.' This alone can feel profoundly validating."
        ),
    },
    "angry": {
        "tone": "steady, grounded, and fully non-judgmental — not defensive, not flinching",
        "opening": (
            "Validate the anger completely and specifically before doing anything else. "
            "Anger is always communicating something — often injustice, hurt, or a violated boundary. "
            "Reflect back what they're angry about with full acknowledgment that it makes sense."
        ),
        "language": (
            "Name the anger directly and without softening it: 'It makes complete sense you'd feel "
            "furious about this', 'that's a genuinely infuriating situation'. Then, once validated, "
            "gently explore what might be underneath — often hurt, fear, or a deep sense of unfairness."
        ),
        "what_they_need": (
            "They need to feel that their anger is legitimate and understood — not managed away or "
            "spiritualised. Then they need help channelling it constructively rather than letting "
            "it spiral or be suppressed."
        ),
        "avoid": (
            "Never tell them to calm down, take a breath first, or 'look at it from the other side' "
            "before the anger has been fully validated. Do not be passive or mealy-mouthed — "
            "match the directness of their emotion."
        ),
        "depth_note": (
            "Anger is often a secondary emotion covering hurt or fear. If you sense this in their "
            "message, name it gently after validating the anger: 'underneath the anger, I wonder "
            "if there's also some hurt here?'"
        ),
    },
    "overwhelmed": {
        "tone": "calm, organised, and quietly containing — a steady hand in chaos",
        "opening": (
            "Acknowledge that when everything feels like too much at once, the mind can't prioritise "
            "and the body feels frozen. Validate the paralysis — it makes complete sense. "
            "Then gently help them zoom in to just one thing."
        ),
        "language": (
            "Use containment language: 'let's just focus on one thing right now', "
            "'we don't need to solve everything today — just the next small step', "
            "'one thing at a time'. Structure is deeply reassuring when someone is overwhelmed — "
            "use numbered steps and keep sentences clear."
        ),
        "what_they_need": (
            "They need the feeling of chaos to be acknowledged, then gently organised. "
            "A sense that they don't have to do everything at once — just one concrete, "
            "achievable next step — is often enough to break the paralysis."
        ),
        "avoid": (
            "Do not give long lists of advice — it adds to the overwhelm. "
            "Do not be overly philosophical or abstract. "
            "Do not minimise by saying 'just take it one day at a time' without "
            "actually helping them identify what that one thing is."
        ),
        "depth_note": (
            "Overwhelm often comes from a combination of too many tasks AND the belief "
            "that they all need to be done perfectly. If you detect perfectionism or "
            "fear of failure in their message, name it."
        ),
    },
    "grieving": {
        "tone": "tender, unhurried, and deeply compassionate — grief has no timeline and no fix",
        "opening": (
            "Sit with the pain before doing anything else. Do not try to reframe, silver-line, "
            "or problem-solve grief — it needs to be witnessed first. "
            "Let them feel that their loss truly matters and that you are not rushing them anywhere."
        ),
        "language": (
            "Use slow, spacious language. Reflect back what they've lost with full weight: "
            "'losing someone/something that meant that much — that's an enormous loss, "
            "and it makes complete sense that you're carrying this so heavily.' "
            "Use silence-adjacent language — short sentences, pauses implied."
        ),
        "what_they_need": (
            "They need to feel that their grief is legitimate, that there is no 'right' way "
            "or timeline to grieve, and that they are not alone in carrying this. "
            "They do not need solutions — they need compassionate company."
        ),
        "avoid": (
            "Never say 'everything happens for a reason', 'they're in a better place', "
            "'at least…', 'you need to move on', or rush toward acceptance or silver linings. "
            "Do not problem-solve. Do not be cheerful."
        ),
        "depth_note": (
            "Grief is not only about death — it includes loss of relationships, identity, "
            "health, opportunities, and dreams. Honour whatever form their grief takes "
            "without implying some losses are more valid than others."
        ),
    },
    "lonely": {
        "tone": "warm, genuinely present, and connecting — the response itself should feel like company",
        "opening": (
            "Acknowledge the specific ache of feeling unseen, disconnected, or invisible. "
            "Loneliness is one of the most painful human experiences and is often accompanied "
            "by shame — as if feeling lonely means something is wrong with them. "
            "Make them feel truly met right now, in this exchange."
        ),
        "language": (
            "Use connecting language that closes distance: 'I hear you', 'what you're describing "
            "makes complete sense', 'you're not invisible here'. Gently normalise loneliness as "
            "a universal human experience without minimising their specific pain."
        ),
        "what_they_need": (
            "They need to feel genuinely heard and less alone — starting right now, in this "
            "conversation. Then they need gentle, low-pressure ideas for rebuilding connection "
            "that don't feel forced or overwhelming."
        ),
        "avoid": (
            "Do not immediately suggest 'just reach out to someone' — they may have tried and "
            "it didn't work, which deepens the pain. Do not be clinical or distant. "
            "Do not minimise by saying 'everyone feels lonely sometimes'."
        ),
        "depth_note": (
            "Loneliness is often accompanied by a belief that one is fundamentally unlikeable "
            "or different from others. If you detect this in their message, name it with care "
            "and gently challenge it."
        ),
    },
    "hopeful": {
        "tone": "warm, energised, and genuinely collaborative — build on their momentum",
        "opening": (
            "Honour and reflect back the hope they're bringing. Hope is a psychological resource — "
            "it deserves to be named and celebrated, not taken for granted. "
            "Show genuine enthusiasm for where they are."
        ),
        "language": (
            "Use forward-looking, strengths-based language: 'that sounds like a real shift', "
            "'what you're describing takes genuine courage', 'let's build on that'. "
            "Help them translate hope into concrete intentions."
        ),
        "what_they_need": (
            "They need their progress or positive orientation affirmed, and then help "
            "translating that energy into specific, sustainable next steps — "
            "so the hope becomes action and doesn't stay abstract."
        ),
        "avoid": (
            "Do not dampen optimism with excessive caution or caveats. "
            "Do not be patronising or treat their hope as naive. "
            "Do not immediately introduce problems or challenges."
        ),
        "depth_note": (
            "Hope is fragile — it can tip into pressure ('now I have to follow through'). "
            "If you sense any anxiety underneath the hope, acknowledge it gently alongside "
            "the celebration."
        ),
    },
    "confused": {
        "tone": "patient, clarifying, and thoughtfully guiding — a calm presence in fog",
        "opening": (
            "Normalise the confusion warmly — confusion often means someone is thinking "
            "carefully about something genuinely complex. It is not a failure of intelligence. "
            "Help them feel less lost by reflecting back what you've heard and organising it."
        ),
        "language": (
            "Use structuring language that creates clarity: 'let's untangle this together', "
            "'it sounds like there might be a few different things pulling at you — let me "
            "reflect back what I'm hearing', 'let's just focus on one piece of this'. "
            "Use numbered steps and clear, plain language."
        ),
        "what_they_need": (
            "They need their confusion acknowledged as legitimate, then gently organised "
            "so it feels less overwhelming. A clear framework or set of questions to work "
            "through gives them a structure to hold onto."
        ),
        "avoid": (
            "Do not overwhelm with information or multiple frameworks at once. "
            "Do not make them feel foolish for being confused. "
            "Do not be abstract or philosophical when they need practical clarity."
        ),
        "depth_note": (
            "Confusion can sometimes be a defence against a difficult decision or feeling "
            "that the person isn't ready to face. If this seems present, hold it gently "
            "rather than pushing for resolution."
        ),
    },
    "neutral": {
        "tone": "clear, warm, and genuinely engaged — professional but fully human",
        "opening": (
            "Respond with genuine interest and warmth. Even neutral or informational queries "
            "deserve full engagement. Be present, not mechanical."
        ),
        "language": (
            "Be clear, direct, and friendly. Use plain language. "
            "Offer useful information drawn from the CBT knowledge context. "
            "Keep the door open for deeper sharing if they want it: "
            "'feel free to share more about what's going on for you if that would help'."
        ),
        "what_they_need": (
            "Accurate, useful information delivered in a way that feels human and accessible — "
            "not like reading a textbook."
        ),
        "avoid": (
            "Do not be cold, robotic, or purely clinical. "
            "Do not assume everything is fine just because the tone is neutral."
        ),
        "depth_note": (
            "Neutral queries can sometimes be a way of approaching something harder indirectly. "
            "Stay open to what might be underneath."
        ),
    },
}

TECHNIQUE_HOMEWORK = {
    "grounding_5_4_3_2_1":    "Once each day this week — especially in moments of stress — run through the 5-4-3-2-1 exercise on your own. Note your anxiety level (0–10) before and after.",
    "breathing_box_4_4_4_4":  "Practise box breathing for 3 rounds every morning this week before checking your phone. Also use it whenever you notice tension rising.",
    "thought_record":         "When a strong negative emotion arises this week, fill in a thought record: Situation | Emotions | Automatic Thought | Evidence Against | Balanced Thought. One record per day is enough.",
    "behavioural_activation": "Do the activity you scheduled. Rate your mood 0–10 before and after. Write both numbers down.",
    "problem_solving":        "Write your specific problem in one sentence, list 4 solutions, circle the most doable one, and take the first small step within 24 hours.",
    "self_compassion_break":  "Once a day this week — especially after a difficult moment — run through the three-step self-compassion break on your own.",
    "socratic_questioning":   "Choose one recurring negative thought this week. Write it down and work through: (1) Evidence for and against, (2) What you'd tell a friend, (3) Most realistic outcome.",
    "positive_data_log":      "Each evening this week, write ONE piece of evidence that challenges your main negative belief — no matter how small.",
    "behavioural_experiment": "Run the experiment we designed within 48 hours. Write your prediction first, then record what actually happened.",
    "decatastrophising":      "When you catch yourself catastrophising, write: (1) The catastrophic thought, (2) Most likely realistic outcome, (3) How you'd cope even if the worst happened.",
    "psychoeducation":        "Read one short article this week from Mind.org.uk or NHS mental health pages about what you're experiencing. Write 2–3 sentences about what resonated.",
}


# ══════════════════════════════════════════════════════════════
#  NODE 5 — CBT Response Generator
#  (handles both opening of interactive techniques and
#   single-turn technique delivery)
# ══════════════════════════════════════════════════════════════
OPENING_PROMPT = """\
You are a compassionate CBT therapist chatbot.

═══ MOOD & EMPATHY ═══════════════════════════════════════════
Mood           : {mood}
Tone to adopt  : {tone}
Opening posture: {opening}
Language guide : {language}
Avoid          : {avoid}
══════════════════════════════════════════════════════════════

═══ CBT KNOWLEDGE ════════════════════════════════════════════
{cbt_context}
══════════════════════════════════════════════════════════════

═══ USER MESSAGE ═════════════════════════════════════════════
{query}
══════════════════════════════════════════════════════════════

Write a warm, empathic opening response (3–4 sentences) that:
1. Validates the user's emotion specifically (no hollow phrases)
2. Normalises their experience briefly
3. Introduces the technique you're about to do together:
   "{technique_name}" — explain in ONE plain sentence why this
   technique fits what they're going through right now.
4. Then seamlessly lead into the first step below:

{first_step}

Do NOT add anything after the first step prompt — the user will answer it next.
Do NOT include the homework yet.
"""

SINGLE_TURN_PROMPT = """\
You are a compassionate, skilled CBT therapist chatbot.

═══ SESSION CONTEXT ══════════════════════════════════════════
Session #{session_number} | Turn #{turn_number}
Mood                   : {mood}
Selected Technique     : {technique}
Why this technique     : {technique_rationale}
Last Homework Assigned : {homework}
══════════════════════════════════════════════════════════════

═══ EMPATHY GUIDE ════════════════════════════════════════════
Tone    : {tone}
Opening : {opening}
Language: {language}
Avoid   : {avoid}
══════════════════════════════════════════════════════════════

═══ CBT KNOWLEDGE ════════════════════════════════════════════
{cbt_context}
══════════════════════════════════════════════════════════════

═══ MEDICAL CONTEXT ══════════════════════════════════════════
{medical_context}
══════════════════════════════════════════════════════════════

═══ HISTORY ══════════════════════════════════════════════════
{history}
══════════════════════════════════════════════════════════════

═══ USER MESSAGE ═════════════════════════════════════════════
{query}
══════════════════════════════════════════════════════════════

Structure your response:
1. EMPATHIC VALIDATION (3–4 sentences) — reflect their specific situation back
2. NORMALISATION (1–2 sentences)
3. CBT TECHNIQUE — deliver the full technique step by step
4. HOMEWORK — use exactly this task:
   {technique_homework}
   Frame warmly: "This week, I'd like to invite you to…"
5. PROFESSIONAL REMINDER — 2 warm sentences encouraging them to see a
   licensed therapist for deeper, personalised support.

Rules: Never diagnose. Never prescribe. No hollow phrases.
"""


def cbt_response_generator_node(state: CBTState) -> CBTState:
    mood      = state.get("mood", "neutral")
    technique = state.get("selected_technique", "")
    empathy   = MOOD_EMPATHY_PROFILES.get(mood, MOOD_EMPATHY_PROFILES["neutral"])
    cbt_ctx   = "\n\n---\n\n".join(state.get("cbt_context", [])) or "No CBT context retrieved."
    med_ctx   = "\n\n---\n\n".join(state.get("medical_context", [])) or "No medical context retrieved."

    is_interactive = technique in TECHNIQUE_STEPS

    if is_interactive:
        # Deliver opening + first step prompt
        steps      = TECHNIQUE_STEPS[technique]
        first_step = steps[0]["prompt"]

        prompt = OPENING_PROMPT.format(
            mood           = mood,
            tone           = empathy["tone"],
            opening        = empathy["opening"],
            language       = empathy["language"],
            avoid          = empathy["avoid"],
            cbt_context    = cbt_ctx,
            query          = state["user_query"],
            technique_name = TECHNIQUE_DESCRIPTIONS.get(technique, technique),
            first_step     = first_step,
        )
        new_state = dict(state)
        new_state["active_technique"]  = technique
        new_state["active_step_index"] = 0
        new_state["step_answers"]      = []

    else:
        # Single-turn delivery
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in state.get("chat_history", [])[-8:]
        ) or "First message."

        prompt = SINGLE_TURN_PROMPT.format(
            session_number      = state.get("session_number", 1),
            turn_number         = state.get("turn_number", 1),
            mood                = mood,
            technique           = TECHNIQUE_DESCRIPTIONS.get(technique, technique),
            technique_rationale = state.get("technique_rationale", ""),
            homework            = state.get("homework", "None assigned yet"),
            tone                = empathy["tone"],
            opening             = empathy["opening"],
            language            = empathy["language"],
            avoid               = empathy["avoid"],
            cbt_context         = cbt_ctx,
            medical_context     = med_ctx,
            history             = history_text,
            query               = state["user_query"],
            technique_homework  = TECHNIQUE_HOMEWORK.get(technique, "Reflect on today's session for 10 minutes."),
        )
        new_state = dict(state)

    response = LLM.invoke([HumanMessage(content=prompt)])
    new_state["final_response"] = response.content
    return new_state


# ══════════════════════════════════════════════════════════════
#  NODE — Interactive Technique Step Handler
#  Called instead of the full pipeline when a technique is
#  already mid-execution.
# ══════════════════════════════════════════════════════════════
STEP_REFLECT_PROMPT = """\
You are a warm CBT therapist guiding a user through {technique_name}.

The user just answered step "{step_label}":
User's answer: "{user_answer}"

Respond in 2–3 sentences:
1. Briefly acknowledge or gently reflect on what they shared (warm, specific)
2. Optionally name anything insightful in their answer
3. Then transition naturally into the next step:

{next_step_prompt}

Do NOT add anything after the next step prompt.
"""

CLOSING_REFLECT_PROMPT = """\
You are a warm CBT therapist who just completed {technique_name} with the user.

Here is a summary of all their answers throughout the exercise:
{all_answers}

Write a warm, 3–4 sentence closing that:
1. Acknowledges what they shared and worked through
2. Highlights one genuinely meaningful insight or shift you noticed
3. Reinforces that this takes practice — one step is enough

Then assign the homework below on a new line, introduced warmly:
"This week, I'd like to invite you to…"
{homework}

Then add 2 warm sentences encouraging them to also see a licensed therapist
for deeper, personalised support.
"""


def technique_step_node(state: CBTState) -> CBTState:
    """
    Handles one user turn while a technique is mid-execution.
    Advances the step index, reflects on the user's answer,
    and either prompts the next step or closes the technique.
    """
    technique   = state["active_technique"]
    step_index  = state["active_step_index"]
    steps       = TECHNIQUE_STEPS[technique]
    user_answer = state["user_query"]

    # Record this answer
    answers = state.get("step_answers", [])
    answers = answers + [{
        "step"  : steps[step_index]["label"],
        "answer": user_answer,
    }]

    next_index = step_index + 1
    is_last    = next_index >= len(steps)
    new_state  = dict(state)

    if is_last:
        # Build closing reflection
        all_answers_text = "\n".join(
            f"  [{a['step']}]: {a['answer']}" for a in answers
        )
        closing_prompt = CLOSING_REFLECT_PROMPT.format(
            technique_name = TECHNIQUE_DESCRIPTIONS.get(technique, technique),
            all_answers    = all_answers_text,
            homework       = TECHNIQUE_HOMEWORK.get(technique, "Reflect on this exercise."),
        )
        response = LLM.invoke([HumanMessage(content=closing_prompt)])
        new_state["final_response"]    = response.content
        new_state["active_technique"]  = ""    # clear — technique complete
        new_state["active_step_index"] = 0
        new_state["step_answers"]      = []
        print(f"[TechniqueStep] '{technique}' COMPLETE")

    else:
        next_step = steps[next_index]

        # Closing step with its own static prompt
        if next_step.get("is_closing") and next_step["prompt"]:
            # Static closing prompt (e.g. box breathing final check-in)
            response = LLM.invoke([HumanMessage(content=
                f"You are a warm CBT therapist. The user just completed a step "
                f"in {TECHNIQUE_DESCRIPTIONS.get(technique, technique)}.\n"
                f"Their last answer: '{user_answer}'\n"
                f"Acknowledge briefly (1–2 sentences), then deliver:\n\n"
                f"{next_step['prompt']}"
            )])
        elif steps[step_index].get("reflect"):
            # Reflect on current answer and transition to next step
            reflect_prompt = STEP_REFLECT_PROMPT.format(
                technique_name   = TECHNIQUE_DESCRIPTIONS.get(technique, technique),
                step_label       = steps[step_index]["label"],
                user_answer      = user_answer,
                next_step_prompt = next_step["prompt"],
            )
            response = LLM.invoke([HumanMessage(content=reflect_prompt)])
        else:
            # No reflection — just show next step
            response = LLM.invoke([HumanMessage(content=
                f"You are a warm CBT therapist. Briefly acknowledge the user "
                f"completed the last step (1 sentence), then deliver:\n\n"
                f"{next_step['prompt']}"
            )])

        new_state["final_response"]    = response.content
        new_state["active_step_index"] = next_index
        new_state["step_answers"]      = answers
        print(f"[TechniqueStep] '{technique}' step {next_index}/{len(steps)-1}")

    return new_state


# ══════════════════════════════════════════════════════════════
#  NODE — Session Tracker
# ══════════════════════════════════════════════════════════════
def _extract_homework(response_text: str) -> str:
    prompt = (
        "From the following CBT therapist response, extract ONLY the homework task "
        "assigned to the user. Return it as a single concise sentence. "
        "If no homework was assigned, return 'None'.\n\n"
        f"Response:\n{response_text}"
    )
    result = LLM.invoke([HumanMessage(content=prompt)])
    return result.content.strip()


def session_tracker_node(state: CBTState) -> CBTState:
    homework = _extract_homework(state["final_response"])

    turn_log = {
        "turn"              : state.get("turn_number", 1),
        "timestamp"         : datetime.datetime.now().isoformat(),
        "user_query"        : state["user_query"],
        "mood"              : state.get("mood", "neutral"),
        "technique"         : state.get("selected_technique") or state.get("active_technique", ""),
        "homework"          : homework,
        "satisfaction_score": state.get("satisfaction_score", -1),
        "response_preview"  : state["final_response"][:120] + "…",
    }

    session_log = state.get("session_log", [])
    session_log.append(turn_log)

    new_state = dict(state)
    new_state["session_log"] = session_log
    if homework and homework.lower() != "none":
        new_state["homework"] = homework
    print(f"[SessionTracker] Turn {state.get('turn_number', 1)} logged.")
    return new_state


# ══════════════════════════════════════════════════════════════
#  Graph Builder
# ══════════════════════════════════════════════════════════════
def _entry_router(state: CBTState) -> Literal["technique_step", "crisis_detection"]:
    """Route to interactive step handler if a technique is mid-execution."""
    if state.get("active_technique"):
        return "technique_step"
    return "crisis_detection"


def build_cbt_graph(medical_store: Chroma, cbt_store: Chroma):
    def rag_node(state: CBTState) -> CBTState:
        return rag_retriever_node(state, medical_store, cbt_store)

    graph = StateGraph(CBTState)

    # Add all nodes
    graph.add_node("entry_router",           lambda s: s)   # pass-through router node
    graph.add_node("technique_step",         technique_step_node)
    graph.add_node("crisis_detection",       crisis_detection_node)
    graph.add_node("mood_detector",          mood_detector_node)
    graph.add_node("cbt_technique_selector", cbt_technique_selector_node)
    graph.add_node("rag_retriever",          rag_node)
    graph.add_node("cbt_response_generator", cbt_response_generator_node)
    graph.add_node("session_tracker",        session_tracker_node)

    # Entry point routes based on whether a technique is active
    graph.set_entry_point("entry_router")
    graph.add_conditional_edges(
        "entry_router",
        _entry_router,
        {
            "technique_step"  : "technique_step",
            "crisis_detection": "crisis_detection",
        },
    )

    # Interactive technique path
    graph.add_edge("technique_step", "session_tracker")

    # Normal pipeline path
    graph.add_conditional_edges(
        "crisis_detection", crisis_router,
        {"__end__": END, "mood_detector": "mood_detector"},
    )
    graph.add_edge("mood_detector",          "cbt_technique_selector")
    graph.add_edge("cbt_technique_selector", "rag_retriever")
    graph.add_edge("rag_retriever",          "cbt_response_generator")
    graph.add_edge("cbt_response_generator", "session_tracker")
    graph.add_edge("session_tracker",        END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════
#  Chat Session
# ══════════════════════════════════════════════════════════════
class CBTChatSession:
    def __init__(self, persist_path: str = "medical_chroma", session_number: int = 1):
        print(f"Loading RAG stores… (strategy='{RETRIEVAL_STRATEGY}', model='{LLM_MODEL}')")
        stores = load_medical_rag_store(persist_path)
        if stores is None:
            raise RuntimeError("RAG stores not found. Run `python medical_rag.py` first.")
        self.graph               = build_cbt_graph(stores["medical"], stores["cbt"])
        self.chat_history        : list[dict] = []
        self.session_log         : list[dict] = []
        self.homework            : str  = "None assigned yet"
        self.session_number      : int  = session_number
        self.turn_number         : int  = 0
        self.satisfaction_scores : list[int] = []
        # Interactive technique state (persisted across turns)
        self.active_technique    : str  = ""
        self.active_step_index   : int  = 0
        self.step_answers        : list[dict] = []

    def chat(self, user_message: str) -> str:
        self.turn_number += 1

        initial_state: CBTState = {
            "user_query":           user_message,
            "crisis_detected":      False,
            "mood":                 "neutral",
            "selected_technique":   "",
            "technique_rationale":  "",
            "medical_context":      [],
            "cbt_context":          [],
            "final_response":       "",
            "session_log":          self.session_log.copy(),
            "session_number":       self.session_number,
            "turn_number":          self.turn_number,
            "homework":             self.homework,
            "chat_history":         self.chat_history.copy(),
            "satisfaction_score":   -1,
            # Pass interactive technique state into graph
            "active_technique":     self.active_technique,
            "active_step_index":    self.active_step_index,
            "step_answers":         self.step_answers.copy(),
        }

        final_state = self.graph.invoke(initial_state)
        response    = final_state["final_response"]

        self.chat_history.append({"role": "user",      "content": user_message})
        self.chat_history.append({"role": "assistant",  "content": response})
        self.session_log       = final_state.get("session_log",       self.session_log)
        self.homework          = final_state.get("homework",          self.homework)
        # Persist interactive technique state for next turn
        self.active_technique  = final_state.get("active_technique",  "")
        self.active_step_index = final_state.get("active_step_index", 0)
        self.step_answers      = final_state.get("step_answers",      [])

        return response

    def record_satisfaction(self, score: int) -> None:
        if not self.session_log:
            return
        score = max(0, min(10, score))
        self.session_log[-1]["satisfaction_score"] = score
        self.satisfaction_scores.append(score)
        if score <= 3:
            label = "😔  Low — we'll aim to do better"
        elif score <= 6:
            label = "😐  Moderate — room to improve"
        else:
            label = "😊  Good — glad this was helpful"
        print(f"[Feedback] Score {score}/10  {label}")

    def get_session_summary(self) -> str:
        if not self.session_log:
            return "No turns completed yet."
        moods      = [t["mood"]      for t in self.session_log]
        techniques = [t.get("technique", "") for t in self.session_log]
        rated_scores = [
            t["satisfaction_score"]
            for t in self.session_log
            if t.get("satisfaction_score", -1) >= 0
        ]
        if rated_scores:
            avg  = sum(rated_scores) / len(rated_scores)
            low  = sum(1 for s in rated_scores if s <= 3)
            mid  = sum(1 for s in rated_scores if 4 <= s <= 6)
            high = sum(1 for s in rated_scores if s >= 7)
            sat_line = (
                f"  Satisfaction avg : {avg:.1f}/10  "
                f"(😔 {low} | 😐 {mid} | 😊 {high})\n"
                f"  Scores by turn   : {rated_scores}\n"
            )
        else:
            sat_line = "  Satisfaction     : no ratings given\n"

        return (
            f"\n{'═'*55}\n"
            f"  SESSION {self.session_number} SUMMARY  |  {self.turn_number} turn(s)\n"
            f"  Strategy : {RETRIEVAL_STRATEGY}  |  Model: {LLM_MODEL}\n"
            f"{'═'*55}\n"
            f"  Moods detected  : {', '.join(moods)}\n"
            f"  Techniques used : {', '.join(t for t in set(techniques) if t)}\n"
            f"  Last homework   : {self.homework}\n"
            f"{sat_line}"
            f"{'═'*55}\n"
        )


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 55)
    print("  CBT Therapeutic Chatbot  —  LangGraph + RAG")
    print(f"  Strategy : {RETRIEVAL_STRATEGY}  |  Model : {LLM_MODEL}")
    print("  Commands: 'summary' | 'history' | 'exit'")
    print("═" * 55)

    session = CBTChatSession()
    print(
        "\nBot: Hello, and welcome. I'm here to support you using "
        "Cognitive Behavioural Therapy (CBT) principles.\n"
        "     How are you feeling today, and what's on your mind?\n"
    )

    while True:
        try:
            # Show step indicator if inside an interactive technique
            if session.active_technique:
                steps      = TECHNIQUE_STEPS[session.active_technique]
                step_label = steps[session.active_step_index]["label"]
                technique_name = TECHNIQUE_DESCRIPTIONS.get(
                    session.active_technique, session.active_technique
                )
                prompt_prefix = (
                    f"[{technique_name} — step "
                    f"{session.active_step_index + 1}/{len(steps)}]"
                )
                user_input = input(f"\n{prompt_prefix}\nYou: ").strip()
            else:
                user_input = input("\nYou: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nBot: Take good care of yourself. Goodbye! 💙")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Bot: It was good to talk with you. Take care! 💙")
            print(session.get_session_summary())
            break
        if user_input.lower() == "summary":
            print(session.get_session_summary())
            continue
        if user_input.lower() == "history":
            for turn in session.session_log:
                print(f"\n[Turn {turn['turn']}] Mood: {turn['mood']} | "
                      f"Technique: {turn.get('technique','')}")
            continue

        response = session.chat(user_input)
        print(f"\nBot:\n{response}\n")

        # Satisfaction feedback — only ask when technique is NOT mid-step
        # (avoids interrupting the exercise flow)
        if not session.active_technique:
            print("─" * 45)
            print("  How satisfied are you with this response?")
            print("  0 = Not helpful   10 = Extremely helpful")
            print("  (Press Enter to skip)")
            print("─" * 45)
            try:
                raw = input("  Your rating (0-10): ").strip()
                if raw == "":
                    print("  [Skipped]")
                elif raw.isdigit() and 0 <= int(raw) <= 10:
                    session.record_satisfaction(int(raw))
                else:
                    print("  [Invalid — skipped]")
            except (EOFError, KeyboardInterrupt):
                print()
            print()
