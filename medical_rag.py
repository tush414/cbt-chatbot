##################################################################
#  medical_rag.py — Medical + CBT RAG Pipeline                  #
#                                                               #
#  Two ChromaDB collections:                                    #
#    1. medical_general   — diseases, symptoms, treatments      #
#    2. cbt_knowledge     — CBT techniques, distortions,        #
#                           therapy frameworks, mental health   #
#                                                               #
#  Single configurable retrieval strategy:                      #
#    Set RETRIEVAL_STRATEGY in cbt_chatbot.py or pass           #
#    strategy= to retrieve() directly.                          #
#                                                               #
#  Run once:  python medical_rag.py                             #
##################################################################

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ──────────────────────────────────────────────────────────────
#  Retrieval strategy type alias
# ──────────────────────────────────────────────────────────────
RetrievalStrategy = Literal["similarity", "mmr", "hybrid"]

# ──────────────────────────────────────────────────────────────
#  Collection names
# ──────────────────────────────────────────────────────────────
MEDICAL_COLLECTION = "medical_general"
CBT_COLLECTION     = "cbt_knowledge"


def get_embeddings() -> OpenAIEmbeddings:
    """Return OpenAI embeddings (downloads once, then cached)."""
    return OpenAIEmbeddings()


# ──────────────────────────────────────────────────────────────
#  SOURCE URLS
# ──────────────────────────────────────────────────────────────
MEDICAL_URLS = [
    "https://www.mayoclinic.org/diseases-conditions",
    "https://www.cdc.gov/diseasesconditions/index.html",
    "https://www.who.int/health-topics",
    "https://medlineplus.gov/healthtopics.html",
    "https://www.nhs.uk/conditions/",
    "https://my.clevelandclinic.org/health/diseases",
    "https://www.health.harvard.edu/diseases-and-conditions",
    "https://www.mountsinai.org/health-library",
]

CBT_URLS = [
    "https://www.apa.org/ptsd-guideline/patients-and-families/cognitive-behavioral",
    "https://www.mind.org.uk/information-support/drugs-and-treatments/cognitive-behavioural-therapy-cbt/",
    "https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/cognitive-behavioural-therapy-cbt/overview/",
    "https://www.mind.org.uk/information-support/types-of-mental-health-problems/depression/",
    "https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/",
    "https://www.mind.org.uk/information-support/types-of-mental-health-problems/stress/",
    "https://www.mind.org.uk/information-support/types-of-mental-health-problems/grief/",
    "https://positivepsychology.com/cbt-cognitive-behavioral-therapy-techniques-worksheets/",
    "https://positivepsychology.com/cognitive-distortions/",
    "https://www.verywellmind.com/what-is-cognitive-behavior-therapy-2795747",
    "https://www.verywellmind.com/cognitive-distortions-and-depression-1065378",
    "https://www.verywellmind.com/behavioral-activation-for-depression-2797557",
    "https://www.verywellmind.com/how-to-identify-and-challenge-automatic-thoughts-3024215",
    "https://www.therapistaid.com/therapy-guide/cbt",
]

# ──────────────────────────────────────────────────────────────
#  Built-in CBT knowledge  (fallback when URLs are unreachable)
# ──────────────────────────────────────────────────────────────
CBT_BUILTIN_DOCS = [
    {
        "content": """
CBT CORE MODEL — The Cognitive Triangle
The cognitive triangle describes the interconnection between:
- Thoughts (cognitions): What we tell ourselves about a situation
- Feelings (emotions): How we feel as a result of those thoughts
- Behaviours: What we do in response to those feelings

CBT works by identifying and changing unhelpful thought patterns, which in turn
changes emotions and behaviours. The key principle: it is not the situation itself
but our INTERPRETATION of it that determines how we feel.
        """,
        "metadata": {"source": "cbt_builtin", "topic": "core_model", "type": "cbt_theory"}
    },
    {
        "content": """
COGNITIVE DISTORTIONS — Common Thinking Errors
1. All-or-Nothing Thinking: Seeing things in black and white, with no middle ground.
   Example: "If I'm not perfect, I'm a complete failure."

2. Catastrophising: Blowing things out of proportion, expecting the worst.
   Example: "I made a mistake at work — I'll definitely get fired."

3. Mind Reading: Assuming you know what others are thinking (usually negatively).
   Example: "They didn't reply — they must be angry with me."

4. Fortune Telling: Predicting the future negatively.
   Example: "I know this interview will go terribly."

5. Emotional Reasoning: Believing something is true because it FEELS true.
   Example: "I feel stupid, therefore I am stupid."

6. Should Statements: Rigid rules about how you/others must behave.
   Example: "I should always be productive. I shouldn't feel sad."

7. Personalisation: Blaming yourself for things outside your control.
   Example: "My friend is upset — it must be something I did."

8. Overgeneralisation: Drawing sweeping conclusions from a single event.
   Example: "I failed once — I always fail at everything."

9. Mental Filter: Focusing only on the negative and ignoring positives.
   Example: Receiving 9 compliments and 1 criticism and dwelling on the criticism.

10. Labelling: Attaching a global negative label to yourself or others.
    Example: "I'm a loser" instead of "I made a mistake."

11. Magnification/Minimisation: Magnifying negatives, minimising positives.
12. Jumping to Conclusions: Making negative interpretations without evidence.
        """,
        "metadata": {"source": "cbt_builtin", "topic": "cognitive_distortions", "type": "cbt_technique"}
    },
    {
        "content": """
THOUGHT RECORD — Core CBT Tool (7-Column)
A thought record helps identify and challenge automatic negative thoughts.

Step 1 — SITUATION: What happened? Where were you? Who was there?
Step 2 — EMOTIONS: What did you feel? Rate intensity 0-100%.
Step 3 — AUTOMATIC THOUGHT: What went through your mind? What images came?
Step 4 — EVIDENCE FOR: What facts support this thought?
Step 5 — EVIDENCE AGAINST: What facts contradict this thought? What would you tell a friend?
Step 6 — BALANCED THOUGHT: Write a more realistic, balanced version of the thought.
Step 7 — OUTCOME: Re-rate emotion intensity. How do you feel now?

The goal is NOT to think positively — it is to think ACCURATELY and FLEXIBLY.
        """,
        "metadata": {"source": "cbt_builtin", "topic": "thought_record", "type": "cbt_technique"}
    },
    {
        "content": """
BEHAVIOURAL ACTIVATION — CBT for Depression
Behavioural Activation (BA) is one of the most evidence-based treatments for depression.

Core principle: Depression creates a vicious cycle — low mood → reduced activity →
less pleasure/achievement → lower mood. BA breaks this cycle by SCHEDULING activities.

Activity types to balance:
- Pleasure activities: Things you enjoy (even 5% enjoyment counts)
- Achievement activities: Tasks that give a sense of accomplishment
- Social activities: Connecting with others

Steps:
1. Monitor current activities and mood (activity diary)
2. Identify avoided activities
3. Schedule small, manageable activities starting with easy wins
4. Gradually increase difficulty and frequency
5. Track mood changes linked to activity

Key rule: Don't wait to FEEL motivated — act first, motivation follows action.
        """,
        "metadata": {"source": "cbt_builtin", "topic": "behavioural_activation", "type": "cbt_technique"}
    },
    {
        "content": """
GROUNDING TECHNIQUES — CBT for Anxiety
Grounding techniques anchor you in the present moment during anxiety or panic.

5-4-3-2-1 Sensory Grounding:
- Name 5 things you can SEE
- Name 4 things you can TOUCH (and feel their texture)
- Name 3 things you can HEAR
- Name 2 things you can SMELL
- Name 1 thing you can TASTE

Box Breathing (4-4-4-4):
- Inhale for 4 counts
- Hold for 4 counts
- Exhale for 4 counts
- Hold for 4 counts
- Repeat 4 times

Progressive Muscle Relaxation:
- Tense each muscle group for 5 seconds, then release
- Work from feet upward to face
- Notice the contrast between tension and relaxation

Cognitive grounding: Repeat "I am safe right now. This feeling will pass."
        """,
        "metadata": {"source": "cbt_builtin", "topic": "grounding_anxiety", "type": "cbt_technique"}
    },
    {
        "content": """
SOCRATIC QUESTIONING — The CBT Therapist's Core Tool
Socratic questioning guides clients to examine their own thinking rather than
being told what to think. It builds insight and self-efficacy.

Key Socratic questions for challenging thoughts:
- "What evidence do you have for this belief?"
- "What evidence do you have against it?"
- "What would you tell a close friend in this situation?"
- "What is the WORST that could realistically happen? How would you cope?"
- "What is the BEST that could happen? What is MOST LIKELY?"
- "Are you confusing a thought with a fact?"
- "Are you thinking in all-or-nothing terms?"
- "In 5 years, how important will this seem?"
- "Is there another way to look at this situation?"
- "What are the advantages and disadvantages of thinking this way?"
        """,
        "metadata": {"source": "cbt_builtin", "topic": "socratic_questioning", "type": "cbt_technique"}
    },
    {
        "content": """
ANXIETY — CBT Conceptualisation
Anxiety is a normal human emotion triggered by perceived threat. It becomes
problematic when the threat is overestimated or coping ability is underestimated.

CBT model of anxiety:
Trigger → Threat appraisal → Physical symptoms (heart racing, shallow breathing)
→ Safety behaviours (avoidance) → Maintained anxiety (never learns threat is manageable)

Maintenance cycles:
- Avoidance: Short-term relief, long-term worsening
- Safety behaviours: Prevent disconfirmation of feared outcome
- Hypervigilance: Scanning for threat keeps anxiety high
- Reassurance seeking: Temporary relief, dependency

CBT interventions:
- Psychoeducation about anxiety (normalise the experience)
- Cognitive restructuring (challenge overestimates of threat)
- Graded exposure (gradually face feared situations)
- Interoceptive exposure (for panic — face physical sensations)
- Dropping safety behaviours
        """,
        "metadata": {"source": "cbt_builtin", "topic": "anxiety_cbt", "type": "cbt_theory"}
    },
    {
        "content": """
DEPRESSION — CBT Conceptualisation
Depression involves a negative cognitive triad (Beck, 1967):
1. Negative view of SELF: "I am worthless, defective, inadequate"
2. Negative view of the WORLD: "Everything is hopeless, the world is unfair"
3. Negative view of the FUTURE: "Nothing will ever get better"

Maintaining factors:
- Withdrawal and inactivity (feeds hopelessness)
- Rumination (dwelling on problems without solving them)
- Self-criticism (harsh internal dialogue)
- All-or-nothing thinking
- Reduced positive reinforcement

CBT interventions:
- Behavioural Activation (counteracts withdrawal)
- Thought records (challenges negative cognitive triad)
- Activity scheduling with pleasure and achievement
- Self-compassion exercises
- Problem-solving therapy
- Relapse prevention planning
        """,
        "metadata": {"source": "cbt_builtin", "topic": "depression_cbt", "type": "cbt_theory"}
    },
    {
        "content": """
CBT SESSION STRUCTURE
A standard CBT session follows this structure:

1. CHECK-IN (5 min): Mood rating (0-10), any crisis concerns
2. AGENDA SETTING (3 min): Agree on 1-2 topics to focus on today
3. HOMEWORK REVIEW (10 min): Review practice from last session, troubleshoot
4. MAIN WORK (25 min): Core CBT work (thought record, exposure, problem-solving)
5. SUMMARY (5 min): Therapist and client both summarise key insights
6. NEW HOMEWORK (5 min): Agree on specific, manageable practice task
7. FEEDBACK (2 min): How was this session? What was helpful/unhelpful?

Key principles:
- Collaborative empiricism: Therapist and client as co-investigators
- Guided discovery: Questions over advice
- Structure provides safety and predictability
- Homework bridges sessions (practice = progress)
        """,
        "metadata": {"source": "cbt_builtin", "topic": "session_structure", "type": "cbt_theory"}
    },
    {
        "content": """
PROBLEM-SOLVING THERAPY — CBT Technique
Useful when practical problems contribute to distress.

6 Steps:
1. DEFINE the problem specifically (not "my life is a mess" but "I have £500 debt")
2. GENERATE solutions — brainstorm ALL options without judging (quantity over quality)
3. EVALUATE each option — pros, cons, feasibility
4. CHOOSE the best option or combination
5. IMPLEMENT — break into small action steps with timeline
6. REVIEW — did it work? If not, revisit step 2.

Common errors to avoid:
- Defining problem too vaguely
- Skipping straight to solutions without defining the real problem
- Giving up after one failed attempt
- Treating unsolvable problems as solvable (acceptance may be needed instead)
        """,
        "metadata": {"source": "cbt_builtin", "topic": "problem_solving", "type": "cbt_technique"}
    },
    {
        "content": """
SELF-COMPASSION IN CBT
Self-compassion (Kristin Neff) has three components:
1. Self-kindness: Treating yourself with the same warmth you'd offer a friend
2. Common humanity: Recognising suffering is a shared human experience
3. Mindfulness: Observing painful feelings without over-identification

The self-compassion break (for difficult moments):
- "This is a moment of suffering" (mindfulness — acknowledge)
- "Suffering is part of life" (common humanity — normalise)
- "May I be kind to myself" (self-kindness — soothe)
        """,
        "metadata": {"source": "cbt_builtin", "topic": "self_compassion", "type": "cbt_technique"}
    },
    {
        "content": """
HOMEWORK IN CBT — Why Practice Matters
CBT homework is not optional — it is where real change happens.
The session plants the seed; homework is where it grows.

Types of CBT homework:
- Thought diaries: Recording automatic thoughts between sessions
- Behavioural experiments: Testing feared predictions in real life
- Activity scheduling: Booking pleasurable/meaningful activities
- Exposure tasks: Gradually facing feared situations
- Reading / psychoeducation: Learning about conditions and CBT
- Relaxation practice: Daily grounding or breathing exercises
- Problem-solving worksheets

Tips for effective homework:
- Keep it small and specific (10 min thought diary, not "think positively")
- Write it down — vague intentions rarely happen
- Identify obstacles in advance and problem-solve them
- Review at the start of the NEXT session (shows it matters)
- If it wasn't done, explore what got in the way non-judgementally
        """,
        "metadata": {"source": "cbt_builtin", "topic": "homework", "type": "cbt_technique"}
    },
]


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _load_urls(urls: list[str]) -> list[Document]:
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            docs.extend(loaded)
            print(f"  ✓  {url}  ({len(loaded)} doc(s))")
        except Exception as exc:
            print(f"  ✗  {url}: {exc}")
    return docs


def _chunk(
    docs: list[Document],
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ──────────────────────────────────────────────────────────────
#  Build
# ──────────────────────────────────────────────────────────────

def build_medical_rag_store(persist_path: str = "medical_chroma") -> dict[str, Chroma]:
    """
    Build TWO ChromaDB collections:
      - medical_general : general medical knowledge from trusted health sites
      - cbt_knowledge   : CBT techniques + mental health content (URLs + built-ins)

    Returns dict {"medical": Chroma, "cbt": Chroma}
    """
    os.makedirs(persist_path, exist_ok=True)
    embeddings = get_embeddings()

    # ── Collection 1: Medical General ──────────────────────────
    print("\n[1/2] Building medical_general collection…")
    medical_docs = _load_urls(MEDICAL_URLS)
    print(f"  Raw docs: {len(medical_docs)}")
    medical_chunks = _chunk(medical_docs)
    print(f"  Chunks  : {len(medical_chunks)}")

    medical_store = Chroma.from_documents(
        documents=medical_chunks,
        embedding=embeddings,
        collection_name=MEDICAL_COLLECTION,
        persist_directory=persist_path,
    )
    medical_store.persist()
    print(f"  ✓ medical_general saved → {persist_path}")

    # ── Collection 2: CBT Knowledge ────────────────────────────
    print("\n[2/2] Building cbt_knowledge collection…")
    cbt_url_docs = _load_urls(CBT_URLS)
    print(f"  Adding {len(CBT_BUILTIN_DOCS)} built-in CBT knowledge documents…")

    builtin_docs = [
        Document(page_content=d["content"].strip(), metadata=d["metadata"])
        for d in CBT_BUILTIN_DOCS
    ]
    cbt_url_chunks = _chunk(cbt_url_docs, chunk_size=1000, overlap=100)
    all_cbt_chunks = cbt_url_chunks + builtin_docs
    print(f"  Raw docs: {len(cbt_url_docs + builtin_docs)}")
    print(f"  Chunks  : {len(all_cbt_chunks)}")

    cbt_store = Chroma.from_documents(
        documents=all_cbt_chunks,
        embedding=embeddings,
        collection_name=CBT_COLLECTION,
        persist_directory=persist_path,
    )
    cbt_store.persist()
    print(f"  ✓ cbt_knowledge saved → {persist_path}")

    print("\n✅ Both collections built successfully.")
    return {"medical": medical_store, "cbt": cbt_store}


# ──────────────────────────────────────────────────────────────
#  Load
# ──────────────────────────────────────────────────────────────

def load_medical_rag_store(
    persist_path: str = "medical_chroma",
) -> dict[str, Chroma] | None:
    """
    Load both ChromaDB collections from *persist_path*.
    Returns dict {"medical": Chroma, "cbt": Chroma} or None on failure.
    """
    embeddings = get_embeddings()
    try:
        medical_store = Chroma(
            collection_name=MEDICAL_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_path,
        )
        cbt_store = Chroma(
            collection_name=CBT_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_path,
        )
        print("✓ Medical RAG store loaded.")
        print("✓ CBT knowledge store loaded.")
        return {"medical": medical_store, "cbt": cbt_store}
    except Exception as exc:
        print(f"Error loading RAG stores: {exc}")
        return None


# ──────────────────────────────────────────────────────────────
#  Single-strategy retriever
# ──────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    vectorstore: Chroma,
    strategy: RetrievalStrategy = "similarity",
    k: int = 4,
    label: str = "",
) -> list[Document]:
    """
    Retrieve documents using ONE configurable strategy.

    Parameters
    ----------
    query       : Search query string.
    vectorstore : ChromaDB Chroma instance to query.
    strategy    : One of:
                    "similarity" — cosine k-NN (most relevant)
                    "mmr"        — Maximal Marginal Relevance (diversity-aware)
                    "hybrid"     — keyword-stripped cosine search (exact term recall)
    k           : Number of documents to retrieve.
    label       : Optional label for console logging.

    Returns
    -------
    List of LangChain Document objects.
    """
    if strategy == "similarity":
        docs = vectorstore.similarity_search(query, k=k)

    elif strategy == "mmr":
        docs = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=k * 5, lambda_mult=0.55
        )

    elif strategy == "hybrid":
        # Strip injected context tags to improve keyword precision
        keyword_query = query.split("[")[0].strip()
        docs = vectorstore.similarity_search(keyword_query, k=k)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from: 'similarity', 'mmr', 'hybrid'."
        )

    if label:
        print(f"  [{label}] strategy={strategy} → {len(docs)} doc(s)")

    return docs


if __name__ == "__main__":
    build_medical_rag_store()