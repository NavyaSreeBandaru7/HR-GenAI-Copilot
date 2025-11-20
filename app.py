import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import pandas as pd

# -------------------------------------------------
# 1) GLOBAL SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="HR GenAI Copilot",
    page_icon="🧠",
    layout="wide",
)

# -------------------------------------------------
# 2) LOAD MODELS (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    sent_model = pipeline("sentiment-analysis")
    return embed_model, sent_model

embed_model, sentiment_model = load_models()

# -------------------------------------------------
# 3) DEMO DATA
# -------------------------------------------------
DEMO_POLICIES = [
    {
        "title": "Paid Time Off Policy",
        "text": """Full-time employees accrue paid time off (PTO) based on their tenure with the company. 
PTO can be used for vacation, personal days, or short-term illness. 
Employees should request PTO at least two weeks in advance whenever possible."""
    },
    {
        "title": "Work From Home Policy",
        "text": """Employees may work from home up to two days per week with manager approval. 
All remote workdays should be recorded in the time tracking system. 
Employees are expected to be available online during core business hours."""
    },
    {
        "title": "Employee Benefits Overview",
        "text": """The company offers health insurance, retirement savings plans, and an employee assistance program. 
Eligible employees may also receive performance bonuses and learning & development reimbursements."""
    },
    {
        "title": "Code of Conduct",
        "text": """All employees are expected to maintain professional behavior, respect colleagues, and comply with company policies. 
Harassment, discrimination, or unethical behavior will not be tolerated."""
    }
]

DEMO_FEEDBACK = [
    "I really enjoy working with my team and my manager supports my growth.",
    "The workload is too high and I feel burned out most of the time.",
    "The new flexible working hours policy has improved my work-life balance.",
    "I am thinking about leaving because my contributions are not recognized.",
    "The office environment is great, but the salary is not competitive.",
    "I have no clear career path here and it makes me demotivated.",
    "The training sessions were useful and well-organized."
]

# -------------------------------------------------
# 4) HELPERS TO PREPARE EMBEDDINGS
# -------------------------------------------------
@st.cache_data
def embed_policies(policies):
    texts = [p["text"] for p in policies]
    return embed_model.encode(texts, convert_to_tensor=True)

@st.cache_data
def embed_feedback(feedback_list):
    return embed_model.encode(feedback_list, convert_to_tensor=True)

# -------------------------------------------------
# 5) INTENT + RAG + FEEDBACK HELPERS
# -------------------------------------------------
def detect_intent(query: str) -> str:
    q = query.lower()
    policy_keywords = [
        "policy", "pto", "leave", "vacation", "remote",
        "work from home", "benefits", "health insurance", "maternity",
        "holiday", "sick leave"
    ]
    feedback_keywords = [
        "feedback", "comments", "how employees feel", "why employees",
        "burnout", "leaving", "demotivated", "unhappy", "engagement",
        "satisfaction", "culture", "morale"
    ]
    if any(k in q for k in policy_keywords):
        return "policy"
    if any(k in q for k in feedback_keywords):
        return "feedback"
    return "unknown"


def get_top_policy(query, policies, policy_embeddings):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, policy_embeddings)[0]
    results = list(zip(policies, cos_scores.tolist()))
    best_policy, best_score = sorted(results, key=lambda x: x[1], reverse=True)[0]
    return best_policy, best_score


def answer_hr_policy_question(query, policies, policy_embeddings):
    policy, score = get_top_policy(query, policies, policy_embeddings)
    sentences = [s.strip() for s in policy["text"].split(".") if s.strip()]
    summary = ". ".join(sentences[:2]) + "."
    answer = (
        f"**Most relevant policy:** **{policy['title']}**  \n\n"
        f"{summary}  \n\n"
        f"_Match score: {score:.3f}_"
    )
    return answer


def find_negative_feedback(query, feedback_list, feedback_embeddings, sim_threshold=0.2, top_k=3):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, feedback_embeddings)[0].tolist()

    results = []
    for text, score in zip(feedback_list, cos_scores):
        sent = sentiment_model(text)[0]
        if sent["label"] == "NEGATIVE" and score >= sim_threshold:
            results.append((text, score, sent["score"]))

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    if not results_sorted:
        return "I checked employee feedback but did not find strongly negative comments on this topic."

    lines = ["Here are the most relevant **negative feedback** comments:\n"]
    for text, sim, conf in results_sorted[:top_k]:
        lines.append(
            f"- {text}  \n"
            f"  _(similarity: {sim:.3f}, sentiment confidence: {conf:.3f})_"
        )
    return "\n".join(lines)


def hr_copilot(query, policies, policy_embeddings, feedback_list, feedback_embeddings):
    intent = detect_intent(query)

    if intent == "policy":
        return answer_hr_policy_question(query, policies, policy_embeddings)

    elif intent == "feedback":
        return find_negative_feedback(query, feedback_list, feedback_embeddings)

    else:
        return (
            "Right now I can help with:\n\n"
            "- HR **policies** (PTO, remote work, benefits, code of conduct)\n"
            "- Employee **feedback analysis** (negative / at-risk comments)\n\n"
            "Try asking, for example:\n"
            "- `What is our PTO policy?`\n"
            "- `Can I work remotely?`\n"
            "- `Why are employees leaving the company?`\n"
            "- `Show me feedback about burnout and workload.`"
        )

# -------------------------------------------------
# 6) SIDEBAR – DATA & MODE SELECTION
# -------------------------------------------------
st.sidebar.title("⚙️ Settings")

data_mode = st.sidebar.radio(
    "Data source:",
    ["Use demo HR data", "Upload my own feedback (CSV)"],
    help="You can start with demo data, or upload your own employee feedback later."
)

# Policy data (for now, keep demo; later you can extend with uploads)
policies = DEMO_POLICIES
policy_embeddings = embed_policies(policies)

# Feedback data: demo or user CSV
feedback_list = DEMO_FEEDBACK

uploaded_feedback_df = None
if data_mode == "Upload my own feedback (CSV)":
    st.sidebar.markdown("Upload a CSV file with a **'feedback'** column.")
    uploaded_file = st.sidebar.file_uploader("Upload feedback CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "feedback" in df.columns:
                uploaded_feedback_df = df
                feedback_list = df["feedback"].dropna().astype(str).tolist()
                st.sidebar.success(f"Loaded {len(feedback_list)} feedback comments.")
            else:
                st.sidebar.error("CSV must contain a 'feedback' column.")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

feedback_embeddings = embed_feedback(feedback_list)

# -------------------------------------------------
# 7) MAIN LAYOUT – TABS
# -------------------------------------------------
st.title("🧠 HR GenAI Copilot (Prototype)")
st.write(
    "This app is a prototype HR assistant that can answer **policy questions** "
    "and surface **at-risk employee feedback** using embeddings and transformer models."
)

tab1, tab2, tab3 = st.tabs(["💬 Ask Copilot", "📊 Feedback Explorer", "ℹ️ About"])

# ---------------- TAB 1: ASK COPILOT ----------------
with tab1:
    st.subheader("💬 Ask the HR GenAI Copilot")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_query = st.text_area(
            "Type your question here:",
            placeholder="Examples: What is our PTO policy? • Why are employees leaving the company? • Show me feedback about workload and burnout.",
            height=100,
        )

    with col2:
        st.markdown("**Examples:**")
        st.code("What is our PTO policy?", language="markdown")
        st.code("Can I work remotely?", language="markdown")
        st.code("Why are employees leaving the company?", language="markdown")
        st.code("Show me feedback about burnout and workload.", language="markdown")

    if st.button("Ask Copilot", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                reply = hr_copilot(user_query, policies, policy_embeddings, feedback_list, feedback_embeddings)
            st.markdown("### 🧠 Copilot Response")
            st.markdown(reply)

# ---------------- TAB 2: FEEDBACK EXPLORER ----------------
with tab2:
    st.subheader("📊 Employee Feedback Explorer")

    st.write(
        "Here you can quickly **scan all feedback comments**, with sentiment labels "
        "and search by a topic using embeddings."
    )

    # Show feedback table with sentiment
    if st.checkbox("Show feedback with sentiment labels"):
        sentiments = [sentiment_model(text)[0] for text in feedback_list]
        df_view = pd.DataFrame({
            "feedback": feedback_list,
            "sentiment": [s["label"] for s in sentiments],
            "sentiment_score": [s["score"] for s in sentiments],
        })
        st.dataframe(df_view)

    st.markdown("---")
    st.markdown("#### 🔎 Search negative feedback by topic")

    topic_query = st.text_input(
        "Topic to search (e.g., 'workload', 'salary', 'career growth'):",
        ""
    )

    if st.button("Find negative feedback"):
        if not topic_query.strip():
            st.warning("Please type a topic to search.")
        else:
            with st.spinner("Searching..."):
                result_text = find_negative_feedback(
                    query=topic_query,
                    feedback_list=feedback_list,
                    feedback_embeddings=feedback_embeddings,
                    sim_threshold=0.2,
                    top_k=5,
                )
            st.markdown(result_text)

# ---------------- TAB 3: ABOUT ----------------
with tab3:
    st.subheader("ℹ️ About this prototype")
    st.write(
        """
        **HR GenAI Copilot** is a learning and portfolio project that demonstrates how 
        large language model tooling can support HR use cases using:
        
        - 🔍 **Semantic search / RAG-style retrieval** over HR policies  
        - 🧪 **Sentiment analysis** over employee feedback  
        - 🧠 A simple **agent-like controller** to decide whether to search policies or feedback
        
        ### How it works (high level)
        - We use `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`) to represent text as vectors.
        - For policy questions, we compute similarity between the user query and policy texts, 
          then return the best-matching policy with a summary.
        - For feedback analysis, we combine similarity + sentiment (only NEGATIVE) to highlight
          at-risk comments on a specific topic (e.g., workload, salary, growth).
        - A small intent detector routes questions to either the policy module or feedback module.
        
        ### Possible extensions
        - Ingest real HR policy PDFs and parse them into sections.
        - Ingest real (anonymized) employee feedback from CSV / database.
        - Use a hosted LLM (e.g., OpenAI or other) to rewrite answers in more natural language.
        - Add filters by team, location, or time period for feedback analytics.
        """
    )
