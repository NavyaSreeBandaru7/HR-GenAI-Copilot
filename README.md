# 🧠 HR GenAI Copilot (Prototype)

An HR-focused GenAI assistant that can:

- Answer HR policy questions using **semantic search (RAG-style)**  
- Analyze **employee feedback** to surface **negative / at-risk** comments  
- Route queries to the right “tool” (policy vs feedback) using a simple **agent-like controller**

Built with:
- `sentence-transformers` for embeddings
- `transformers` for sentiment analysis
- `Streamlit` for the UI

---

## 🚀 Features

### 1. Policy Q&A (Mini-RAG)
- Stores HR policies (PTO, work-from-home, benefits, code of conduct).
- Uses sentence embeddings to find the **most relevant policy** for a user’s question.
- Returns a short summary of the best-matching policy.

Example:
> **Input:** “How far in advance should I request PTO?”  
> **Output:** Picks the *Paid Time Off Policy* and highlights that PTO should be requested about two weeks in advance.

### 2. Employee Feedback Analysis
- Classifies employee feedback as **POSITIVE / NEGATIVE** using a pretrained transformer model.
- Combines **similarity + sentiment** to return the **most relevant negative comments** for a given topic.

Example:
> **Input:** “Show me negative feedback about workload and burnout.”  
> **Output:** Returns comments like *“The workload is too high and I feel burned out most of the time.”*

### 3. Mini-Agent for Intent Routing
- A simple rule-based “agent”:
  - If the question is about **policy** → uses policy RAG module  
  - If about **feedback / employees leaving / burnout** → uses feedback analysis module  
  - Otherwise → explains what it can do

This is an early prototype of an **HR GenAI Copilot** that could be extended with:
- real HR PDFs and CSVs,
- a production LLM for natural language responses,
- authentication and role-based access.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – web UI
- **sentence-transformers** – embeddings (`all-MiniLM-L6-v2`)
- **transformers** – sentiment analysis pipeline
- **PyTorch** – backend for models

---

## 📦 Setup & Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/hr-genai-copilot.git
cd hr-genai-copilot

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate     # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/hr_copilot_app.py
