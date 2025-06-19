# Multi-Agent HR System for Shift Resilience & Fatigue Risk

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)
![CrewAI](https://img.shields.io/badge/CrewAI-0.35%2B-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green.svg)

An advanced Streamlit application demonstrating a multi-agent AI system built with **CrewAI** and **LangChain**. This system analyzes job candidates for roles with demanding schedules (e.g., healthcare, logistics) to assess their compatibility and predict their risk of burnout and fatigue, powered by a custom RAG knowledge base.

---

## 📝 Problem Statement

In high-stakes industries like healthcare, hospitals often struggle to identify early signs of burnout or mismatch in a candidate's suitability for specific shift patterns. This can negatively impact patient safety, operational efficiency, and long-term staff retention.

This project, "Shift Resilience & Fatigue Risk Evaluator," was built to solve this problem by creating an AI system that provides a deep, multi-faceted evaluation of a candidate's fit for demanding roles.

## ✨ Key Features

* **Multi-Agent Architecture:** Utilizes **CrewAI** to orchestrate a team of four distinct AI agents, each with a specialized role, to collaborate on a final recommendation.
* **RAG-Powered Insights:** Features a **RAG (Retrieval-Augmented Generation)** agent that grounds its risk predictions in scientific literature and HR case studies provided by the user.
* **Interactive Web UI:** A user-friendly interface built with **Streamlit** that allows HR personnel to input candidate data, manage the knowledge base, and receive a comprehensive report.
* **End-to-End Analysis:** The system performs a complete workflow from raw data extraction to a final, strategic recommendation with actionable alerts.
* **Custom Knowledge Base:** Users can upload their own PDF documents (e.g., internal studies, academic papers) to create a bespoke knowledge base for the RAG agent.

## 🤖 System Architecture & Agent Roles

The application functions as a sequential pipeline managed by a **CrewAI** crew. Each agent completes its task and passes its findings to the next, creating a rich context for the final decision.

1.  **Work History Analyzer Agent:**
    * **Role:** Data Extractor
    * **Task:** Receives raw text from a candidate's CV and interview notes. It meticulously extracts objective facts like past shift types, overtime frequency, job-hopping patterns, and career gaps. It does not interpret, it only reports.

2.  **Shift Compatibility Evaluator Agent:**
    * **Role:** Matchmaker
    * **Task:** Takes the structured facts from the Analyzer and the candidate's stated preferences. It assesses and rates the candidate's compatibility with specific, high-stress shift models (Night-Only, Rotating, ER-based).

3.  **Fatigue Risk Predictor Agent (RAG-Enabled):**
    * **Role:** Risk Analyst & Researcher
    * **Task:** This is the RAG-enabled agent. It takes the behavioral patterns identified by the Analyzer and uses its `Fatigue Research Knowledge Base` tool to find relevant precedents in the uploaded documents. It then synthesizes this information to predict a Low, Moderate, or High risk of burnout, justifying its conclusion with evidence from the literature.

4.  **Schedule Fit Recommender Agent:**
    * **Role:** Senior HR Strategist
    * **Task:** The final agent in the chain. It receives the complete dossier from the previous three agents. It synthesizes all information to provide a conclusive recommendation on the optimal role and shift allocation, and flags 2-3 critical "HR Alerts" for the hiring manager.

## 🛠️ Technology Stack

* **Orchestration Framework:** CrewAI
* **Core AI/LLM Library:** LangChain, LangChain Google Generative AI
* **Web Framework:** Streamlit
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings Model:** Google Generative AI Embeddings
* **LLM:** Google Gemini (e.g., `gemini-1.5-flash-latest`)
* **PDF Processing:** PyPDF2

---

## ⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/CyberHawk030/Agentic-AI.git](https://github.com/CyberHawk030/Agentic-AI.git)
cd Agentic-AI
cd Day 9 Task
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```