import streamlit as st
import google.generativeai as genai
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Multi-Agent HR System", page_icon="🧑‍💼", layout="wide")

# --- SESSION STATE INITIALIZATION ---
# This helps us store variables across reruns
st.session_state.setdefault('vector_store', None)
for agent in ["analyzer", "evaluator", "predictor", "recommender"]:
    st.session_state.setdefault(f'{agent}_output', "")

# --- SIDEBAR: CONFIGURATION & KNOWLEDGE BASE ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Enter your Google AI Studio API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.header("2. Knowledge Base for RAG Agent")
    pdf_docs = st.file_uploader("Upload PDF studies on burnout, fatigue, etc.", accept_multiple_files=True)

    if st.button("Process & Index Documents"):
        if not api_key:
             st.error("Please provide your API Key before processing.")
        elif not pdf_docs:
            st.warning("Please upload at least one PDF document.")
        else:
            with st.spinner("Processing documents..."):
                raw_text = "".join(page.extract_text() for pdf in pdf_docs for page in PdfReader(pdf).pages)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                text_chunks = text_splitter.split_text(raw_text)
                
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    st.session_state.vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                    st.success("Knowledge base is ready!")
                except Exception as e:
                    st.error(f"An error occurred during embedding: {e}")

# --- CORE AGENT FUNCTION ---
def run_agent(agent_name: str, input_data: str, model_name='gemini-1.5-flash-latest'):
    """A generic function to run a specific agent."""
    if not os.environ.get("GOOGLE_API_KEY"):
        return "Error: API Key is not configured. Please enter it in the sidebar."

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_name)
    
    agent_prompts = {
        "Work History Analyzer": {
            "persona": "You are a meticulous Data Extraction Agent.",
            "task": f"""
                Analyze the provided 'Candidate's Raw Data'. Extract and structure the following information into clear bullet points:
                - Past Shift Patterns (e.g., nights, rotating, days)
                - Overtime History (e.g., frequent, rare, specific hours)
                - Behavioral Indicators (e.g., job-hopping frequency, promotions, career gaps)
                
                Present only these extracted facts. Do not add any interpretation or analysis.

                **Candidate's Raw Data:**
                {input_data}
            """
        },
        "Shift Compatibility Evaluator": {
            "persona": "You are a Shift Compatibility Evaluator Agent.",
            "task": f"""
                You will receive a candidate's stated preferences and their analyzed work history. Your task is to match this information to assess compatibility with specific shift models.
                Provide a compatibility rating (High, Medium, Low) and a 1-2 sentence justification for each of the following shifts:
                - Night-Only Shifts
                - Rotating Shifts
                - High-Pressure ER-based Shifts

                **Analysis Data:**
                {input_data}
            """
        },
        "Fatigue Risk Predictor": {
            "persona": "You are a RAG-Enabled Fatigue Risk Predictor Agent.",
            "task": f"""
                You are given a candidate's behavioral patterns and relevant context from scientific studies and HR research. Your task is to predict the candidate's fatigue risk in high-pressure units (ICU/ER).
                You MUST ground your prediction in the 'Retrieved RAG Context'.
                1.  State a clear risk level: **Low Risk, Moderate Risk, or High Risk.**
                2.  Provide a detailed justification for your prediction, explicitly referencing both the candidate's patterns and the research findings from the RAG context.

                **Input for Prediction:**
                {input_data}
            """
        },
        "Schedule Fit Recommender": {
            "persona": "You are the final Schedule Fit Recommender Agent, acting as a Senior HR Strategist.",
            "task": f"""
                You have been provided with a complete dossier from three other agents: History Analysis, Compatibility Assessment, and Risk Prediction.
                Your final task is to synthesize all this information into a conclusive report for the hiring manager.
                1.  **Optimal Role & Shift Allocation:** Recommend the most suitable role and specific shift schedule for this candidate.
                2.  **HR Alerts:** Create a bulleted list of 2-3 critical alerts or points of consideration for the HR team regarding this candidate (e.g., potential for burnout, need for mentorship, etc.).

                **Complete Dossier:**
                {input_data}
            """
        }
    }
    
    full_prompt = f"**Persona:** {agent_prompts[agent_name]['persona']}\n\n**Task:**\n{agent_prompts[agent_name]['task']}"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with the API: {e}"

# --- MAIN APP UI ---
st.title("🧑‍💼 HR Agentic Evaluation Workflow")

if st.session_state.vector_store is None:
    st.info("Please upload and process your knowledge base documents in the sidebar to begin.")
else:
    st.success("Knowledge base is active and ready.")
    
    with st.form("candidate_input_form"):
        st.header("Enter Candidate Details")
        name = st.text_input("Candidate Name")
        experience = st.text_area("Previous Experience (from CV/Resume)", height=150)
        work_patterns = st.text_area("Work Patterns / Career History (from CV/LinkedIn)", height=150)
        preferences = st.text_area("Self-Reported Preferences & Goals (from interview notes)", height=100)
        
        submitted = st.form_submit_button("Run 4-Agent Analysis")

    if submitted:
        st.divider()
        st.header(f"Comprehensive Report for: {name}")

        # --- AGENT 1: Work History Analyzer ---
        with st.status("Agent 1: Work History Analyzer is extracting data...", expanded=True) as status:
            raw_data_for_analyzer = f"Experience: {experience}\nWork Patterns: {work_patterns}"
            st.session_state.analyzer_output = run_agent("Work History Analyzer", raw_data_for_analyzer)
            st.markdown(st.session_state.analyzer_output)
            status.update(label="Analysis Complete!", state="complete")

        # --- AGENT 2: Shift Compatibility Evaluator ---
        with st.status("Agent 2: Shift Compatibility Evaluator is assessing fit...", expanded=True) as status:
            time.sleep(1) # Small delay for UX
            input_for_evaluator = f"Analyzed History: {st.session_state.analyzer_output}\n\nStated Preferences: {preferences}"
            st.session_state.evaluator_output = run_agent("Shift Compatibility Evaluator", input_for_evaluator)
            st.markdown(st.session_state.evaluator_output)
            status.update(label="Compatibility Assessed!", state="complete")

        # --- AGENT 3: Fatigue Risk Predictor (RAG) ---
        # RAG happens here
        vector_store = st.session_state.vector_store
        query_for_rag = f"Fatigue and burnout risks related to these patterns: {st.session_state.analyzer_output}"
        docs = vector_store.similarity_search(query_for_rag)
        retrieved_context = "\n".join([doc.page_content for doc in docs])
        
        with st.status("Agent 3: Fatigue Risk Predictor is consulting the knowledge base...", expanded=True) as status:
            time.sleep(1)
            input_for_predictor = f"""
            **Retrieved RAG Context from HR Studies:**
            {retrieved_context}
            ---
            **Candidate's Analyzed Behavioral Patterns:**
            {st.session_state.analyzer_output}
            """
            st.session_state.predictor_output = run_agent("Fatigue Risk Predictor", input_for_predictor)
            st.markdown(st.session_state.predictor_output)
            status.update(label="Risk Predicted!", state="complete")
        
        # --- THIS IS THE FIX: The expander is now OUTSIDE the status block ---
        with st.expander("Show RAG Context Used by Predictor Agent"):
            st.info(retrieved_context)
            
        # --- AGENT 4: Schedule Fit Recommender ---
        with st.status("Agent 4: Recommender Agent is preparing the final report...", expanded=True) as status:
            time.sleep(1)
            input_for_recommender = f"""
            **1. History Analysis:**
            {st.session_state.analyzer_output}
            ---
            **2. Compatibility Evaluation:**
            {st.session_state.evaluator_output}
            ---
            **3. Fatigue Risk Prediction:**
            {st.session_state.predictor_output}
            """
            st.session_state.recommender_output = run_agent("Schedule Fit Recommender", input_for_recommender)
            st.markdown(st.session_state.recommender_output)
            status.update(label="Final Recommendations Ready!", state="complete")

