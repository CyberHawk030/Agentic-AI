import streamlit as st
import google.generativeai as genai
import os
import time
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Multi-Agent HR AI System",
    page_icon="🏆",
    layout="wide"
)

# --- App Header ---
st.title("🏆 Advanced Multi-Agent HR AI System")
st.markdown("This system uses a **Scraper**, **Profiler**, and **Report-Writer** agent for a deep-dive evaluation.")

# --- Session State Initialization ---
st.session_state.setdefault('vector_store', None)
st.session_state.setdefault('scraper_output', "")
st.session_state.setdefault('profiler_output', "")
st.session_state.setdefault('final_report', "")


# --- API Key & Knowledge Base in Sidebar ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Enter your Google AI Studio API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.header("2. Knowledge Base")
    pdf_docs = st.file_uploader("Upload PDF documents (HR policies, burnout studies, etc.)", accept_multiple_files=True)

    if st.button("Process & Index Documents"):
        if not api_key:
             st.error("Please provide your API Key before processing.")
        elif not pdf_docs:
            st.warning("Please upload at least one PDF document.")
        else:
            with st.spinner("Processing documents... This may take a moment."):
                raw_text = "".join(page.extract_text() for pdf in pdf_docs for page in PdfReader(pdf).pages)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                text_chunks = text_splitter.split_text(raw_text)
                
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    st.session_state.vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                    st.success("Knowledge base is ready!")
                except Exception as e:
                    st.error(f"An error occurred during embedding: {e}")

# --- Agent Definitions (Advanced) ---
def run_agent(agent_name: str, input_data: str, model_name='gemini-1.5-flash-latest'):
    """A generic function to run any agent with a specific persona and task."""
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_name)
    
    prompts = {
        "LinkedIn_Scraper": {
            "persona": "You are the 'LinkedIn Profile Scraper Agent'. Your job is to parse unstructured text pasted from a candidate's LinkedIn profile and extract key, structured information.",
            "task": f"""
                From the provided raw profile text, extract the following objective information. Be precise and factual.
                1.  **Summary of Experience:** A brief overview of the candidate's roles and years of experience.
                2.  **Job History & Stability:** List the last 2-3 roles with their durations. Note any job hopping (less than 2 years per role) or employment gaps.
                3.  **Skills & Endorsements:** List the top 5 most relevant skills mentioned.
                Present this as a structured summary.

                **Raw Profile Text to Scrape:**
                {input_data}
            """
        },
        "Profiler": {
            "persona": "You are the 'Candidate Profiler Agent', a meticulous HR analyst. You synthesize information from multiple sources to create a holistic view.",
            "task": f"""
                You have been given a candidate's self-reported preferences, a structured summary from the Scraper Agent, and context from a RAG knowledge base. Combine all three to create a profile.
                1.  **Key Strengths:** 2-3 bullet points, citing evidence from the data.
                2.  **Potential Red Flags:** 2-3 bullet points, citing evidence (e.g., mismatch between preferences and scraped history).
                3.  **Preliminary Fatigue Risk Score:** (Low, Medium, High) with a one-sentence justification.

                **Input Dossier:**
                {input_data}
            """
        },
        "Report_Writer": {
            "persona": "You are the 'Final Report Writer', a senior HR manager creating the official candidate evaluation dossier.",
            "task": f"""
                Synthesize all information from the Scraper and Profiler into a comprehensive final report that directly addresses these four points:
                1.  **Assess compatibility with specific shift models (rotating, night-only, ER-based).** Provide a rating and detailed justification.
                2.  **Predict fatigue risk in high-pressure units like ICU or ER.** State a risk level and explain the behavioral patterns supporting it.
                3.  **Recommend schedule-matching roles.** Provide a specific, actionable recommendation for the initial role and schedule.
                4.  **Alert HR on potential burnout-prone fits.** Create a clear 'HR ALERTS' section with 2-3 bullet points.

                **Full Dossier for Final Analysis:**
                {input_data}
            """
        }
    }
    
    full_prompt = f"**Persona:** {prompts[agent_name]['persona']}\n\n**Task:**\n{prompts[agent_name]['task']}"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error executing {agent_name} agent: {e}"

# --- Main Application UI ---
if st.session_state.vector_store is None:
    st.warning("Please upload and process documents in the sidebar to begin.")
else:
    st.success("Knowledge base is active. You can now start an evaluation.")
    
    st.header("Candidate Evaluation Workflow")
    
    with st.form("advanced_agent_form"):
        st.write("Enter candidate details to initiate the advanced three-agent analysis.")
        name = st.text_input("Candidate Name")
        linkedin_url = st.text_input("Candidate LinkedIn Profile URL")
        profile_text = st.text_area("Paste LinkedIn Profile Text Here", height=200, placeholder="To get this, go to the candidate's LinkedIn page, click 'More', then 'Save to PDF', and copy the text from the PDF.")
        preferences = st.text_area("Self-Reported Preferences & Goals", height=100, placeholder="e.g., 'Looking for a stable night shift role...'")
        
        submitted = st.form_submit_button("Run Advanced Analysis")

    if submitted:
        st.divider()
        st.header(f"Analysis Dossier for {name}")

        # --- Agent 1: LinkedIn Scraper ---
        with st.spinner("Step 1/3: Scraper Agent is analyzing the profile text..."):
            st.session_state.scraper_output = run_agent("LinkedIn_Scraper", profile_text)
        st.subheader("1. Structured Career History (from Scraper Agent)")
        st.markdown(st.session_state.scraper_output)

        # --- Agent 2: RAG Profiler ---
        with st.spinner("Step 2/3: RAG Profiler is creating a holistic profile..."):
            query = f"Create a profile for a candidate with this history: {st.session_state.scraper_output}"
            docs = st.session_state.vector_store.similarity_search(query)
            retrieved_context = "\n".join([doc.page_content for doc in docs])
            
            profiler_input = f"""
            **Retrieved RAG Context:**
            {retrieved_context}
            ---
            **Scraper Agent Summary:**
            {st.session_state.scraper_output}
            ---
            **Candidate Preferences:**
            {preferences}
            """
            st.session_state.profiler_output = run_agent("Profiler", profiler_input)
        
        st.subheader("2. Synthesized Candidate Profile")
        st.markdown(st.session_state.profiler_output)
        with st.expander("Show Context Used by Profiler"):
            st.info(retrieved_context)

        # --- Agent 3: Report Writer ---
        with st.spinner("Step 3/3: Report Writer is compiling the final evaluation..."):
            report_writer_input = f"""
            **Profiler Agent's Analysis:**
            {st.session_state.profiler_output}
            ---
            **Scraper Agent's Summary:**
            {st.session_state.scraper_output}
            """
            st.session_state.final_report = run_agent("Report_Writer", report_writer_input)
            
        st.subheader("3. Final Evaluation Report & Recommendations")
        st.markdown(st.session_state.final_report)
        st.success("Advanced Multi-Agent Analysis Complete!")
