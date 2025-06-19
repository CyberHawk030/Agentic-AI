import streamlit as st
import os
import time
from PyPDF2 import PdfReader

# LangChain & CrewAI Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from crewai import Agent, Task, Crew, Process

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Multi-Agent HR System", page_icon="🧑‍💼", layout="wide")

# --- SESSION STATE INITIALIZATION ---
# This helps us store variables across reruns
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'final_report' not in st.session_state:
    st.session_state.final_report = ""


# --- RAG TOOL DEFINITION ---
# This is the tool that our agent will use to access the knowledge base.
@tool("Fatigue Research Knowledge Base")
def fatigue_rag_tool(query: str) -> str:
    """
    Searches and returns relevant context about fatigue, burnout, and shift work 
    from HR research documents and scientific studies.
    """
    vector_store = st.session_state.get('vector_store', None)
    if vector_store:
        docs = vector_store.similarity_search(query)
        return "\n".join([doc.page_content for doc in docs])
    return "Knowledge base not initialized. Please upload documents in the sidebar."

# --- SIDEBAR: CONFIGURATION & KNOWLEDGE BASE ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Enter your Google AI Studio API Key", type="password", key="api_key_input")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.header("2. Knowledge Base for RAG Agent")
    pdf_docs = st.file_uploader("Upload PDF studies on burnout, fatigue, etc.", accept_multiple_files=True)

    if st.button("Process & Index Documents"):
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Please provide your API Key before processing.")
        elif not pdf_docs:
            st.warning("Please upload at least one PDF document.")
        else:
            with st.spinner("Processing documents... This may take a moment."):
                # 1. Extract Text
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text() or ""
                
                # 2. Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                text_chunks = text_splitter.split_text(raw_text)
                
                # 3. Create Embeddings & Vector Store
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    st.session_state.vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                    st.success("Knowledge base is ready!")
                except Exception as e:
                    st.error(f"An error occurred during embedding: {e}")

# --- MAIN APP UI ---
st.title("🧑‍💼 HR Agentic Evaluation Workflow")
st.markdown("This system uses a crew of AI agents to analyze a candidate for shift compatibility and burnout risk.")

# Check if the knowledge base is ready before proceeding
if st.session_state.vector_store is None:
    st.info("Please configure your API Key and process your knowledge base documents in the sidebar to begin.")
else:
    st.success("Knowledge base is active. You can now enter candidate details.")
    
    with st.form("candidate_input_form"):
        st.header("Enter Candidate Details")
        name = st.text_input("Candidate Name")
        experience = st.text_area("Previous Experience (from CV/Resume)", height=150, placeholder="e.g., Worked as a registered nurse at City General Hospital for 5 years on the night shift...")
        work_patterns = st.text_area("Work Patterns / Career History (from CV/LinkedIn)", height=150, placeholder="e.g., Consistently worked 10-15 hours of overtime per month. Switched jobs every 2 years...")
        preferences = st.text_area("Self-Reported Preferences & Goals (from interview notes)", height=100, placeholder="e.g., Expressed strong preference for stable day shifts. Mentioned wanting to avoid high-stress environments...")
        
        submitted = st.form_submit_button("Run 4-Agent Analysis")

    if submitted:
        if not all([name, experience, work_patterns, preferences]):
            st.warning("Please fill out all candidate details.")
        else:
            # --- AGENT & TASK DEFINITION (CREWAI) ---
            
            # Initialize the LLM for the agents
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)

            # 1. Define Agents
            analyzer_agent = Agent(
                role="Work History Analyzer Agent",
                goal="Extract objective facts about past shift patterns, overtime history, and behavioral indicators from a candidate's professional history.",
                backstory="You are a meticulous data extraction specialist. Your sole purpose is to state facts as structured data without any form of interpretation or analysis.",
                llm=llm,
                verbose=True
            )

            evaluator_agent = Agent(
                role="Shift Compatibility Evaluator Agent",
                goal="Assess a candidate's compatibility with various demanding shift models (night-only, rotating, ER-based) by cross-referencing their work history with their stated preferences.",
                backstory="You are an expert HR analyst specializing in shift work compatibility. You provide clear ratings and concise justifications.",
                llm=llm,
                verbose=True
            )

            predictor_agent = Agent(
                role="Fatigue Risk Predictor Agent",
                goal="Predict a candidate's potential for burnout and fatigue in high-pressure roles by grounding your analysis in scientific research and HR case studies.",
                backstory="You are a data-driven HR risk analyst. You MUST use the provided 'Fatigue Research Knowledge Base' tool to find relevant studies and cite them in your prediction.",
                tools=[fatigue_rag_tool],
                llm=llm,
                verbose=True
            )

            recommender_agent = Agent(
                role="Schedule Fit Recommender Agent",
                goal="Synthesize all prior analyses (history, compatibility, risk) into a final, actionable recommendation report for an HR hiring manager.",
                backstory="You are a seasoned Senior HR Strategist. Your focus is on maximizing employee potential while ensuring long-term well-being and retention. Your recommendations are clear, strategic, and include actionable alerts.",
                llm=llm,
                verbose=True
            )

            # 2. Define Tasks
            # Combine raw data for the first task
            full_history_data = f"""
            Candidate Experience: {experience}
            Candidate Work Patterns & Career History: {work_patterns}
            """

            # This is the data for the second task
            preferences_data = f"Candidate Stated Preferences & Goals: {preferences}"
            
            task1 = Task(
                description=f"Analyze the following candidate history data:\n\n{full_history_data}",
                expected_output="A clean, bulleted list of extracted facts covering: Past Shift Patterns, Overtime History, and Behavioral Indicators.",
                agent=analyzer_agent
            )

            task2 = Task(
                description=f"Using the candidate's preferences below, and the history analysis from the previous step, evaluate shift compatibility.\n\nPreferences: {preferences_data}",
                expected_output="A compatibility rating (High, Medium, Low) and a 1-2 sentence justification for each of these shifts: Night-Only, Rotating, and High-Pressure ER-based.",
                context=[task1],
                agent=evaluator_agent
            )
            
            task3 = Task(
                description="Based on the candidate's behavioral patterns from the history analysis, predict their long-term fatigue risk. Use your Knowledge Base tool to find relevant studies on patterns like job-hopping, overtime, or specific shift histories to support your prediction.",
                expected_output="A final risk level (Low, Moderate, or High) with a detailed justification that explicitly references findings from the retrieved research documents.",
                context=[task1],
                agent=predictor_agent
            )

            task4 = Task(
                description="Create a final, comprehensive recommendation report for the hiring manager. Synthesize the outputs from all previous tasks (history facts, compatibility ratings, and the risk prediction) into a single report.",
                expected_output="A final report containing two sections: 1. 'Optimal Role & Shift Allocation' (your final recommendation). 2. 'HR Alerts' (a bulleted list of 2-3 critical considerations).",
                context=[task1, task2, task3],
                agent=recommender_agent
            )

            # 3. Assemble and Kickoff the Crew
            hr_crew = Crew(
                agents=[analyzer_agent, evaluator_agent, predictor_agent, recommender_agent],
                tasks=[task1, task2, task3, task4],
                process=Process.sequential,
                verbose=2
            )

            st.divider()
            st.header(f"Comprehensive Report for: {name}")

            with st.spinner("The HR Agent crew is evaluating the candidate... This may take a few minutes."):
                try:
                    # Run the crew's tasks
                    result = hr_crew.kickoff()
                    st.session_state.final_report = result
                except Exception as e:
                    st.error(f"An error occurred while running the agent crew: {e}")
                    st.session_state.final_report = ""

# Display the final report if it exists
if st.session_state.final_report:
    st.markdown("---")
    st.subheader("Final Recommendation Report")
    st.markdown(st.session_state.final_report)