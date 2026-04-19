import streamlit as st
import requests
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="DocLens - Intelligent Document Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .question-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #333;
    }
    .answer-card {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #333;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Session state init
if 'questions' not in st.session_state:
    st.session_state.questions = [""]
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'pdf_url' not in st.session_state:
    st.session_state.pdf_url = ""


def validate_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def is_pdf_url(url: str) -> bool:
    return url.lower().endswith('.pdf') or 'pdf' in url.lower()


def add_question():
    st.session_state.questions.append("")


def remove_question(index: int):
    if len(st.session_state.questions) > 1:
        st.session_state.questions.pop(index)
        st.rerun()


def process_document():
    if not st.session_state.pdf_url or not st.session_state.pdf_url.strip():
        st.error("Please enter a valid PDF URL")
        return

    valid_questions = [q.strip() for q in st.session_state.questions if q.strip()]
    if not valid_questions:
        st.error("Please enter at least one question")
        return

    if not validate_url(st.session_state.pdf_url):
        st.error("Please enter a valid URL")
        return

    if not is_pdf_url(st.session_state.pdf_url):
        st.warning("The URL doesn't appear to point to a PDF. Proceeding anyway...")

    st.session_state.processing = True

    try:
        payload = {
            "documents": st.session_state.pdf_url.strip(),
            "questions": valid_questions
        }

        # ---------- UPDATE THIS URL after deploying the backend to Vercel ----------
        api_url = os.getenv("BACKEND_DOCLENS_API_URL", "http://localhost:8000/DocLens")
        # --------------------------------------------------------------------------

        with st.spinner("Processing document and generating answers..."):
            response = requests.post(api_url, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                st.session_state.answers = result.get('answers', [])
                st.success("Document processed successfully!")
            else:
                error_detail = response.json().get('detail', 'Unknown error occurred')
                st.error(f"Error: {error_detail}")

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        st.info("Make sure your backend URL is correct in the sidebar settings.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        st.session_state.processing = False


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Backend URL")

    # Allow overriding backend URL from sidebar at runtime
    sidebar_backend = st.text_input(
        "Backend API URL",
        value=os.getenv("BACKEND_DOCLENS_API_URL", "http://localhost:8000/DocLens"),
        help="Your Vercel backend URL, e.g. https://your-app.vercel.app/DocLens"
    )
    if sidebar_backend:
        os.environ["BACKEND_DOCLENS_API_URL"] = sidebar_backend

    st.divider()

    # Health check
    st.subheader("🔍 Backend Status")
    base_url = sidebar_backend.replace("/DocLens", "")
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Error")
    except:
        st.warning("⚠️ Backend Unreachable")
        st.caption("Set the correct backend URL above.")

    st.divider()
    st.subheader("ℹ️ About DocLens")
    st.markdown("""
    DocLens uses AI to:
    - Extract text from PDF documents
    - Find relevant sections for each question
    - Generate context-aware answers with Groq LLaMA
    """)


# ---------- Header ----------
st.markdown("""
<div class="main-header">
    <h1>📄 DocLens</h1>
    <p style="font-size: 1.2rem; margin: 0;">Intelligent Document Analysis with RAG</p>
    <p style="font-size: 1rem; margin: 0.5rem 0 0 0;">Upload PDF documents and get intelligent answers to your questions</p>
</div>
""", unsafe_allow_html=True)

# ---------- Main Layout ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📚 Document Analysis")
    st.subheader("1. Document Input")
    pdf_url = st.text_input(
        "PDF Document URL",
        placeholder="https://example.com/document.pdf",
        help="Enter the URL of the PDF document you want to analyze"
    )
    if pdf_url:
        st.session_state.pdf_url = pdf_url
        if validate_url(pdf_url):
            if is_pdf_url(pdf_url):
                st.success("✅ Valid PDF URL detected")
            else:
                st.warning("⚠️ Valid URL but may not be a PDF")
        else:
            st.error("❌ Invalid URL format")

with col2:
    st.header("📊 Quick Stats")
    if st.session_state.answers:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Results</h3>
            <h2>{len(st.session_state.answers)}</h2>
            <p>Questions Answered</p>
        </div>
        """, unsafe_allow_html=True)

# ---------- Questions ----------
st.header("❓ Questions")
st.markdown("Enter your questions. You can add multiple questions to analyze different aspects of the document.")

for i, question in enumerate(st.session_state.questions):
    col1, col2 = st.columns([6, 1])
    with col1:
        st.session_state.questions[i] = st.text_input(
            f"Question {i+1}",
            value=question,
            placeholder=f"Enter your question {i+1} here...",
            key=f"question_{i}"
        )
    with col2:
        if len(st.session_state.questions) > 1:
            if st.button("🗑️", key=f"remove_{i}", help="Remove this question"):
                remove_question(i)

if st.button("➕ Add Another Question", on_click=add_question):
    pass

st.divider()

if st.button(
    "🚀 Process Document & Get Answers",
    on_click=process_document,
    disabled=st.session_state.processing,
    type="primary"
):
    pass

# ---------- Results ----------
if st.session_state.answers:
    st.header("💡 Answers")
    st.markdown("Here are the answers based on the document content:")

    for i, (question, answer) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
        if question.strip():
            st.markdown(f"""
            <div class="question-card">
                <h4>❓ Question {i+1}:</h4>
                <p><strong>{question}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="answer-card">
                <h4>💡 Answer:</h4>
                <p>{answer}</p>
            </div>
            """, unsafe_allow_html=True)
            st.divider()

# ---------- How It Works ----------
with st.expander("ℹ️ How DocLens Works"):
    st.markdown("""
    ### 🔍 Document Processing Pipeline
    1. **PDF Extraction**: Downloads and extracts text from your PDF document
    2. **Text Chunking**: Breaks the document into overlapping chunks
    3. **TF-IDF Retrieval**: Finds the most relevant chunks for each question
    4. **RAG Generation**: Groq's LLaMA model generates answers from retrieved context

    ### 📋 Best Practices
    - Use clear, specific questions for better results
    - Ensure PDF URLs are publicly accessible
    - For complex topics, break questions into smaller parts
    - Maximum 20 questions per request
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Built with ❤️ using Streamlit and FastAPI | DocLens v1.0.0"
    "</div>",
    unsafe_allow_html=True
)
