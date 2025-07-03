# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
    
import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
import time
import random


# Custom CSS for better appearance
def add_custom_css():
    """Add custom styling to make app look professional and modern"""
    st.markdown("""
    <style>
    /* Full Page Background */
    .stApp {
        background-image: url("https://i.imgur.com/34EAPUj.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Main content area structure */
    .main {
        display: flex;
        justify-content: center;
        width: 100%;
        color: white;
    }
    
    /* Sidebar Background - deeper blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72, #152a54);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Content Area Background - purple gradient */
    .main .block-container {
        background: linear-gradient(135deg, rgba(80, 50, 126, 0.95), rgba(102, 40, 140, 0.95));
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        width: 1200px; /* Fixed width for the content */
    }
    
    /* Remove the max-width from stApp to allow full-width background */
    .stApp {
        max-width: none;
        margin: 0;
    }
    
    /* Create space on the sides to let background image show through */
    [data-testid="stAppViewContainer"] {
        padding: 0 20px;
    }
    
    /* Constrain the main content width with pastel green */
    .main .block-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4CAF50, #2196F3, #7f53ac);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        font-family: 'Segoe UI', system-ui, sans-serif;
        text-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Card Styling */
    .metric-card {
        background: rgba(40, 80, 150, 0.6);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        border: none;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.2);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: none;
        padding: 0;
        gap: 1rem; /* space between tabs */
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 60, 114, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
        margin: 0;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(60, 103, 186, 0.7);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3c67ba, #7f53ac);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    /* Answer container styling */
    .answer-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 0.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        color: #1a1a1a;
    }

    .answer-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E7D32;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }

    .answer-header span {
        margin-left: 0.5rem;
    }

    .answer-source {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }

    .answer-content {
        font-size: 1.1rem;
        line-height: 1.6;
        max-height: 400px; /* Set a max height */
        overflow-y: auto; /* Add scroll for long answers */
        padding: 0.5rem;
        background: rgba(240, 240, 240, 0.5);
        border-radius: 8px;
    }
    
    /* Input Fields */
    .stTextInput > div > div {
        border-radius: 10px;
        border: 2px solid #eee;
        padding: 0.5rem;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(60, 103, 186, 0.5) !important;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1rem;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(80, 120, 200, 0.6) !important;
    }
    
    /* Alert/Info Boxes */
    .stAlert {
        background: rgba(60, 103, 186, 0.6) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #2196F3, #1e88e5);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        margin: 1rem 0;
    }
    
    /* Sidebar Styling - ensure no white background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72, #152a54);
        padding: 2rem 1rem;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        font-size: 0.9rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        background: rgba(60, 103, 186, 0.7);
        padding: 0.5rem;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Charts */
    [data-testid="stPlotlyChart"] > div {
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    /* General Typography - light text on dark background */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    p, li, label, .streamlit-expanderHeader {
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stMarkdown, .stAlert {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Style file uploader - darker with glow effect */
    [data-testid="stFileUploader"] {
        background: rgba(30, 60, 114, 0.6);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 0 10px rgba(127, 83, 172, 0.5);
    }
    
    /* Style text inputs */
    .stTextInput > div {
        background: rgba(30, 60, 114, 0.6);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 60, 114, 0.5);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(127, 83, 172, 0.7);
    }
    
    /* Style expanders */
    .streamlit-expanderHeader {
        background: rgba(60, 103, 186, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Ensure no white backgrounds anywhere */
    div.stButton button:hover {
        background: linear-gradient(135deg, #3c67ba, #7f53ac) !important;
    }
    
    .stTextInput input, .stSelectbox, [data-baseweb="select"] {
        background-color: rgba(30, 60, 114, 0.6) !important;
        color: white !important;
    }
    
    [data-testid="stMarkdownContainer"] code {
        background-color: rgba(30, 30, 70, 0.6) !important;
    }
    
    /* Remove extra padding around text input */
    .stTextInput {
        padding: 0 !important;
        margin-bottom: 1rem !important;
    }

    /* Remove white container backgrounds */
    div[data-testid="stVerticalBlock"] > div > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Clean empty containers */
    .element-container {
        margin-bottom: 0 !important;
    }
    
    /* Remove gray background around input */
    div.stTextInput > label {
        background-color: transparent !important;
    }
    
    div.stTextInput > div {
        background-color: transparent !important;
    }

    .answer-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E7D32;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }

    .answer-header span {
        margin-left: 0.5rem;
    }

    .answer-source {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }

    .answer-content {
        font-size: 1.1rem;
        line-height: 1.6;
        max-height: 400px; /* Set a max height */
        overflow-y: auto; /* Add scroll for long answers */
        padding: 0.5rem;
        background: rgba(240, 240, 240, 0.5);
        border-radius: 8px;
    }
    
    /* Input Fields */
    .stTextInput > div > div {
        border-radius: 10px;
        border: 2px solid #eee;
        padding: 0.5rem;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(60, 103, 186, 0.5) !important;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1rem;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(80, 120, 200, 0.6) !important;
    }
    
    /* Alert/Info Boxes */
    .stAlert {
        background: rgba(60, 103, 186, 0.6) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #2196F3, #1e88e5);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        margin: 1rem 0;
    }
    
    /* Sidebar Styling - ensure no white background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72, #152a54);
        padding: 2rem 1rem;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        font-size: 0.9rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        background: rgba(60, 103, 186, 0.7);
        padding: 0.5rem;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Charts */
    [data-testid="stPlotlyChart"] > div {
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    /* General Typography - light text on dark background */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    p, li, label, .streamlit-expanderHeader {
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stMarkdown, .stAlert {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Style file uploader - darker with glow effect */
    [data-testid="stFileUploader"] {
        background: rgba(30, 60, 114, 0.6);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 0 10px rgba(127, 83, 172, 0.5);
    }
    
    /* Style text inputs */
    .stTextInput > div {
        background: rgba(30, 60, 114, 0.6);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 60, 114, 0.5);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(127, 83, 172, 0.7);
    }
    
    /* Style expanders */
    .streamlit-expanderHeader {
        background: rgba(60, 103, 186, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Ensure no white backgrounds anywhere */
    div.stButton button:hover {
        background: linear-gradient(135deg, #3c67ba, #7f53ac) !important;
    }
    
    .stTextInput input, .stSelectbox, [data-baseweb="select"] {
        background-color: rgba(30, 60, 114, 0.6) !important;
        color: white !important;
    }
    
    [data-testid="stMarkdownContainer"] code {
        background-color: rgba(30, 30, 70, 0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Convert uploaded file to markdown text
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# Reset ChromaDB collection
def reset_collection(client, collection_name: str):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)


# Add text chunks to ChromaDB
def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}

    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection

    collection = add_text_to_chromadb.collections[collection_name]

    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()

        metadata = {
            "filename": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }

        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )

    return collection


# Q&A function
def get_answer_with_source(collection, question):
    """Get answer from documents based on current AI personality."""
    # Query the collection
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]
    
    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic.", "No source"
    
    # Get the current personality settings from session state
    personality = st.session_state.get('personality', {
        "style": "formal and precise",
        "tone": "professional",
        "context": "focusing on accuracy and clarity"
    })
    
    # Create context with relevant document snippets
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # Generate a response prompt based on personality
    prompt = f"""Based on the following context, provide a {personality['style']} answer,
using a {personality['tone']} tone, {personality['context']}.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the above context."""

    # Use text generation instead of question-answering for more flexible responses
    model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # Using a larger model for better responses
        max_length=200,
        temperature=0.7
    )
    
    # Get the answer
    response = model(prompt, max_length=200, num_return_sequences=1)
    answer = response[0]['generated_text'].strip()
    
    # Find the most relevant source document and its metadata
    best_idx = 0
    min_distance = distances[0]
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        if dist < min_distance:
            min_distance = dist
            best_idx = i
    
    source_doc = docs[best_idx]
    source_meta = results["metadatas"][0][best_idx]
    source_filename = source_meta["filename"]
    
    # Format the answer based on personality
    formatted_answer = answer
    prefix = ""
    if personality["tone"] == "friendly":
        prefix = "Here's what I found"
    elif personality["tone"] == "professional":
        prefix = "Analysis"
    elif personality["tone"] == "expert":
        prefix = "Based on the documentation"
    
    formatted_answer = f"""{prefix} (from '{source_filename}'):

{answer}"""
    
    return formatted_answer, source_filename


# FEATURE: Search history
def add_to_search_history(question, answer, source):
    """Add search to history."""

    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Add new search to beginning of list
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Keep only last 10 searches
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]


def show_search_history():
    """Display search history with enhanced visual appeal."""
    
    st.markdown("""
        <div style='padding: 1rem; margin: 1rem 0; background: linear-gradient(135deg, rgba(46,125,50,0.1), rgba(33,150,243,0.1)); border-radius: 15px;'>
            <h3 style='color: #2E7D32; margin-bottom: 1rem;'>🕒 Your Learning Journey</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("Start your journey by asking questions about your documents!")
        return
    
    for i, search in enumerate(st.session_state.search_history):
        with st.expander(
            f"🔍 Q{i+1}: {search['question'][:50]}... ({search['timestamp']})", 
            expanded=(i == 0)  # Auto-expand the most recent search
        ):
            st.markdown("""
                <div style='padding: 1rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);'>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style='margin-bottom: 0.5rem;'>
                    <span style='color: #1a1a1a; font-weight: 600;'>Question:</span>
                    <p style='margin: 0.5rem 0; color: #2E7D32;'>{search['question']}</p>
                </div>
                
                <div style='margin-bottom: 0.5rem;'>
                    <span style='color: #1a1a1a; font-weight: 600;'>Answer:</span>
                    <p style='margin: 0.5rem 0; padding: 0.5rem; background: rgba(46,125,50,0.05); border-radius: 5px;'>
                        {search['answer']}
                    </p>
                </div>
                
                <div style='font-size: 0.9rem; color: #666;'>
                    <span style='color: #1a1a1a; font-weight: 600;'>Source:</span> 📄 {search['source']}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)


def enhanced_question_interface():
    """Professional question asking interface."""
    st.subheader("💬 Ask Your Question")
    
    # Provide example questions
    with st.expander("💡 Example questions you can ask"):
        example_questions = [
            "What are the main topics covered in these documents?",
            "Summarize the key points from [document name]",
            "What does the document say about [specific topic]?",
            "Compare information between documents",
            "Find specific data or statistics"
        ]
        st.markdown("\n".join([f"- {q}" for q in example_questions]))
    
    # Question input with suggestions - using a container to better control styling
    question_container = st.container()
    with question_container:
        question = st.text_input(
            "Type your question here:",
            placeholder="e.g., What are the main findings in the research paper?"
        )
    
    # Two-column layout for buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        search_button = st.button("🔍 Search Documents", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear History")
    
    return question, search_button, clear_button


def show_document_manager():
    """Display document manager interface."""
    
    st.subheader("📋 Manage Documents")
    
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []

    if not st.session_state.converted_docs:
        st.info("No documents uploaded yet.")
        return
    
    # Show each document with delete button
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        
        with col2:
            # Preview button
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        
        with col3:
            # Delete button
            if st.button("Delete", key=f"delete_{i}"):
                # Remove from session state
                st.session_state.converted_docs.pop(i)
                # Rebuild database
                client = chromadb.Client()
                try:
                    collection = client.delete_collection(name="documents")
                    collection = client.create_collection(name="documents")
                    # Re-add remaining documents
                    for doc in st.session_state.converted_docs:
                        add_text_to_chromadb(doc['content'], doc['filename'])
                except Exception as e:
                    st.error(f"Error rebuilding database: {e}")
                st.rerun()
        
        # Show preview if requested
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()


# Document Analytics feature
def show_document_analytics():
    """Display interactive analytics and insights about the documents"""
    st.subheader("📊 Document Analytics & Insights")
    
    if not st.session_state.converted_docs:
        st.info("Upload some documents to see analytics!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Document type distribution
        file_types = {}
        for doc in st.session_state.converted_docs:
            ext = Path(doc['filename']).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(file_types.keys()),
            values=list(file_types.values()),
            hole=.3,
            marker_colors=['#2E7D32', '#43A047', '#66BB6A', '#81C784']
        )])
        fig.update_layout(title="Document Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Document sizes comparison
        doc_sizes = [{'name': doc['filename'], 'words': len(doc['content'].split())} 
                    for doc in st.session_state.converted_docs]
        doc_sizes.sort(key=lambda x: x['words'], reverse=True)
        
        fig = px.bar(
            doc_sizes, 
            x='name', 
            y='words',
            title="Document Sizes (Word Count)",
            color_discrete_sequence=['#2E7D32']
        )
        fig.update_layout(xaxis_title="Document", yaxis_title="Word Count")
        st.plotly_chart(fig, use_container_width=True)
    
    # Just keep the document type distribution and size comparison charts above
    
    # Search patterns analysis
    if 'search_history' in st.session_state and st.session_state.search_history:
        st.subheader("🔍 Search Pattern Analysis")
        
        # Most referenced documents
        doc_references = {}
        for search in st.session_state.search_history:
            doc_references[search['source']] = doc_references.get(search['source'], 0) + 1
        
        fig = px.bar(
            x=list(doc_references.keys()),
            y=list(doc_references.values()),
            title="Most Referenced Documents in Q&A",
            color_discrete_sequence=['#2E7D32']
        )
        fig.update_layout(
            xaxis_title="Document",
            yaxis_title="Times Referenced",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent search patterns
        st.markdown("#### Recent Search Patterns")
        searches_by_hour = {}
        for search in st.session_state.search_history:
            hour = datetime.strptime(search['timestamp'], "%Y-%m-%d %H:%M:%S").hour
            searches_by_hour[hour] = searches_by_hour.get(hour, 0) + 1
        
        # Fill in missing hours with 0
        for hour in range(24):
            if hour not in searches_by_hour:
                searches_by_hour[hour] = 0
        
        # Sort by hour
        hours = sorted(searches_by_hour.keys())
        counts = [searches_by_hour[hour] for hour in hours]
        
        fig = px.line(
            x=hours,
            y=counts,
            title="Search Activity by Hour",
            color_discrete_sequence=['#2E7D32']
        )
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Number of Searches"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Document statistics
    show_document_stats()


def show_document_stats():
    """Show detailed statistics about uploaded documents"""
    if not st.session_state.converted_docs:
        st.info("No documents to analyze.")
        return

    # Calculate stats with more detail
    total_docs = len(st.session_state.converted_docs)
    total_words = sum(len(doc['content'].split()) for doc in st.session_state.converted_docs)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    
    # Get word distribution
    word_counts = [len(doc['content'].split()) for doc in st.session_state.converted_docs]
    max_words = max(word_counts)
    min_words = min(word_counts)
    
    # Display in columns with enhanced metrics
    st.markdown("""
        <div class='metric-card'>
            <h3 style='text-align: center; color: #2E7D32; margin-bottom: 1rem;'>📊 Document Overview</h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", f"{total_docs:,}")
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Average Words", f"{avg_words:,}")
    with col4:
        st.metric("Word Range", f"{min_words:,} - {max_words:,}")
    
    # Show breakdown by file type with visual enhancement
    st.markdown("### 📁 File Type Distribution")
    file_types = {}
    for doc in st.session_state.converted_docs:
        ext = Path(doc['filename']).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    # Create a more visual representation
    for ext, count in file_types.items():
        percentage = (count / total_docs) * 100
        st.markdown(f"""
            <div style='background: linear-gradient(90deg, rgba(46,125,50,0.2) {percentage}%, transparent {percentage}%);
                      padding: 0.5rem 1rem; border-radius: 5px; margin: 0.2rem 0;'>
                {ext}: {count} files ({percentage:.1f}%)
            </div>
        """, unsafe_allow_html=True)


# Enhanced error handling
def safe_convert_files(uploaded_files):
    """Convert files with comprehensive error handling"""
    converted_docs = []
    errors = []
    
    if not uploaded_files:
        return converted_docs, ["No files uploaded"]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Converting {uploaded_file.name}...")
            
            # Check file size (limit to 10MB)
            if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
                errors.append(f"{uploaded_file.name}: File too large (max 10MB)")
                continue
            
            # Check file type
            allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            if file_ext not in allowed_extensions:
                errors.append(f"{uploaded_file.name}: Unsupported file type")
                continue
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                # Convert to markdown
                markdown_content = convert_to_markdown(tmp_path)
                
                # Validate content
                if len(markdown_content.strip()) < 10:
                    errors.append(f"{uploaded_file.name}: File appears to be empty or corrupted")
                    continue
                
                # Store successful conversion
                converted_docs.append({
                    'filename': uploaded_file.name,
                    'content': markdown_content,
                    'size': len(uploaded_file.getvalue()),
                    'word_count': len(markdown_content.split())
                })
                
            finally:
                # Always cleanup temp file
                Path(tmp_path).unlink(missing_ok=True)
                
        except Exception as e:
            errors.append(f"{uploaded_file.name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Conversion complete!")
    return converted_docs, errors

def show_conversion_results(converted_docs, errors):
    """Display conversion results with good UX"""
    if converted_docs:
        st.success(f"✅ Successfully converted {len(converted_docs)} documents!")
        
        # Show summary
        total_words = sum(doc['word_count'] for doc in converted_docs)
        st.info(f"📊 Total words added to knowledge base: {total_words:,}")
        
        # Show converted files
        with st.expander("📋 View converted files"):
            for doc in converted_docs:
                st.write(f"• **{doc['filename']}** - {doc['word_count']:,} words")
    
    if errors:
        st.error(f"❌ {len(errors)} files failed to convert:")
        for error in errors:
            st.write(f"• {error}")


# MAIN APP
def main():
    st.set_page_config(page_title="Simon's Personal AI Assistant", layout="wide")
    
    # Apply custom CSS
    add_custom_css()

    # Header with modern styling
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="main-header">🤖 Simon's Personal AI Assistant</h1>
            <p style="font-size: 1.2rem; color: #666; margin-top: 1rem;">
                Your intelligent companion for document analysis and insights
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []

    if 'collection' not in st.session_state:
        client = chromadb.Client()
        try:
            st.session_state.collection = client.get_collection(name="documents")
        except:
            st.session_state.collection = client.create_collection(name="documents")
            
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Initialize active tab state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0


    # Create tabs with the stored active index
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Knowledge Hub", 
        "🧠 Smart Q&A", 
        "🎯 Document Central", 
        "📈 Insights Lab"
    ])
    
    # Store the active tab when a question is submitted
    if 'question_submitted' in st.session_state and st.session_state.question_submitted:
        # st.session_state.active_tab = 1  # Index of Smart Q&A tab
        st.session_state.question_submitted = False

    with tab1:
        st.header("📁 Document Upload & Conversion")
        uploaded_files = st.file_uploader(
            "Select documents to add to your knowledge base",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word documents, and text files"
        )

        if st.button("🚀 Convert & Add to Knowledge Base", type="primary"):
            if uploaded_files:
                with st.spinner("Converting documents..."):
                    converted_docs, errors = safe_convert_files(uploaded_files)
                
                if converted_docs:
                    client = chromadb.Client()
                    collection = reset_collection(client, "documents")
                    
                    for doc in converted_docs:
                        collection = add_text_to_chromadb(doc['content'], doc['filename'])
                        st.session_state.converted_docs.append(doc)
                    
                    # Show results
                    show_conversion_results(converted_docs, errors)
            else:
                st.warning("Please select files to upload first.")

    with tab2:
        if st.session_state.converted_docs:
            # Create two columns: one for Q&A, one for history
            qa_col, history_col = st.columns([3, 2])
            
            with qa_col:
                # Add a clean container for the Q&A interface
                qa_container = st.container()
                with qa_container:
                    question, search_button, clear_button = enhanced_question_interface()
                
                # Initialize session state for Q&A
                if 'last_question' not in st.session_state:
                    st.session_state.last_question = None
                    st.session_state.last_answer = None
                    st.session_state.last_source = None
                
                # Remove any potential empty space
                st.write("")
                
                client = chromadb.Client()
                collection = client.get_collection(name="documents")
                
                answer_container = st.container()
                
                if search_button and question:
                    with st.spinner("🔍 Exploring your knowledge base..."):
                        try:
                            client = chromadb.Client()
                            collection = client.get_collection(name="documents")
                            answer, source = get_answer_with_source(collection, question)
                            
                            # Store the results in session state
                            st.session_state.last_question = question
                            st.session_state.last_answer = answer
                            st.session_state.last_source = source
                            
                            # Add to search history
                            add_to_search_history(question, answer, source)
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                # Display the last answer if it exists and has content
                with answer_container:
                    if (st.session_state.last_question is not None and 
                        st.session_state.last_answer is not None and 
                        st.session_state.last_answer.strip()):
                        st.markdown("<div class='answer-container'>", unsafe_allow_html=True)
                        st.markdown("<div class='answer-header'>💡<span>Answer</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='answer-source'>📄 Source: <strong>{st.session_state.last_source}</strong></div>", unsafe_allow_html=True)
                        
                        # Display the answer in the styled content box
                        st.markdown(f"<div class='answer-content'>{st.session_state.last_answer}</div>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            
            with history_col:
                st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)  # Spacing to align with question input
                if clear_button:
                    st.session_state.search_history = []
                    st.session_state.last_question = None
                    st.session_state.last_answer = None
                    st.session_state.last_source = None
                    st.markdown("""
                        <div style='padding: 1rem; background: linear-gradient(135deg, #4CAF50, #43a047); color: white; border-radius: 10px;'>
                            ✨ Search history has been cleared!
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show search history
                show_search_history()
        else:
            st.info("🔼 Upload some documents first to start asking questions!")

    with tab3:
        show_document_manager()
    
    with tab4:
        show_document_stats()  # Show detailed stats first
        st.markdown("---")
        show_document_analytics()  # Show charts and analytics after

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by AI*")


if __name__ == "__main__":
    main()


def show_loading_animation():
    """Display a loading animation while processing the query."""
    with st.spinner('🤔 Thinking...'):
        time.sleep(0.5)  # Brief pause for visual feedback
