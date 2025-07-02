import streamlit as st
from pathlib import Path
import tempfile
from datetime import datetime  # Add this for the search history feature
import time  # For loading animations
import base64  # For embedding images

# NameError handling - create a debug function to catch issues
def debug_log(message):
    """Log debug messages"""
    with open("streamlit_debug.log", "a") as f:
        f.write(f"{datetime.now()}: {message}\n")

# Import libraries with error handling
try:
    import chromadb
    debug_log("ChromaDB imported successfully")
except Exception as e:
    debug_log(f"Error importing ChromaDB: {str(e)}")
    st.error(f"Error importing ChromaDB: {str(e)}. Please make sure it's installed with 'pip install chromadb'")

try:
    from transformers import pipeline
    debug_log("Transformers pipeline imported successfully")
except Exception as e:
    debug_log(f"Error importing Transformers: {str(e)}")
    st.error(f"Error importing Transformers: {str(e)}. Please make sure it's installed.")

try:
    from sentence_transformers import SentenceTransformer
    debug_log("SentenceTransformer imported successfully")
except Exception as e:
    debug_log(f"Error importing SentenceTransformer: {str(e)}")
    st.error(f"Error importing SentenceTransformer: {str(e)}. Please make sure it's installed.")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    debug_log("Langchain imported successfully")
except Exception as e:
    debug_log(f"Error importing Langchain: {str(e)}")
    st.error(f"Error importing Langchain: {str(e)}. Please make sure it's installed.")

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
    debug_log("Docling imported successfully")
except Exception as e:
    debug_log(f"Error importing Docling: {str(e)}")
    st.error(f"Error importing Docling: {str(e)}. Please make sure it's installed.")
    debug_log(f"Error importing Transformers: {str(e)}")
    st.error(f"Error importing Transformers: {str(e)}. Please make sure it's installed.")


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
def get_answer(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]

    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic in my documents."

    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with \"I don't know.\" Do not add information from outside the context.

Answer:"""

    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    return response[0]['generated_text'].strip()


# FEATURE 1: Show which document answered the question
def get_answer_with_source(collection, question):
    """Enhanced answer function that shows source document"""
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]  # This tells us which document
    
    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic.", "No source"
    
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    prompt = f"""Context information:
{context}

Question: {question}

Answer:"""
    
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    
    answer = response[0]['generated_text'].strip()
    
    # Extract source from best matching document
    best_source = ids[0].split('_chunk_')[0]  # Get filename from ID
    
    return answer, best_source

# FEATURE 2: Document manager with delete option
def show_document_manager():
    """Display document manager interface"""
    st.subheader("📋 Manage Documents")
    
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
                st.session_state.collection = setup_documents()
                add_docs_to_database(st.session_state.collection, st.session_state.converted_docs)
                st.rerun()
        
        # Show preview if requested
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()

# FEATURE 3: Search history
def add_to_search_history(question, answer, source):
    """Add search to history"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Add new search to beginning of list
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    
    # Keep only last 10 searches
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    """Display search history with a modern chat-like interface"""
    st.subheader("🕒 Recent Conversations")
    
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("No conversations yet.")
        return
    
    # Add a styled container for the chat history
    st.markdown("""
    <div style="background-color:rgba(255,255,255,0.7);border-radius:15px;padding:15px;margin:10px 0;box-shadow:0 4px 10px rgba(0,0,0,0.05);">
        <h4 style="margin-bottom:15px;color:var(--primary);">Your Conversation History</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Display each conversation as a chat bubble
    for i, search in enumerate(st.session_state.search_history):
        col1, col2 = st.columns([1, 5])
        
        with col1:
            st.markdown(f"""
            <div style="background:var(--gradient-2);width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:10px;">
                <span style="color:white;font-weight:bold;">{i+1}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            with st.expander(f"{search['question'][:50]}..." if len(search['question']) > 50 else search['question'], expanded=False):
                # User question
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div style="font-weight:bold;margin-bottom:5px;">You asked:</div>
                    {search['question']}
                    <div style="font-size:0.8rem;text-align:right;margin-top:5px;color:#666;">{search['timestamp']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI response
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div style="font-weight:bold;margin-bottom:5px;">Simon answered:</div>
                    {search['answer']}
                    <div style="font-size:0.8rem;margin-top:5px;">Source: {search['source']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Add a small divider between conversations
        if i < len(st.session_state.search_history) - 1:
            st.markdown('<hr style="margin:10px 0;border:none;border-top:1px dashed #ddd;">', unsafe_allow_html=True)

# FEATURE 4: Document statistics
def show_document_stats():
    """Show statistics about uploaded documents"""
    st.subheader("📊 Document Statistics")
    
    if not st.session_state.converted_docs:
        st.info("No documents to analyze.")
        return
    
    # Calculate stats
    total_docs = len(st.session_state.converted_docs)
    total_words = sum(len(doc['content'].split()) for doc in st.session_state.converted_docs)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", total_docs)
    
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        st.metric("Average Words/Doc", f"{avg_words:,}")
    
    # Show breakdown by file type
    file_types = {}
    for doc in st.session_state.converted_docs:
        ext = Path(doc['filename']).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    st.write("**File Types:**")
    for ext, count in file_types.items():
        st.write(f"• {ext}: {count} files")

# FEATURE 5: Enhanced UI with tabs
def create_tabbed_interface():
    """Create a tabbed interface for better organization"""
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "❓ Ask Questions", "📋 Manage", "📊 Stats"])
    
    with tab1:
        st.header("Upload & Convert Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            key="tab_uploader"
        )
        
        if st.button("Convert & Add", key="convert_add_btn"):
            if uploaded_files:
                if 'converted_docs' not in st.session_state:
                    st.session_state.converted_docs = []
                
                converted_docs = []
                for file in uploaded_files:
                    suffix = Path(file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(file.getvalue())
                        temp_file_path = temp_file.name

                    text = convert_to_markdown(temp_file_path)
                    converted_docs.append({
                        'filename': file.name,
                        'content': text
                    })
                
                if converted_docs:
                    # Create or get collection
                    client = chromadb.Client()
                    try:
                        collection = client.get_collection(name="documents")
                    except:
                        collection = client.create_collection(name="documents")
                    
                    st.session_state.collection = collection
                    
                    # Add documents to database
                    for doc in converted_docs:
                        add_text_to_chromadb(doc['content'], doc['filename'])
                    
                    st.session_state.converted_docs.extend(converted_docs)
                    st.success(f"Added {len(converted_docs)} documents!")
    
    with tab2:
        st.header("Ask Questions")
        
        if 'converted_docs' in st.session_state and st.session_state.converted_docs:
            question = st.text_input("Your question:", key="tab_question")
            
            if st.button("Get Answer", key="tab_get_answer"):
                if question:
                    client = chromadb.Client()
                    try:
                        collection = client.get_collection(name="documents")
                        answer, source = get_answer_with_source(collection, question)
                        
                        st.write("**Answer:**")
                        st.write(answer)
                        st.write(f"**Source:** {source}")
                        
                        # Add to history
                        add_to_search_history(question, answer, source)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Upload documents first!")
        
        # Show recent searches
        show_search_history()
    
    with tab3:
        st.header("Manage Documents")
        show_document_manager()
    
    with tab4:
        st.header("Document Statistics")
        show_document_stats()

# Helper function to setup the database structure
def setup_documents():
    """Initialize the ChromaDB collection"""
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="documents")
    except:
        collection = client.create_collection(name="documents")
    return collection

# Helper to add documents to the database
def add_docs_to_database(collection, docs):
    """Add documents to the ChromaDB collection"""
    count = 0
    for doc in docs:
        add_text_to_chromadb(doc['content'], doc['filename'])
        count += 1
    return count

# Function to load and display the AI Assistant logo
def display_logo():
    """Display a generic AI Assistant logo"""
    # Base64 encoded generic AI robot logo
    logo_base64 = """
    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
    """
    
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{}" alt="Simon's AI Assistant" width="100">
    </div>
    """.format(logo_base64), unsafe_allow_html=True)

# Custom CSS for better appearance with a modern gradient theme
def add_custom_css():
    """Add custom styling with a modern purple-to-teal gradient theme"""
    st.markdown("""
    <style>
    /* Modern Vibrant Gradient Colors */
    :root {
        --primary: #7e22ce;       /* Vibrant purple */
        --primary-dark: #581c87;  /* Dark purple */
        --secondary: #2563eb;     /* Bright blue */
        --secondary-dark: #1d4ed8;/* Dark blue */
        --accent: #0d9488;        /* Teal */
        --accent-light: #10b981;  /* Green */
        --light-bg: #f8f9fa;
        --dark-text: #212529;
        --light-text: #ffffff;
        --gradient-1: linear-gradient(135deg, var(--primary), var(--secondary));
        --gradient-2: linear-gradient(135deg, var(--secondary), var(--accent-light));
        --gradient-3: linear-gradient(135deg, var(--primary-dark), var(--accent));
    }
    
    /* Background with dynamic geometric pattern */
    .stApp {
        background: 
            radial-gradient(circle at 10% 20%, rgba(123, 44, 191, 0.1) 0%, rgba(123, 44, 191, 0.05) 40%),
            radial-gradient(circle at 90% 80%, rgba(58, 134, 255, 0.1) 0%, rgba(58, 134, 255, 0.05) 40%),
            radial-gradient(circle at 80% 10%, rgba(6, 214, 160, 0.1) 0%, rgba(6, 214, 160, 0.05) 40%),
            url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%239C92AC' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
        background-color: var(--light-bg);
        position: relative;
        overflow: hidden;
    }
    
    /* Add animated floating shapes */
    .stApp::before {
        content: "";
        position: absolute;
        top: -50px;
        left: -50px;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: linear-gradient(45deg, var(--primary), var(--secondary));
        opacity: 0.1;
        filter: blur(30px);
        animation: float-1 15s infinite ease-in-out;
        z-index: -1;
    }
    
    .stApp::after {
        content: "";
        position: absolute;
        bottom: -100px;
        right: -100px;
        width: 200px;
        height: 200px;
        border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
        background: linear-gradient(45deg, var(--accent), var(--secondary));
        opacity: 0.1;
        filter: blur(30px);
        animation: float-2 20s infinite ease-in-out;
        z-index: -1;
    }
    
    @keyframes float-1 {
        0% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(100px, 100px) rotate(180deg); }
        100% { transform: translate(0, 0) rotate(360deg); }
    }
    
    @keyframes float-2 {
        0% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(-100px, -50px) rotate(-180deg); }
        100% { transform: translate(0, 0) rotate(-360deg); }
    }
    
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 10px;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.1);
        font-weight: bold;
    }
    
    .success-box {
        padding: 1rem;
        background-color: rgba(6, 214, 160, 0.2);
        border: 1px solid var(--accent);
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .info-box {
        padding: 1rem;
        background-color: rgba(58, 134, 255, 0.2);
        border: 1px solid var(--secondary);
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 50px;
        height: 3rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    /* Gradient primary buttons */
    .stButton > button[data-baseweb="button"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: var(--light-text);
        border: none;
    }
    
    .stButton > button[data-baseweb="button"]:hover {
        background: linear-gradient(135deg, var(--primary-dark), var(--secondary-dark));
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s ease;
        border-top: 4px solid var(--accent);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.7);
        border-radius: 10px 10px 0 0;
        padding: 10px 16px;
        border: 1px solid #e6e6e6;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Chat banner style */
    .chat-banner {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Accent divider */
    .accent-divider {
        background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
        height: 4px;
        border-radius: 2px;
        margin: 15px 0;
    }
    
    /* Document cards */
    .doc-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
        border-left: none;
        border-top: 4px solid var(--secondary);
    }
    
    .doc-card:hover {
        transform: translateY(-3px);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background-color: rgba(255,255,255,0.8);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(0,0,0,0.1);
        font-size: 14px;
        color: #666;
    }
    
    /* Chat message bubbles */
    .chat-message {
        margin: 10px 0;
        padding: 15px;
        border-radius: 15px;
        max-width: 80%;
        position: relative;
    }
    
    .user-message {
        background-color: rgba(123, 44, 191, 0.1);
        border-left: 3px solid var(--primary);
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    
    .bot-message {
        background-color: rgba(58, 134, 255, 0.1);
        border-left: 3px solid var(--secondary);
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# AI Assistant Banner with slogan
def add_banner():
    """Add modern AI Assistant banner with slogan"""
    st.markdown("""
    <div class="chat-banner">
        <h2 style="margin:0;padding:0;font-size:1.5rem;">🤖 SIMON'S AI ASSISTANT 🤖</h2>
        <p style="margin:10px 0;font-style:italic;font-size:1rem;">Your intelligent companion for document analysis</p>
        <div style="display:flex;justify-content:center;gap:10px;margin-top:15px;">
            <span style="padding:5px 10px;background:rgba(255,255,255,0.2);border-radius:15px;font-size:0.8rem;">Smart</span>
            <span style="padding:5px 10px;background:rgba(255,255,255,0.2);border-radius:15px;font-size:0.8rem;">Fast</span>
            <span style="padding:5px 10px;background:rgba(255,255,255,0.2);border-radius:15px;font-size:0.8rem;">Helpful</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Custom footer with modern theme
def add_footer():
    """Add a custom footer with modern style"""
    st.markdown("""
    <div class="footer">
        <div class="accent-divider"></div>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:20px;margin:10px 0;">
            <div style="display:flex;align-items:center;gap:5px;">
                <span style="font-size:18px;">🤖</span> Simon's AI Assistant
            </div>
            <div style="display:flex;align-items:center;gap:5px;">
                <span style="font-size:18px;">💬</span> Powered by AI
            </div>
            <div style="display:flex;align-items:center;gap:5px;">
                <span style="font-size:18px;">�</span> Fast & Intelligent
            </div>
        </div>
        <p style="margin-top:10px;font-style:italic;color:#666;">Making documents smarter, one question at a time!</p>
    </div>
    """, unsafe_allow_html=True)

# App health check
def check_app_health():
    """Check if all components are working"""
    issues = []
    
    # Check session state
    required_keys = ['converted_docs', 'collection']
    for key in required_keys:
        if key not in st.session_state:
            issues.append(f"Missing session state: {key}")
    
    # Check ChromaDB
    try:
        if st.session_state.get('collection'):
            st.session_state.collection.count()
    except Exception as e:
        issues.append(f"Database issue: {e}")
    
    # Check AI model
    try:
        pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception as e:
        issues.append(f"AI model issue: {e}")
    
    return issues

# NameError handling - create a debug function to catch issues
def debug_log(message):
    """Log debug messages"""
    with open("streamlit_debug.log", "a") as f:
        f.write(f"{datetime.now()}: {message}\n")

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

# Better user feedback
def show_conversion_results(converted_docs, errors):
    """Display conversion results with good UX"""
    
    if converted_docs:
        st.markdown('<div class="success-box">✅ Successfully converted ' + 
                    f'{len(converted_docs)} documents!</div>', unsafe_allow_html=True)
        
        # Show summary
        total_words = sum(doc['word_count'] for doc in converted_docs)
        st.markdown(f'<div class="info-box">📊 Total words added to knowledge base: {total_words:,}</div>', 
                   unsafe_allow_html=True)
        
        # Show converted files
        with st.expander("📋 View converted files"):
            for doc in converted_docs:
                st.markdown(f'<div class="doc-card">📄 <b>{doc["filename"]}</b> - {doc["word_count"]:,} words</div>', 
                           unsafe_allow_html=True)
    
    if errors:
        st.error(f"❌ {len(errors)} files failed to convert:")
        for error in errors:
            st.write(f"• {error}")

# Better question interface
def enhanced_question_interface():
    """Professional question asking interface with modern chat-like theme"""
    
    st.subheader("💬 Ask Your Question")
    
    # Provide example questions
    with st.expander("💡 Example questions you can ask"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            • What are the main topics covered in these documents?
            • Summarize the key points from [document name]
            • What does the document say about [specific topic]?
            • Compare information between documents
            """)
        with col2:
            st.markdown("""
            • Find specific data or statistics
            • What is the main conclusion in this document?
            • When was [specific event] mentioned?
            • How many references are there to [topic]?
            """)
    
    # Question input with stylish container
    st.markdown("""
    <div style="background-color:white;padding:15px;border-radius:15px;box-shadow:0 4px 8px rgba(0,0,0,0.05);">
        <div style="display:flex;align-items:center;margin-bottom:10px;">
            <div style="background:var(--gradient-1);width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:10px;">
                <span style="color:white;font-weight:bold;">?</span>
            </div>
            <span style="font-weight:bold;">Your Question</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Question input with suggestions
    question = st.text_input(
        "",
        placeholder="e.g., What are the main findings in the document?",
        key="enhanced_question"
    )
    
    # Two-column layout for buttons with enhanced styling
    col1, col2 = st.columns([1, 1])
    
    with col1:
        search_button = st.button("🔍 Search Documents", type="primary", key="enhanced_search")
    
    with col2:
        clear_button = st.button("🗑️ Clear History", type="secondary", key="enhanced_clear")
    
    # Add a decorative divider
    st.markdown('<div class="accent-divider"></div>', unsafe_allow_html=True)
    
    return question, search_button, clear_button

# Loading animations
def show_loading_animation(text="Processing..."):
    """Show professional loading animation"""
    with st.spinner(text):
        time.sleep(0.5)  # Brief pause for better UX

# FEATURE 6: AI Chatbot Personality
def get_ai_personality():
    """Get the AI's personality based on session state"""
    personalities = {
        "helpful": {
            "name": "Helpful Assistant",
            "description": "Focused on providing accurate and informative answers",
            "emoji": "🧠",
            "color": "var(--secondary)"
        },
        "friendly": {
            "name": "Friendly Guide",
            "description": "Warm and conversational with simple explanations",
            "emoji": "😊",
            "color": "var(--accent-light)"
        },
        "expert": {
            "name": "Document Expert",
            "description": "Provides detailed technical analysis and insights",
            "emoji": "🔍",
            "color": "var(--primary)"
        }
    }
    
    # Get current personality or set default
    if 'ai_personality' not in st.session_state:
        st.session_state.ai_personality = "helpful"
    
    return personalities[st.session_state.ai_personality]

# FEATURE 7: Daily tips
def show_daily_tip():
    """Show a daily tip about using the assistant"""
    tips = [
        "Upload multiple documents at once to compare information across them!",
        "You can ask follow-up questions to get more specific information.",
        "Try asking for summaries to get quick overviews of long documents.",
        "Clear your search history periodically to start fresh conversations.",
        "Use specific keywords from your documents in your questions for better results.",
        "Ask about relationships between concepts mentioned in different documents.",
        "You can delete documents you no longer need from the Manage tab.",
        "Check the Analytics tab to see statistics about your document collection.",
        "Upload different document types to build a comprehensive knowledge base.",
        "Ask questions that combine information from multiple documents for deeper insights."
    ]
    
    # Get a consistent tip for the day
    import datetime
    today = datetime.date.today().day
    tip_index = today % len(tips)
    
    st.markdown(f"""
    <div style="background: var(--gradient-2);padding:15px;border-radius:10px;margin:15px 0;color:white;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <span style="font-size:24px;">💡</span>
            <strong>TIP OF THE DAY</strong>
        </div>
        <p style="margin:0;">{tips[tip_index]}</p>
    </div>
    """, unsafe_allow_html=True)

# FEATURE 8: Conversation starters
def show_conversation_starters():
    """Display suggested conversation starters"""
    starters = [
        "Summarize the main points of all my documents",
        "What are the key themes across my documents?",
        "Find all mentions of important dates in my documents",
        "Compare information between different documents",
        "What conclusions can be drawn from these documents?",
        "Identify the most frequent topics in my collection"
    ]
    
    st.markdown("<strong>🚀 Try asking:</strong>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, starter in enumerate(starters):
        col = cols[i % 2]
        with col:
            if st.button(starter, key=f"starter_{i}"):
                # Set the question in the text input
                st.session_state.enhanced_question = starter
                # Force a rerun to update the UI
                st.rerun()

# FEATURE 9: Personality selector
def show_personality_selector():
    """Show a selector for the AI assistant's personality"""
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🎭 Assistant Personality")
    
    personality = get_ai_personality()
    
    # Display current personality
    st.sidebar.markdown(f"""
    <div style="background:linear-gradient(45deg, {personality['color']}, {personality['color']}66);
        padding:10px;border-radius:10px;margin:10px 0;color:white;">
        <div style="font-size:24px;text-align:center;">{personality['emoji']}</div>
        <div style="font-weight:bold;text-align:center;">{personality['name']}</div>
        <div style="font-size:0.8rem;text-align:center;">{personality['description']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Personality selector
    selected = st.sidebar.radio(
        "Choose a personality:",
        ["helpful", "friendly", "expert"],
        format_func=lambda x: {"helpful": "🧠 Helpful Assistant", 
                              "friendly": "😊 Friendly Guide", 
                              "expert": "🔍 Document Expert"}[x],
        index=["helpful", "friendly", "expert"].index(st.session_state.ai_personality),
        key="personality_selector"
    )
    
    if selected != st.session_state.ai_personality:
        st.session_state.ai_personality = selected
        st.sidebar.success(f"Personality changed to {selected.title()}!")
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# FEATURE 10: Export conversation history
def show_export_option():
    """Show option to export conversation history"""
    if 'search_history' in st.session_state and st.session_state.search_history:
        st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.sidebar.markdown("### 📤 Export Conversation")
        
        if st.sidebar.button("Export History as Text", key="export_btn"):
            # Generate export text
            export_text = "# Simon's AI Assistant - Conversation History\n\n"
            export_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for i, search in enumerate(st.session_state.search_history):
                export_text += f"## Conversation {i+1} ({search['timestamp']})\n\n"
                export_text += f"**Q:** {search['question']}\n\n"
                export_text += f"**A:** {search['answer']}\n\n"
                export_text += f"**Source:** {search['source']}\n\n"
                export_text += "---\n\n"
            
            # Create download link
            b64 = base64.b64encode(export_text.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="conversation_history.txt">Download Conversation History</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)
            st.sidebar.success("Ready to download!")
        
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

# MAIN APP
def main():
    try:
        debug_log("Starting app...")
        # Apply custom CSS for modern theme
        add_custom_css()
        
        # Configure sidebar
        st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        display_logo()
        st.sidebar.markdown("### 🤖 Simon's AI Assistant")
        st.sidebar.markdown("*Your intelligent companion for documents*")
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        # Show AI personality selector in sidebar
        show_personality_selector()
        
        # Quick tips in sidebar
        st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.sidebar.markdown("### 💡 Quick Tips")
        st.sidebar.info("📄 Upload any document type")
        st.sidebar.info("❓ Ask specific questions")
        st.sidebar.info("📊 View document analytics")
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        # Add export option to sidebar
        show_export_option()
        
        # App health check in sidebar (collapsible)
        health_issues = check_app_health()
        if health_issues:
            st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            with st.sidebar.expander("⚠️ System Status"):
                for issue in health_issues:
                    st.warning(issue)
            st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        # Main header
        st.markdown('<h1 class="main-header">🤖 Simon\'s AI Assistant</h1>', unsafe_allow_html=True)
        
        # AI Assistant banner
        add_banner()
        
        # Show daily tip 
        show_daily_tip()
        
        # Brief description
        st.markdown("Upload documents, convert them automatically, and ask intelligent questions about your content!")
        
        # Initialize session state
        if 'converted_docs' not in st.session_state:
            st.session_state.converted_docs = []
        
        if 'collection' not in st.session_state:
            st.session_state.collection = setup_documents()
        
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if 'ai_personality' not in st.session_state:
            st.session_state.ai_personality = "helpful"
        
        # Create enhanced tabbed interface
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "❓ Questions", "📋 Manage", "📊 Analytics"])
        
        with tab1:
            st.header("📁 Document Upload & Conversion")
            
            # File uploader with better description
            uploaded_files = st.file_uploader(
                "Select documents to add to your knowledge base",
                type=["pdf", "doc", "docx", "txt"],
                accept_multiple_files=True,
                key="enhanced_uploader",
                help="Supported formats: PDF, Word documents, and text files"
            )
            
            # Improved conversion button
            if st.button("🚀 Convert & Add to Knowledge Base", type="primary", key="enhanced_convert_btn"):
                if uploaded_files:
                    show_loading_animation("Converting your documents...")
                    converted_docs, errors = safe_convert_files(uploaded_files)
                    
                    if converted_docs:
                        # Add to database
                        num_added = add_docs_to_database(st.session_state.collection, converted_docs)
                        st.session_state.converted_docs.extend(converted_docs)
                    
                    # Show results
                    show_conversion_results(converted_docs, errors)
                else:
                    st.warning("Please select files to upload first.")
        
        with tab2:
            st.header("❓ Ask Questions")
            
            if st.session_state.converted_docs:
                # Show conversation starters
                show_conversation_starters()
                
                # Enhanced question interface
                question, search_button, clear_button = enhanced_question_interface()
                
                if search_button and question:
                    show_loading_animation("🔍 Searching through your documents...")
                    try:
                        answer, source = get_answer_with_source(st.session_state.collection, question)
                        
                        # Get current personality
                        personality = get_ai_personality()
                        
                        # Display answer with better formatting and personality
                        st.markdown(f"### {personality['emoji']} Answer from {personality['name']}")
                        st.markdown(f'<div class="info-box" style="border-left: 4px solid {personality["color"]};">{answer}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="doc-card">📄 Source: {source}</div>', unsafe_allow_html=True)
                        
                        # Add to history
                        add_to_search_history(question, answer, source)
                    except Exception as e:
                        st.error(f"Error retrieving answer: {e}")
                        debug_log(f"Error in question answering: {str(e)}")
                
                if clear_button:
                    st.session_state.search_history = []
                    st.success("Search history cleared!")
                
                # Show search history
                if st.session_state.search_history:
                    show_search_history()
            
            else:
                st.info("🔼 Upload some documents first to start asking questions!")
        
        with tab3:
            st.header("Manage Documents")
            show_document_manager()
        
        with tab4:
            st.header("Document Statistics")
            show_document_stats()
        
        # Footer
        add_footer()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        debug_log(f"Critical error in main app: {str(e)}")

if __name__ == "__main__":
    main()
