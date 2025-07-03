try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from datetime import datetime
import torch
import os

# Suppress HuggingFace tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_chroma_client():
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = chromadb.Client()
    return st.session_state.chroma_client

def get_or_create_collection(client, name="documents"):
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(name=name)

# --- Add text chunks to ChromaDB ---
def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    if 'collections' not in st.session_state:
        st.session_state.collections = {}
    client = get_chroma_client()
    if collection_name not in st.session_state.collections:
        collection = get_or_create_collection(client, collection_name)
        st.session_state.collections[collection_name] = collection
    collection = st.session_state.collections[collection_name]
    for i, chunk in enumerate(chunks):
        embedding = st.session_state.embedding_model.encode(chunk).tolist()
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

# --- QA function ---
def get_answer_with_source(collection, question):
    # Debug: check collection count
    try:
        count = collection.count()
        if count == 0:
            return "No documents in the knowledge base. Please upload and add documents first.", "No source"
    except Exception as e:
        return f"ChromaDB error: {e}", "No source"
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0] if "ids" in results else ["unknown"] * len(docs)
    # Lower threshold for easier matching
    if not docs or min(distances) > 2.0:
        return "I don't have information about that topic.", "No source"
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    answer = response[0]['generated_text'].strip()
    best_source = ids[0].split('_chunk_')[0] if ids else "unknown"
    return answer, best_source

# --- FIX: Reset collection when problems occur

def reset_database():
    client = get_chroma_client()
    try:
        client.delete_collection("docs")
    except:
        pass
    return client.create_collection("docs")

client = get_chroma_client()
collection = client.get_or_create_collection("docs")

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
            if st.button("Delete", key=f"delete_{i}"):
                # Get the filename to delete
                filename_to_delete = doc['filename']
                
                # Find all chunk IDs for this document
                try:
                    # Get all documents from collection
                    all_docs = st.session_state.collection.get()
                    
                    # Find IDs that start with the filename
                    ids_to_delete = []
                    for doc_id in all_docs['ids']:
                        if doc_id.startswith(f"{filename_to_delete}_chunk_"):
                            ids_to_delete.append(doc_id)
                    
                    # Delete the specific chunks
                    if ids_to_delete:
                        st.session_state.collection.delete(ids=ids_to_delete)
                        st.success(f"Deleted {len(ids_to_delete)} chunks for {filename_to_delete}")
                    
                except Exception as e:
                    st.error(f"Error deleting from ChromaDB: {str(e)}")
                
                # Remove from session state
                st.session_state.converted_docs.pop(i)
                st.rerun()
        
        # Show preview if requested
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()


def add_to_search_history(question, answer, source):
    """Add search to history"""
    try:
        # Initialize search history if it doesn't exist
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        # Ensure parameters are strings
        question = str(question) if question is not None else "Unknown question"
        answer = str(answer) if answer is not None else "No answer"
        source = str(source) if source is not None else "No source"
        
        # Create timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add new search to beginning of list
        search_entry = {
            'question': question,
            'answer': answer,
            'source': source,
            'timestamp': timestamp
        }
        
        st.session_state.search_history.insert(0, search_entry)
        
        # Keep only last 10 searches
        if len(st.session_state.search_history) > 10:
            st.session_state.search_history = st.session_state.search_history[:10]
            
    except Exception as e:
        st.error(f"Error adding to search history: {str(e)}")
        # Initialize empty history if there's an error
        st.session_state.search_history = []

def show_search_history():
    """Display search history"""
    try:
        st.subheader("🕒 Recent Searches")
        
        if 'search_history' not in st.session_state or not st.session_state.search_history:
            st.info("No searches yet.")
            return
        
        for i, search in enumerate(st.session_state.search_history):
            try:
                # Safely get search data with defaults
                question = search.get('question', 'Unknown question')
                answer = search.get('answer', 'No answer')
                source = search.get('source', 'No source')
                timestamp = search.get('timestamp', 'Unknown time')
                
                # Create expandable section for each search
                with st.expander(f"Q: {question[:50]}... ({timestamp})"):
                    st.write("**Question:**", question)
                    st.write("**Answer:**", answer)
                    st.write("**Source:**", source)
                    
            except Exception as e:
                st.error(f"Error displaying search {i+1}: {str(e)}")
                continue
                
    except Exception as e:
        st.error(f"Error displaying search history: {str(e)}")
        # Reset search history if there's a persistent error
        st.session_state.search_history = []

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
            if st.button("Delete", key=f"delete_{i}"):
                # Get the filename to delete
                filename_to_delete = doc['filename']
                
                # Find all chunk IDs for this document
                try:
                    # Get all documents from collection
                    all_docs = st.session_state.collection.get()
                    
                    # Find IDs that start with the filename
                    ids_to_delete = []
                    for doc_id in all_docs['ids']:
                        if doc_id.startswith(f"{filename_to_delete}_chunk_"):
                            ids_to_delete.append(doc_id)
                    
                    # Delete the specific chunks
                    if ids_to_delete:
                        st.session_state.collection.delete(ids=ids_to_delete)
                        st.success(f"Deleted {len(ids_to_delete)} chunks for {filename_to_delete}")
                    
                except Exception as e:
                    st.error(f"Error deleting from ChromaDB: {str(e)}")
                
                # Remove from session state
                st.session_state.converted_docs.pop(i)
                st.rerun()
        
        # Show preview if requested
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()

# Fixed search section for main function
def handle_search_section():
    """Handle the search section in tab2"""
    st.header("Ask Questions About Travel, Cultures, and Food")
    if st.session_state.converted_docs:
        question, search_button, clear_button = enhanced_question_interface()
        
        if search_button and question:
            try:
                # Get answer from documents
                answer, source = get_answer_with_source(st.session_state.collection, question)
                
                # Display results
                st.markdown("### 💡 Answer")
                st.write(answer)
                st.info(f"📄 Source: {source}")
                
                # Add to search history with error handling
                try:
                    add_to_search_history(question, answer, source)
                except Exception as e:
                    st.warning(f"Could not save to search history: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
                st.info("Please try rephrasing your question or check if documents are properly loaded.")
        
        if clear_button:
            try:
                st.session_state.search_history = []
                st.success("Search history cleared!")
            except Exception as e:
                st.error(f"Error clearing history: {str(e)}")
        
        # Show search history
        if st.session_state.get('search_history', []):
            try:
                show_search_history()
            except Exception as e:
                st.error(f"Error displaying search history: {str(e)}")
    else:
        st.info("🔼 Upload some documents first to start asking questions!")

# --- Custom CSS for professional look ---
def add_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@700&display=swap');
    html, body {
        height: 100% !important;
        min-height: 100vh !important;
        background: #B7D9B1 !important; /* RAL 6019 pastel green */
        box-sizing: border-box !important;
    }
    [data-testid="stAppViewContainer"], .stApp, .block-container {
        background: transparent !important;
        box-shadow: none !important;
    }
    .main-header {
        font-family: 'Baloo 2', cursive, sans-serif;
        font-size: 2.7rem;
        color: #3A6B35;
        text-align: center;
        margin-bottom: 0;
        margin-top: 0.5rem;
        padding: 1.2rem;
        background: rgba(255,255,255,0.97);
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(58,107,53,0.10);
        letter-spacing: 1px;
        border: 2.5px solid #3A6B35;
        display: block;
    }
    .main-header-spacer {
        height: 3.5rem;
        width: 100%;
        display: block;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #B7D9B1 0%, #3A6B35 100%);
        color: #222;
        border: 2px solid #3A6B35;
        box-shadow: 0 2px 8px rgba(58,107,53,0.10);
        transition: background 0.3s;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #3A6B35 0%, #B7D9B1 100%);
        color: #fff;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.92);
        border-radius: 14px;
        margin-bottom: 1.2rem;
        border: none;
    }
    /* Remove custom selected tab color, revert to Streamlit default */
    .stExpanderHeader {
        font-size: 1.1rem;
        color: #3A6B35;
    }
    .st-bb, .st-cq, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
        border-radius: 16px !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Enhanced question interface ---
def enhanced_question_interface():
    st.subheader("💬 Ask About Travel, Cultures, and Food")
    with st.expander("💡 Example questions you can ask"):
        st.write("""
        • What are the most famous dishes in [country/city]?
        • Describe unique cultural traditions in [place].
        • What landmarks should I visit in [destination]?
        • Compare the food culture between [place1] and [place2].
        • What festivals are celebrated in [country/city]?
        """)
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What are the best foods to try in Italy?"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        search_button = st.button("🔍 Search Documents", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear History")
    return question, search_button, clear_button

# --- Enhanced main function ---
def main():
    add_custom_css()
    st.markdown('<h1 class="main-header">🌍✈️ Travel, Cultures & Food Knowledge Hub 🍜🍕🥑🍣</h1>', unsafe_allow_html=True)
    st.markdown('<div class="main-header-spacer"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; font-size:1.2rem; margin-bottom:1.5rem;'>
        Welcome!<br>
        <b>Upload travel guides, cultural stories, and food adventures.<br>
        Ask questions, discover new places, and celebrate the world's diversity! 🌎🍲🕌🍉</b>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'collection' not in st.session_state:
        client = get_chroma_client()
        try:
            st.session_state.collection = get_or_create_collection(client, "documents")
        except Exception:
            st.session_state.collection = client.create_collection(name="documents")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "❓ Ask Questions", "📋 Manage"])
    
    with tab1:
        st.header("Upload & Convert Travel, Culture, and Food Documents")
        uploaded_files = st.file_uploader(
            "Choose files (PDF, DOC, DOCX, TXT)",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("Convert & Add"):
            if uploaded_files:
                converted_docs = []
                errors = []
                
                for uploaded in uploaded_files:
                    file_ext = Path(uploaded.name).suffix.lower()
                    
                    if len(uploaded.getvalue()) > 10 * 1024 * 1024:
                        errors.append(f"{uploaded.name}: File too large (max 10MB)")
                        continue
                    
                    if file_ext not in ['.pdf', '.doc', '.docx', '.txt']:
                        errors.append(f"{uploaded.name}: Unsupported file type")
                        continue
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        md = convert_to_markdown(tmp_path)
                        if len(md.strip()) < 10:
                            errors.append(f"{uploaded.name}: File appears to be empty or corrupted")
                            continue
                        
                        st.session_state.converted_docs.append({
                            "filename": uploaded.name,
                            "content": md
                        })
                        
                        add_text_to_chromadb(md, uploaded.name)
                        converted_docs.append({
                            'filename': uploaded.name,
                            'word_count': len(md.split())
                        })
                        st.success(f"Converted {uploaded.name} successfully.")
                        
                    except Exception as e:
                        errors.append(f"{uploaded.name}: {str(e)}")
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                
                if converted_docs:
                    total_words = sum(doc['word_count'] for doc in converted_docs)
                    st.info(f"📊 Total words added: {total_words:,}")
                    with st.expander("📋 View converted files"):
                        for doc in converted_docs:
                            st.write(f"• **{doc['filename']}** - {doc['word_count']:,} words")
                
                if errors:
                    st.error(f"❌ {len(errors)} files failed to convert:")
                    for error in errors:
                        st.write(f"• {error}")
            else:
                st.warning("Please select files to upload first.")
    
    with tab2:
        # Use the new search section handler
        handle_search_section()
    
    with tab3:
        show_document_manager()
    
    with st.expander("About this Travel & Culture Q&A System"):
        st.write("""
        I created this app to answer questions about:
        - 🍲 Traditional foods and how they reflect culture
        - 🎉 Global festivals and their social meaning
        - 🗣️ Languages and worldviews
        - 🌱 Sustainable tourism practices
        - 🙏 Cultural etiquette around the world

        ✨ Try asking about specific dishes, customs, festivals, or etiquette rules in different countries!
        """)
    
    st.markdown("---")
    st.markdown("<div style='text-align:center; font-size:1.1rem;'>Made with ❤️ for explorers, foodies, and culture lovers!</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

