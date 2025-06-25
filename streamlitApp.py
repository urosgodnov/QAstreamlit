import streamlit as st
import tempfile
import os
import chromadb
from chromadb.utils import embedding_functions
from docling.document_converter import DocumentConverter
import uuid
import pandas as pd
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SpacyTextSplitter
)

# Page config
st.set_page_config(
    page_title="Document QA Demo",
    layout="wide"
)

st.title("📄 Document Q&A with ChromaDB")

# Sidebar options
st.sidebar.header("Chunking Options")
strategy = st.sidebar.selectbox(
    "Strategy", 
    ["Recursive Character", "Character", "Spacy"]
)

chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)

# File uploader
uploaded_file = st.file_uploader(
    "Upload document",
    type=["pdf", "docx"]
)

def chunk_text(text, strategy, chunk_size, overlap):
    """Split text into chunks using LangChain splitters"""
    
    if strategy == "Recursive Character":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    elif strategy == "Character":
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separator="\n\n"
        )
    
    elif strategy == "Spacy":
        try:
            splitter = SpacyTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len
            )
        except Exception as e:
            st.warning(f"Spacy not available: {e}. Using Recursive Character instead.")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len
            )
    
    chunks = splitter.split_text(text)
    return chunks

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_filename = tmp_file.name
    
    try:
        # Convert document
        with st.spinner('Converting document...'):
            conv = DocumentConverter()
            result = conv.convert(temp_filename)
            text = result.document.export_to_markdown()
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        st.success('Document converted successfully!')
        st.info(f"Extracted {len(text)} characters")
        
        # Chunk the text
        with st.spinner('Chunking text...'):
            chunks = chunk_text(text, strategy, chunk_size, chunk_overlap)
        
        st.info(f"Created {len(chunks)} chunks")
        
        if chunks:
            # Set up embeddings
            with st.spinner('Setting up ChromaDB...'):
                embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                
                # Create client and collection
                client = chromadb.Client()
                
                # Generate unique collection name to avoid conflicts
                collection_name = f"doc_collection_{uuid.uuid4().hex[:8]}"
                
                collection = client.create_collection(
                    collection_name,
                    embedding_function=embedding_model
                )
                
                # Add chunks with metadata
                ids = [f"chunk_{i}" for i in range(len(chunks))]
                metadatas = [{"source": uploaded_file.name, "chunk": i, "strategy": strategy} 
                           for i in range(len(chunks))]
                
                collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            
            st.success(f'Added {len(chunks)} chunks to ChromaDB!')
            
            # Show preview of chunks
            st.subheader("Chunk Preview")
            preview_chunks = min(3, len(chunks))
            
            for i in range(preview_chunks):
                with st.expander(f"Chunk {i+1} ({len(chunks[i])} chars)"):
                    st.text(chunks[i][:500] + "..." if len(chunks[i]) > 500 else chunks[i])
            
            # Query interface
            st.subheader("Query Your Document")
            query = st.text_input("Ask a question:")
            
            if query:
                with st.spinner('Searching...'):
                    results = collection.query(
                        query_texts=[query],
                        n_results=3
                    )
                
                st.subheader("Search Results")
                
                # Display results
                answers = results["documents"][0]
                distances = results["distances"][0]
                metadatas = results["metadatas"][0]
                
                for ans, score, meta in zip(answers, distances, metadatas):
                    st.write(f"**Answer:** {ans}")
                    st.write(f"*Source:* {meta['source']}, "
                            f"chunk {meta['chunk']} "
                            f"(distance: {score:.4f})")
                    st.write("---")
                
                # Create download button
                if answers:
                    # Prepare data
                    out_data = [{
                        **meta, 
                        "text": ans,
                        "distance": score
                    } for ans, meta, score in zip(answers, metadatas, distances)]
                    
                    # Convert to CSV
                    df = pd.DataFrame(out_data)
                    csv = df.to_csv(index=False).encode('utf-8')
                    
                    # Download button
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "results.csv",
                        "text/csv"
                    )
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.unlink(temp_filename)
        st.error(f"Error: {str(e)}")
        st.error("Make sure you have installed all dependencies: `pip install docling chromadb sentence-transformers langchain spacy pandas`")
        st.error("For Spacy: `python -m spacy download en_core_web_sm`")

else:
    st.info("Upload a PDF or DOCX file to get started")
    st.markdown("""
    **Features:**
    - Document conversion with Docling
    - Multiple chunking strategies
    - ChromaDB vector storage
    - Semantic search capabilities
    
    **Requirements:**
    ```bash
    pip install docling chromadb sentence-transformers streamlit langchain spacy pandas
    python -m spacy download en_core_web_sm
    ```
    """)