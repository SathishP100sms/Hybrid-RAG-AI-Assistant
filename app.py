import streamlit as st
import os
import google.generativeai as genai
from rag_system import HybridRAG

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Hybrid RAG AI Assistant",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1em;
    }
    .sub-header {
        font-size: 1.5em;
        color: #ff7f0e;
        margin-top: 1em;
    }
    .chat-message {
        padding: 1em;
        border-radius: 10px;
        margin-bottom: 1em;
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f3e5f5;
    }
    .source-item {
        background-color: #fff3e0;
        padding: 0.5em;
        border-radius: 5px;
        margin: 0.2em 0;
    }
    .status-box {
        background-color: #e8f5e8;
        padding: 1em;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📚 Hybrid RAG AI Assistant</h1>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key = st.secrets["GEMINI_API_KEY"]
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Initialize RAG
            if "rag" not in st.session_state:
                st.session_state.rag = HybridRAG()
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            rag = st.session_state.rag
            
            # Get document count
            try:
                data = rag.vectorstore.get()
                doc_count = len(data["documents"]) if data["documents"] else 0
            except:
                doc_count = 0
            
            st.success("✅ API Key configured and RAG initialized!")
            st.markdown(f'<div class="status-box">📄 Documents in knowledge base: {doc_count}</div>', unsafe_allow_html=True)
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
                
            # Clear documents button
            if st.button("🗑️ Clear Knowledge Base"):
                # Reset vectorstore
                st.session_state.rag = HybridRAG()
                rag = st.session_state.rag
                st.success("Knowledge base cleared!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error configuring API: {str(e)}")
            st.stop()
    else:
        st.warning("Sorry the API did't load, try again later")
        st.stop()

# =========================
# MAIN CONTENT
# =========================
tab1, tab2 = st.tabs(["📂 Document Management", "💬 Chat Assistant"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📂 Upload Documents</h2>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or DOCX files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Select multiple files to upload and process"
        )
        
        if uploaded_files:
            st.write("**Uploaded Files:**")
            for file in uploaded_files:
                st.write(f"- {file.name}")
            
            if st.button("🚀 Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        os.makedirs("temp_data", exist_ok=True)
                        
                        progress_bar = st.progress(0)
                        total_files = len(uploaded_files)
                        
                        for i, file in enumerate(uploaded_files):
                            with open(f"temp_data/{file.name}", "wb") as f:
                                f.write(file.getbuffer())
                            progress_bar.progress((i + 1) / total_files)
                        
                        rag.add_documents("temp_data")
                        st.success(f"✅ {total_files} document(s) processed and added to knowledge base!")
                        
                        # Clean up temp files
                        for file in uploaded_files:
                            os.remove(f"temp_data/{file.name}")
                        
                        st.rerun()  # Refresh to update document count
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Knowledge Base Status")
        try:
            data = rag.vectorstore.get()
            doc_count = len(data["documents"]) if data["documents"] else 0
            st.metric("Total Document Chunks", doc_count)
        except:
            st.metric("Total Document Chunks", 0)
        
        if doc_count > 0:
            st.info("Knowledge base is ready for queries!")
        else:
            st.warning("No documents uploaded yet. Upload some to get started.")

with tab2:
    st.markdown('<h2 class="sub-header">💬 Ask Questions</h2>', unsafe_allow_html=True)
    
    # Display chat history using Streamlit's chat components
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📌 Sources"):
                    for source in message["sources"]:
                        st.markdown(f'<div class="source-item">{source}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = rag.query(prompt, model)
                    st.write(answer)
                    if sources:
                        with st.expander("📌 Sources"):
                            for source in sources:
                                st.markdown(f'<div class="source-item">{source}</div>', unsafe_allow_html=True)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
