# Hybrid RAG AI Assistant

A powerful AI-powered document assistant that combines Retrieval-Augmented Generation (RAG) with hybrid search capabilities. Upload your documents and chat with an AI that provides accurate answers based on your content.

## Live Demo [Hybrid RAG AI Assistant](https://hybrid-rag-ai-assistant.streamlit.app/)

## 🚀 Features

- **Hybrid Search**: Combines semantic vector search with BM25 keyword search for optimal retrieval
- **Document Support**: Upload PDF, TXT, and DOCX files
- **AI-Powered Chat**: Uses Google's Gemini 2.5 Flash model for intelligent responses
- **Source Citations**: Every answer includes references to source documents
- **Persistent Knowledge Base**: Documents are stored in ChromaDB for efficient querying
- **Modern UI**: Clean Streamlit interface with chat-like experience
- **Real-time Processing**: Progress indicators and status updates

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-mpnet-base-v2)
- **Search**: BM25 + Cross-Encoder reranking
- **Document Processing**: LangChain (PyMuPDF, python-docx)

## 📋 Prerequisites

- Python 3.8+
- Google AI API Key (from [Google AI Studio](https://makersuite.google.com/app/apikey))

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SathishP100sms/Hybrid-RAG-AI-Assistant
   cd Hybrid-RAG-Assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - The app will prompt you to enter it when you run it

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Configure Settings**
   - Enter your Google Gemini API key in the sidebar
   - The system will initialize the RAG components

3. **Upload Documents**
   - Go to the "Document Management" tab
   - Upload PDF, TXT, or DOCX files
   - Click "Process Documents" to add them to the knowledge base

4. **Start Chatting**
   - Switch to the "Chat Assistant" tab
   - Ask questions about your uploaded documents
   - View source citations for each answer

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── rag_system.py          # Hybrid RAG implementation
├── requirements.txt       # Python dependencies
├── chromadb/             # Vector database storage
├── temp_data/            # Temporary file storage (created at runtime)
└── README.md             # This file
```

## 🔍 How It Works

1. **Document Processing**: Uploaded files are split into chunks and embedded using sentence transformers
2. **Hybrid Retrieval**: Queries use both semantic similarity (via embeddings) and keyword matching (BM25)
3. **Reranking**: Results are reranked using a cross-encoder for better relevance
4. **Generation**: The top results form the context for the AI model to generate answers


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for document processing
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Google Gemini](https://ai.google.dev/) for AI generation
- [Streamlit](https://streamlit.io/) for the web interface
