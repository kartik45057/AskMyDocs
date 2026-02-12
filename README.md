📚 AskMyDocs

🧠 Conversational RAG System with LangGraph, ChromaDB & LLM Orchestration

  A production-style Retrieval-Augmented Generation (RAG) system that enables context-aware question answering over PDF documents using graph-based execution and persistent conversational memory.



🚀 Overview

AskMyDocs is a modular, stateful document question-answering system built using modern GenAI architecture principles.

It allows users to:
=> Upload a (pdf, doc, text) document
=> Index it into a vector database
=> Ask natural language questions
=> Receive context-grounded answers
=> Maintain multi-turn conversational memory

This project demonstrates practical experience in:
=> LLM orchestration
=> Vector databases
=> Stateful graph execution


🏗️ System Architecture

The system is built using LangGraph to manage a state-driven execution pipeline.

Execution Flow
User provides file path
    ↓
Document is loaded into memory from that path
    ↓
chunking is done to improve semantic searching and to avoid error due to token size limit
    ↓
Convert chunks to embeddings
    ↓
store embeddings in vector database
    ↓
Retrieve Top-K Similar Chunks (Vector Search)
    ↓
Context Builder (Token-Limited)
    ↓
Prompt Augmentation
    ↓
LLM Generation
    ↓
Return Response + Persist Conversation State


Graph Structure
START
  ↓
Retrieve_k_Most_Similar_Chunks
  ↓
build_context
  ↓
Augmentation
  ↓
Generation
  ↓
END

Here We do the loading of document, chunking, generating embeddings and storing them in vector db only once,
then the conversational flow starts where the user can query the document and get the llm generated context aware response.

=> Unlike linear chains, the graph-based architecture allows:
=> Stateful execution
=> Expandability (routing, tools, agents)
=> Controlled memory persistence



Follow production design patterns

3️⃣ Modular Architecture

The project follows separation of concerns:

AskMyDocs/
│
├── main.py
│
├── Rag/
│   ├── Workflow.py
│   ├── llm.py
│   ├── ChromaDbVectorStoreManager.py
│   ├── DocumentProcessor.py
│   ├── models.py
│
└── README.md

File	Responsibility
main.py	CLI orchestration & user interface
Workflow.py	LangGraph state machine
llm.py	LLM configuration
ChromaDbVectorStoreManager.py	Embeddings & vector storage
DocumentProcessor.py	PDF loading & chunking
models.py	Typed state definitions


⚙️ Tech Stack
Component	Technology
Language	Python 3.10+
LLM Orchestration	LangGraph
LLM	Ollama (local LLM execution)
Embeddings	Sentence Transformers (all-mpnet-base-v2)
Vector Database	ChromaDB
State Persistence	LangGraph MemorySaver
Document Processing	LangChain Document Loaders


🔍 Features
✅ Conversational RAG
Maintains multi-turn memory using LangGraph checkpointer.

✅ Token-Limited Context Builder
Prevents prompt overflow by dynamically restricting context size.

✅ Top-K Semantic Retrieval
Uses dense embeddings for accurate similarity search.

✅ Persistent Vector Store
Documents are embedded and stored once per session.

✅ Clean CLI Experience



🛠️ Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/AskMyDocs.git
cd AskMyDocs

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Install & Pull Ollama Model
ollama pull llama3

▶️ Running the Application
python main.py


Example session:

📚 AskMyDocs - Intelligent Document Q&A System
Enter path to your PDF file: sample.pdf

You: Explain the architecture
AI: The architecture follows a retrieval-augmented generation pipeline...


Type exit to quit.
