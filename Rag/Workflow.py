import uuid
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from Rag.ChromaDbVectorStoreManager import ChromadbVectorStoreManager
from Rag.DocumentProcessor import DocumentProcessor
from Rag.models import FileInfo, Query
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import messages_from_dict
from Rag.llm import llm

vector_store_manager = None

class QAState(TypedDict):
    query: str
    token_size: int
    k_similar_chunks: List[Document]
    context: str
    prompt: str
    llm_response: str
    messages: List[BaseMessage]

def Retrieve_k_Most_Similar_Chunks(state: QAState) -> QAState:
    query = state["query"]
    query = Query(
        query=query,
        k=5
    )
    k_similar_chunks = vector_store_manager.search(query)
    return {"k_similar_chunks": k_similar_chunks}

def build_context(state: QAState) -> QAState:
    k_similar_chunks = state["k_similar_chunks"]
    max_chars = state["token_size"]
    context_blocks = []
    total_chars = 0

    for idx, doc in enumerate(k_similar_chunks, start=1):
        text = doc.page_content.strip()
        if not text:
            continue

        block = f"[Document {idx}]\n{text}\n"

        if total_chars + len(block) > max_chars:
            break

        context_blocks.append(block)
        total_chars += len(block)

    context = "\n\n".join(context_blocks)
    return {"context": context}

def Augumentation(state: QAState) -> QAState:
    query = state["query"]
    context = state["context"]

    prompt = f"""
        You are a helpful and precise AI assistant.
        You are given a user question and a set of retrieved context passages from documents.
        Your task is to answer the question using ONLY the information provided in the context.
        
        Rules:
        - Use only the given context to answer the question.
        - Do NOT make up facts or use outside knowledge.
        - If the answer is not present in the context, say:
          "I don’t know based on the provided documents."
        - Be concise, clear, and factual.
        - If helpful, you may quote short phrases from the context.


        Context:
        {context}

        Question:
        {query}
    """

    return {"prompt": prompt}

def Generation(state: QAState) -> QAState:
    prompt = state["prompt"]
    raw_history = state.get("messages", [])

    if raw_history and isinstance(raw_history[0], dict):
        history = messages_from_dict(raw_history)
    else:
        history = raw_history
        
    messages = history + [prompt]
    response = llm.invoke(messages)

    history.append(HumanMessage(content=state["query"]))
    history.append(AIMessage(content=response.content))

    return {"llm_response": response.content, "messages": history}


graph = StateGraph(QAState)

#add nodes
graph.add_node("Retrieve_k_Most_Similar_Chunks", Retrieve_k_Most_Similar_Chunks)
graph.add_node("build_context", build_context)
graph.add_node("Augumentation", Augumentation)
graph.add_node("Generation", Generation)

#add edges
graph.add_edge(START, "Retrieve_k_Most_Similar_Chunks")
graph.add_edge("Retrieve_k_Most_Similar_Chunks", "build_context")
graph.add_edge("build_context", "Augumentation")
graph.add_edge("Augumentation", "Generation")
graph.add_edge("Generation", END)


def Load_Documents(file_info: FileInfo, document_processor: DocumentProcessor) -> List[Document]:
    documents = document_processor.Load_Document(file_info=file_info)
    return documents

def Create_Chunks(document_processor: DocumentProcessor, documents: List[Document]) -> List[Document]:
    chunks = document_processor.Split_Document_Objects(documents)
    return chunks

def Convert_Chunks_And_Store_In_Vector_Store(chunks: List[Document], vector_store_manager: Chroma) -> None:
    vector_store_manager.Convert_And_Store(chunks)

def Execute_Workflow(file_info: FileInfo) -> None:
    global vector_store_manager
    document_processor = DocumentProcessor()
    documents = Load_Documents(file_info, document_processor)
    chunks = Create_Chunks(document_processor, documents)

    vector_store_manager = ChromadbVectorStoreManager()
    Convert_Chunks_And_Store_In_Vector_Store(chunks, vector_store_manager)
    memory = MemorySaver()
    workflow = graph.compile(checkpointer=memory)
    thread_id = str(uuid.uuid4())

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit", "q", "bye", "goodbye"]:
            print("Goodbye 👋")
            break

        initial_state = {
            "query": query,
            "token_size": 5000
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = workflow.invoke(initial_state, config=config)
        print("\nAI:", result["llm_response"])
