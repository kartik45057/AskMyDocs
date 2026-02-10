from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from Rag.DocumentProcessor import DocumentProcessor
from Rag.FaissVectorStoreManager import FaissVectorStoreManager
from Rag.models import FileInfo, Query
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from Rag.llm import llm



class QAState(TypedDict):
    query: str
    token_size: int
    file_path: str
    file_type: str
    documents: List[Document]
    chunks: List[Document]
    vector_store: FAISS
    k_similar_chunks: List[Document]
    context: str
    prompt: str
    llm_response: str

def Load_Documents(state: QAState) -> QAState:
    document_processor = DocumentProcessor()
    file_path = state["file_path"]
    file_type = state["file_type"]
    file_info = FileInfo(
        file_path=file_path,
        file_type=file_type
    )

    documents = document_processor.Load_Document(file_info=file_info)
    return {"documents": documents}

def Create_Chunks(state: QAState) -> QAState:
    document_processor = DocumentProcessor()
    documents = state["documents"]
    chunks = document_processor.Split_Document_Objects(documents)
    return {"chunks": chunks}

def Convert_Chunks_And_Store_In_Vector_Store(state: QAState) -> QAState:
    faiss_vector_store = FaissVectorStoreManager()
    chunks = state["chunks"]
    faiss_vector_store.Convert_And_Store(chunks)

    return {"vector_store": faiss_vector_store}

def Retrieve_k_Most_Similar_Chunks(state: QAState) -> QAState:
    query = state["query"]
    query = Query(
        query=query
    )
    vector_store = state["vector_store"]
    k_similar_chunks = vector_store.search(query)
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
    messages = [HumanMessage(content=prompt)]

    response = llm.invoke(messages)

    return {"llm_response": response}


graph = StateGraph(QAState)

#add nodes
graph.add_node("Load_Documents", Load_Documents)
graph.add_node("Create_Chunks", Create_Chunks)
graph.add_node("Convert_Chunks_And_Store_In_Vector_Store", Convert_Chunks_And_Store_In_Vector_Store)
graph.add_node("Retrieve_k_Most_Similar_Chunks", Retrieve_k_Most_Similar_Chunks)
graph.add_node("build_context", build_context)
graph.add_node("Augumentation", Augumentation)
graph.add_node("Generation", Generation)

#add edges
graph.add_edge(START, "Load_Documents")
graph.add_edge("Load_Documents", "Create_Chunks")
graph.add_edge("Create_Chunks", "Convert_Chunks_And_Store_In_Vector_Store")
graph.add_edge("Convert_Chunks_And_Store_In_Vector_Store", "Retrieve_k_Most_Similar_Chunks")
graph.add_edge("Retrieve_k_Most_Similar_Chunks", "build_context")
graph.add_edge("build_context", "Augumentation")
graph.add_edge("Augumentation", "Generation")
graph.add_edge("Generation", END)


def Execute_Workflow(query: str, file_info: FileInfo):
    workflow = graph.compile()
    initial_state = {"query": query, "token_size": 5000, "file_path": file_info.file_path, "file_type": file_info.file_type}
    result = workflow.invoke(initial_state)
    return result
