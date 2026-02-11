from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import List
from Rag.embeddings import embeddings
from Rag.models import Query
import os


class ChromadbVectorStoreManager:
    """Manage Faiss vector store with embedding conversion and storage with persistence"""
    def __init__(self, persist_directory = "./chroma_persistent_db", collection_name = "default_collection"):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = self.Initialize_Vectorstore()

    def Initialize_Vectorstore(self) -> Chroma:
        """
        Initialize or load existing vector store
        
        Returns: 
            Chroma vector store instance
        """
        try:
            if os.path.exists(self.persist_directory):
                return Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                os.makedirs(self.persist_directory, exist_ok=True)
                return Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        except Exception as e:
            raise e

    def Convert_And_Store(self, chunks: List[Document]) -> None:
        
        """
        Convert chunks to embeddings and store in FAISS
    
        Args:
            chunks: List[Document]
        """
        try:
            self.vectorstore.add_documents(chunks)
        except Exception as e:
            raise e
    
    def search(self, args: Query) -> List[Document]:
        """
        Search for similar documents

        Args:
            query: search query from the user
            k: Numbers of similar documents

        Returns:
            List of k most similar documents
        """
        query = args.query
        k = args.k
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized, load the vector store or create first")

            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            raise e
    
    def search_with_score(self, args: Query) -> List[Document]:
        """
        Search for similar documents

        Args:
            query: search query from the user
            k: Numbers of similar documents

        Returns:
            Tuples of k most similar documents with scores(Document, score)
        """
        query = args.query
        k = args.k
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized, load the vector store or create first")

            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            raise e
