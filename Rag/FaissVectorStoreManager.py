from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from typing import List, Optional
from Rag.models import Query


class FaissVectorStoreManager:
    """Manage Faiss vector store with embedding conversion and storage with persistence"""
    def __init__(self):
        #initialize embedding model
        """
        connects to huggingface.io | Downloads model(first time only) | caches locally into ~/.cache/huggingface/
        loads model into memory
        configures where the model runs | device = cpu(for running on cpu), device = cuda(for runnning on gpu)
        'normalize_embeddings': True | to make sure that all vectors have equal length for better similarity comparision
        """
        self.embeddings = HuggingFaceEmbeddings(
            model = "sentence-transformers/all-mpnet-base-v2",
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {'normalize_embeddings': True}
        )

        self.vectorstore: Optional[FAISS] = None

    def Convert_And_Store(self, chunks: List[Document]) -> None:
        
        """
        Convert chunks to embeddings and store in FAISS
    
        Args:
            chunks: List[Document]
        """
        try:
            if self.vectorstore is None:
                #Creating new vector store and adding chunks to it 
                self.vectorstore = FAISS.from_documents(
                    chunks,
                    self.embeddings
                )
            else:
                #Adding chunks to existing vector store
                self.vectorstore.add_documents(chunks)
            
            return self.vectorstore
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
