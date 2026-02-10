from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional
from Rag.models import FileInfo

class DocumentProcessor():
    """Load Documents and split them into chunks"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = ["\n\n", "\n", " ", ""]):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators = separators
        )
        
    def Load_Document(self, file_info: FileInfo) -> List[Document]:
        """
        load document based on file type
    
        Args:
            file_info: FileInfo object containing file_path and file_type(pdf, doc, text)
    
        Returns: 
            List of document objects
        """
        file_path = file_info.file_path
        file_type = file_info.file_type
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "doc":
                loader = Docx2txtLoader(file_path)
            elif file_type == "text":
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        
            documents = loader.load()
            return documents

        except Exception as e:
            raise e
    
    def Split_Document_Objects(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into chunks
    
        Args:
            documents: List of document objects
    
        Returns:
            List of document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            raise e
