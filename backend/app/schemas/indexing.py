from typing import Optional
from pydantic import BaseModel, Field


class IndexDocumentResponse(BaseModel):
    document_id: str
    status: str = Field(..., description="Indexing status ('indexed')")
    indexed_chunks: int = Field(..., description="Number of chunks successfully indexed in ChromaDB")
    total_chunks: int = Field(..., description="Total chunks for this document")
    embedding_model: str = Field(..., description="Embedding model name used for indexing")
    embedding_dimension: int = Field(default=384, description="Embedding dimension")
    collection_name: str = Field(..., description="ChromaDB collection name")
    message: str = Field(default="Document successfully indexed in vector store")


class IndexStatusResponse(BaseModel):
    document_id: str
    status: str = Field(..., description="Indexing status: 'indexed', 'not_indexed', or 'partial'")
    is_indexed: bool = Field(..., description="Whether document chunks are currently indexed in ChromaDB")
    indexed_chunks: int = Field(..., description="Number of chunks indexed in ChromaDB")
    total_chunks: int = Field(..., description="Total chunks created for this document")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model name")
    embedding_dimension: Optional[int] = Field(default=384, description="Embedding dimension")
    collection_name: str = Field(..., description="ChromaDB collection name")
