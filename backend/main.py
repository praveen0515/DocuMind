"""
DocuMind - FastAPI Application Server
Enterprise RAG-Powered Document Intelligence Platform API
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    status,
    Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.rag_engine import rag_engine

# Path configurations
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


# -----------------------------------------------------------------------------
# Lifespan Events
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown event handler."""
    # Ensure working directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    yield
    # Cleanup operations (if needed)


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -----------------------------------------------------------------------------
class CitationModel(BaseModel):
    citation_id: int = Field(..., description="Numerical reference ID")
    source: str = Field(..., description="Source document filename")
    page: int = Field(..., description="Document page number")
    chunk_index: int = Field(..., description="Zero-indexed chunk position")
    similarity_score: float = Field(..., description="Cosine similarity score")
    excerpt: str = Field(..., description="Preview excerpt of the source context")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="The user's question or search query")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="Number of context chunks to retrieve")
    filter_document: Optional[str] = Field(default=None, description="Optional filename filter")


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    retrieved_chunks_count: int


class DocumentStats(BaseModel):
    source: str
    file_type: str
    total_pages: int
    chunk_count: int
    doc_id: str


class UploadResponse(BaseModel):
    filename: str
    document_id: str
    total_pages: int
    chunks_indexed: int
    status: str
    message: str


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    google_api_configured: bool
    total_indexed_documents: int
    total_vector_chunks: int


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """Verify backend health, LLM API key status, and vector database stats."""
    api_key_set = rag_engine._validate_api_key()
    indexed_docs = rag_engine.get_indexed_documents()
    total_chunks = rag_engine.collection.count()

    return HealthCheckResponse(
        status="healthy",
        version=settings.VERSION,
        google_api_configured=api_key_set,
        total_indexed_documents=len(indexed_docs),
        total_vector_chunks=total_chunks
    )


@app.post(
    "/api/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document (PDF, DOCX, or TXT).
    Extracts text, generates semantic chunks, embeds via Google GenAI,
    and stores in ChromaDB.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload payload."
        )

    ext = Path(file.filename).suffix.lower()
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(allowed_extensions)}"
        )

    # Destination file path
    safe_filename = Path(file.filename).name
    save_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

    try:
        # Save file to disk
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write file to disk: {str(e)}"
        )
    finally:
        await file.close()

    # Index into vector database
    try:
        result = rag_engine.index_document(save_path, safe_filename)
        return UploadResponse(
            filename=result["filename"],
            document_id=result["document_id"],
            total_pages=result["total_pages"],
            chunks_indexed=result["chunks_indexed"],
            status=result["status"],
            message=f"Successfully indexed '{safe_filename}' into {result['chunks_indexed']} semantic chunks."
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexing document: {str(e)}"
        )


@app.post("/api/query", response_model=QueryResponse, tags=["RAG Query"])
async def query_documents(request: QueryRequest):
    """
    Submit a question against the indexed documents.
    Retrieves the most semantically relevant chunks and generates a grounded,
    cited answer using Google Gemini.
    """
    try:
        response = rag_engine.query(
            question=request.query,
            top_k=request.top_k,
            filter_doc=request.filter_document
        )
        return QueryResponse(**response)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@app.get("/api/documents", response_model=List[DocumentStats], tags=["Documents"])
async def list_documents():
    """Retrieve metadata and chunk counts for all indexed documents."""
    try:
        return rag_engine.get_indexed_documents()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.delete("/api/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Remove a document and its indexed chunks from the knowledge base."""
    try:
        chunks_removed = rag_engine.delete_document(filename)
        # Attempt to delete file from disk if present
        disk_path = os.path.join(settings.UPLOAD_DIR, filename)
        if os.path.exists(disk_path):
            os.remove(disk_path)

        return {
            "filename": filename,
            "status": "deleted",
            "chunks_removed": chunks_removed,
            "message": f"Successfully removed '{filename}' and {chunks_removed} associated chunks."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.post("/api/documents/clear", tags=["Documents"])
async def clear_all_documents():
    """Clear all documents and vectors from the knowledge base."""
    try:
        count = rag_engine.clear_all_documents()
        return {
            "status": "cleared",
            "chunks_cleared": count,
            "message": "All indexed documents and vector chunks have been removed."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )


# -----------------------------------------------------------------------------
# Static Files & Frontend SPA Serving
# -----------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the single page application interface."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(
            content={"message": "DocuMind API is running. Visit /docs for OpenAPI documentation."}
        )


# Global exception handler for fallback
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unhandled server error occurred: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
