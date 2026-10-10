"""
DocuMind - Enterprise RAG Engine
Handles document ingestion, text extraction, chunking, embedding,
vector indexing with ChromaDB, and cited question answering with Google Gemini.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

# Document Parsing
import pypdf
from docx import Document as DocxDocument

# LangChain components
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings


class RAGEngine:
    """Enterprise RAG pipeline orchestrating parsing, embeddings, indexing, and synthesis."""

    COLLECTION_NAME = "documind_documents"

    def __init__(self):
        self._validate_api_key()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "DocuMind Knowledge Base"}
        )

        # Lazy-initialized components
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    def _validate_api_key(self) -> bool:
        """Check if Google API Key is present in settings or environment."""
        api_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        return bool(api_key and api_key.strip() and api_key != "your_google_gemini_api_key_here")

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Instantiate Google GenAI Embeddings with caching."""
        if not self._validate_api_key():
            raise ValueError(
                "GOOGLE_API_KEY is not configured. Please set your Gemini API key in backend/.env"
            )
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
            )
        return self._embeddings

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Instantiate Google Gemini Chat model."""
        if not self._validate_api_key():
            raise ValueError(
                "GOOGLE_API_KEY is not configured. Please set your Gemini API key in backend/.env"
            )
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL,
                google_api_key=settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY"),
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.MAX_OUTPUT_TOKENS
            )
        return self._llm

    # =========================================================================
    # Document Ingestion & Extraction
    # =========================================================================

    def extract_text_from_file(self, file_path: str, filename: str) -> List[Document]:
        """Extract text content and page-level metadata from PDF, DOCX, or TXT."""
        file_ext = Path(filename).suffix.lower()
        documents: List[Document] = []

        if file_ext == ".pdf":
            reader = pypdf.PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_idx + 1,
                                "total_pages": len(reader.pages),
                                "file_type": "pdf"
                            }
                        )
                    )

        elif file_ext == ".docx":
            doc = DocxDocument(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        full_text.append(row_text)

            combined_text = "\n\n".join(full_text)
            if combined_text.strip():
                documents.append(
                    Document(
                        page_content=combined_text,
                        metadata={
                            "source": filename,
                            "page": 1,
                            "total_pages": 1,
                            "file_type": "docx"
                        }
                    )
                )

        elif file_ext in [".txt", ".md", ".csv"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if content.strip():
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "page": 1,
                            "total_pages": 1,
                            "file_type": "txt"
                        }
                    )
                )
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: PDF, DOCX, TXT")

        if not documents:
            raise ValueError(f"No extractable text found in file '{filename}'.")

        return documents

    def index_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Parses document, splits it into chunks, embeds chunks via Google GenAI,
        and saves them into the persistent ChromaDB collection.
        """
        # 1. Extract raw document pages
        raw_docs = self.extract_text_from_file(file_path, filename)

        # 2. Split into semantic chunks while maintaining metadata
        chunked_docs = self.text_splitter.split_documents(raw_docs)

        if not chunked_docs:
            raise ValueError(f"Document chunking produced 0 chunks for '{filename}'.")

        # 3. Prepare data for ChromaDB
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        doc_id = str(uuid.uuid4())[:8]

        for idx, chunk in enumerate(chunked_docs):
            chunk_id = f"{filename}_{doc_id}_chunk_{idx}"
            texts.append(chunk.page_content)
            metadatas.append({
                "source": filename,
                "doc_id": doc_id,
                "page": chunk.metadata.get("page", 1),
                "total_pages": chunk.metadata.get("total_pages", 1),
                "chunk_index": idx,
                "total_chunks": len(chunked_docs),
                "file_type": chunk.metadata.get("file_type", "unknown")
            })
            ids.append(chunk_id)

        # 4. Generate embeddings via Google GenAI Embeddings
        embeddings_list = self.embeddings.embed_documents(texts)

        # 5. Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )

        return {
            "filename": filename,
            "document_id": doc_id,
            "total_pages": raw_docs[-1].metadata.get("total_pages", len(raw_docs)),
            "chunks_indexed": len(chunked_docs),
            "status": "indexed"
        }

    # =========================================================================
    # RAG Retrieval & Answer Generation with Citations
    # =========================================================================

    def query(self, question: str, top_k: Optional[int] = None, filter_doc: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves relevant document chunks and synthesizes an answer with citations
        using Google Gemini.
        """
        k = top_k or settings.TOP_K_RESULTS

        # Check if vector store has any documents
        total_items = self.collection.count()
        if total_items == 0:
            return {
                "query": question,
                "answer": "No documents have been indexed yet. Please upload one or more documents (PDF, DOCX, TXT) to query the knowledge base.",
                "citations": [],
                "retrieved_chunks_count": 0
            }

        # 1. Embed query
        query_embedding = self.embeddings.embed_query(question)

        # 2. Query ChromaDB
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, total_items)
        }
        if filter_doc:
            query_kwargs["where"] = {"source": filter_doc}

        results = self.collection.query(**query_kwargs)

        retrieved_docs = results.get("documents", [[]])[0]
        retrieved_metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(retrieved_docs)

        if not retrieved_docs:
            return {
                "query": question,
                "answer": "No relevant context was found in the indexed documents matching your query.",
                "citations": [],
                "retrieved_chunks_count": 0
            }

        # 3. Construct context string and structured citations
        context_blocks = []
        citations = []

        for idx, (doc_text, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metas, distances)):
            source = meta.get("source", "Unknown Document")
            page = meta.get("page", 1)
            chunk_idx = meta.get("chunk_index", idx)

            # ChromaDB cosine distance to similarity score
            similarity = round(max(0.0, 1.0 - (dist if dist is not None else 0.5)), 3)

            citation_ref = f"[Source {idx+1}: {source}, Page {page}]"
            context_blocks.append(f"{citation_ref}\n{doc_text}\n")

            citations.append({
                "citation_id": idx + 1,
                "source": source,
                "page": page,
                "chunk_index": chunk_idx,
                "similarity_score": similarity,
                "excerpt": doc_text[:280] + ("..." if len(doc_text) > 280 else "")
            })

        formatted_context = "\n---\n".join(context_blocks)

        # 4. Prompt Engineering for Grounded Synthesis with Citations
        prompt_template = PromptTemplate(
            template="""You are DocuMind, an enterprise AI document intelligence assistant.
Your goal is to answer the user query with precision, clarity, and factual accuracy strictly based on the provided context.

Context Excerpts:
{context}

User Query:
{query}

Guidelines:
1. Answer the query thoroughly using ONLY the facts present in the context excerpts above.
2. For every factual assertion, attribute your answer using brackets indicating the source reference, e.g., [Source 1: filename.pdf, Page X].
3. If the context does not contain adequate information to answer the question, state: "Based on the provided documents, I do not have enough information to answer this question." Do not make assumptions or fabricate details.
4. Maintain a professional, executive enterprise tone. Format code, key terms, or structured summaries using clear Markdown tables or bullet points where appropriate.

Synthesized Answer:""",
            input_variables=["context", "query"]
        )

        prompt = prompt_template.format(context=formatted_context, query=question)

        # 5. Invoke Gemini LLM
        response = self.llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        return {
            "query": question,
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunks_count": len(citations)
        }

    # =========================================================================
    # Knowledge Base Management
    # =========================================================================

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Returns aggregate metadata for all documents stored in the vector base."""
        count = self.collection.count()
        if count == 0:
            return []

        all_records = self.collection.get(include=["metadatas"])
        metadatas = all_records.get("metadatas", [])

        docs_map: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            source = meta.get("source", "Unknown")
            if source not in docs_map:
                docs_map[source] = {
                    "source": source,
                    "file_type": meta.get("file_type", "unknown"),
                    "total_pages": meta.get("total_pages", 1),
                    "chunk_count": 0,
                    "doc_id": meta.get("doc_id", "")
                }
            docs_map[source]["chunk_count"] += 1

        return list(docs_map.values())

    def delete_document(self, filename: str) -> int:
        """Deletes all chunks associated with the specified document from ChromaDB."""
        # Find matching IDs
        records = self.collection.get(
            where={"source": filename},
            include=["metadatas"]
        )
        ids_to_delete = records.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def clear_all_documents(self) -> int:
        """Purges the entire knowledge base collection."""
        count = self.collection.count()
        self.chroma_client.delete_collection(name=self.COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "DocuMind Knowledge Base"}
        )
        return count


# Singleton instance
rag_engine = RAGEngine()
