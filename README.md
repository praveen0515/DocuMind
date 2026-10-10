# 🧠 DocuMind — Enterprise RAG-Powered Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.14-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B6B?style=for-the-badge)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **DocuMind** is a production-grade, enterprise Retrieval-Augmented Generation (RAG) platform that transforms complex documents (PDFs, DOCX, TXT) into grounded, verifiable intelligence with precise source citations.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation & Local Run](#installation--local-run)
- [Docker Deployment](#-docker-deployment)
- [API Reference](#-api-reference)
- [Environment Variables](#-environment-variables)
- [Screenshots & UI Showcase](#-screenshots--ui-showcase)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🌟 Overview

Enterprise knowledge bases are often locked inside unstructured documents like vendor agreements, financial disclosures, technical specifications, and legal audits. Standard keyword search lacks semantic comprehension, while naive LLM queries suffer from hallucinations and lack factual citations.

**DocuMind** bridges this gap. It processes and embeds your documents into a persistent vector store and utilizes Google's Gemini models to deliver verifiable, context-grounded answers. Every claim is strictly attributed to the exact source document, page number, and chunk index.

---

## ✨ Key Features

1. **Multi-Format Document Ingestion**: Seamlessly ingest multi-page PDFs, Microsoft Word (.docx), Markdown, and plain text files with automatic metadata preservation.
2. **Page-Aware Semantic Chunking**: Employs LangChain's `RecursiveCharacterTextSplitter` with intelligent boundary detection while capturing exact page numbers and chunk offsets.
3. **High-Performance Vector Indexing**: Persistent vector storage powered by **ChromaDB**, ensuring fast cosine similarity lookups without external database overhead.
4. **State-of-the-Art Google GenAI SDK**: Embeddings powered by `text-embedding-004` and answer generation orchestrated through `gemini-1.5-flash` for high throughput and low latency.
5. **Strict Grounded Attribution & Citations**: Responses contain structured citations with source filename, page number, relevance score, and verbatim excerpt previews.
6. **Full-Featured Async REST API**: Built on **FastAPI** with async endpoints, OpenAPI documentation (/docs), CORS support, and robust exception handling.
7. **Interactive Enterprise Web UI**: Modern Single Page Application featuring drag-and-drop document uploads, real-time index stats, customizable retrieval depth, and interactive citation drawers.
8. **Containerized Production Multi-Stage Build**: Complete with Dockerfile and Docker Compose featuring a non-root application user and persistent volume mounts.

---

## 🏗️ System Architecture

### Processing & Query Flow

The platform separates document ingestion from interactive question answering:

```
[ User Upload ] ──────> [ Document Parser ] ──────> [ Semantic Chunking ]
(PDF, DOCX, TXT)        (pypdf / python-docx)       (Recursive Splitter)
                                                            │
                                                            ▼
[ ChromaDB Storage ] <── [ Google GenAI Embeddings ] <──────┘
(Persistent Vector Store)  (text-embedding-004)
        │
        │ Semantic Search (Cosine Similarity)
        ▼
[ Top-K Context Chunks ] ──┐
                           ├─> [ Grounding Prompt ] ──> [ Google Gemini 1.5 ] ──> [ Cited Response ]
[ User Query ] ────────────┘                                                    (Answer + Sources)
```

### Flow Breakdown

1. **User Upload**: Files are received via the `/api/documents/upload` endpoint.
2. **Document Processing**: Extracted page-by-page (PDF) or paragraph-by-paragraph (DOCX) while capturing page metadata.
3. **Chunking**: Text is split into overlapping chunks (default 1,000 characters with 150-character overlap) to preserve contextual boundaries.
4. **Embedding**: Chunks are transformed into high-dimensional vector representations using Google's `text-embedding-004` model.
5. **ChromaDB Persistence**: Embeddings, source text, and rich metadata are saved to the persistent vector database.
6. **Query & Retrieval**: A user's query is converted to an embedding, retrieving the top $K$ most semantically relevant chunks.
7. **Gemini LLM Synthesis**: Gemini synthesizes the final response, restricted strictly to facts contained in the retrieved context.
8. **Cited Response**: The user receives the synthesized answer alongside verifiable source references, page numbers, and similarity scores.

---

## 🛠️ Tech Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | `0.111.0` | Asynchronous high-performance REST API |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) | `0.30.1` | Production ASGI web server |
| **LLM & Embeddings** | [Google GenAI SDK](https://ai.google.dev/) / Gemini | `1.5-flash` / `text-embedding-004` | Language generation and vector embeddings |
| **RAG Orchestrator** | [LangChain](https://www.langchain.com/) | `0.2.14` | Splitters, chains, and prompt management |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | `0.5.5` | Embedded, persistent vector database |
| **Document Parsers** | [pypdf](https://pypdf.readthedocs.io/) & [python-docx](https://python-docx.readthedocs.io/) | `4.3.1` / `1.1.2` | Extraction from PDF, DOCX, and TXT |
| **Frontend** | Vanilla JS + Tailwind CSS + Marked.js | CDN | Modern, responsive enterprise interface |
| **Containerization** | Docker & Docker Compose | Multi-Stage | Production deployment and local orchestration |

---

## 📂 Project Structure

```text
DocuMind/
├── backend/
│   ├── __init__.py
│   ├── config.py             # Pydantic Settings & environment variables
│   ├── main.py               # FastAPI application, routes, and lifespan handlers
│   ├── rag_engine.py         # Ingestion, ChromaDB client, embeddings & Gemini chain
│   └── requirements.txt      # Pinned Python dependencies
├── frontend/
│   └── index.html            # Enterprise SPA UI (Upload, Query, Citations)
├── data/
│   ├── chroma_db/            # Persistent ChromaDB storage directory
│   └── uploads/              # Local storage for uploaded documents
├── .env.example              # Template environment configuration file
├── .gitignore                # Git ignore for Python, Node, data, and cache
├── docker-compose.yml        # Docker Compose configuration for multi-container run
├── Dockerfile                # Multi-stage production Docker build
├── LICENSE                   # MIT License
└── README.md                 # Complete documentation
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or 3.11+
- **Google AI Studio API Key**: Get a free API key at [Google AI Studio](https://aistudio.google.com/)
- **Git**

### Installation & Local Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/praveen0515/DocuMind.git
   cd DocuMind
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set your `GOOGLE_API_KEY`:
   ```ini
   GOOGLE_API_KEY=AIzaSy...your_actual_gemini_api_key...
   ```

5. **Start the DocuMind server**:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the application**:
   - **Interactive Web Interface**: Open [http://localhost:8000](http://localhost:8000)
   - **Interactive Swagger API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Alternative ReDoc**: Open [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🐳 Docker Deployment

DocuMind includes a production-grade multi-stage `Dockerfile` and `docker-compose.yml` for turnkey containerization.

### Using Docker Compose (Recommended)

1. Make sure your `.env` contains your `GOOGLE_API_KEY`.
2. Build and launch the container:
   ```bash
   docker compose up --build -d
   ```
3. Check container logs:
   ```bash
   docker compose logs -f backend
   ```
4. Access the web app at [http://localhost:8000](http://localhost:8000).
5. Stop the container:
   ```bash
   docker compose down
   ```

### Using Standalone Docker Build

```bash
# Build production image
docker build -t documind:latest .

# Run container with persistent volumes
docker run -d \
  --name documind-app \
  -p 8000:8000 \
  -e GOOGLE_API_KEY="your_api_key_here" \
  -v $(pwd)/data/chroma_db:/app/data/chroma_db \
  -v $(pwd)/data/uploads:/app/data/uploads \
  documind:latest
```

---

## 📡 API Reference

| Method | Endpoint | Description | Request Payload | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | System health, vector DB stats, and key status | None | `HealthCheckResponse` |
| `POST` | `/api/documents/upload` | Ingest and index document (PDF, DOCX, TXT) | `multipart/form-data` | `UploadResponse` |
| `POST` | `/api/query` | Ask questions with semantic grounding | `{"query": "...", "top_k": 4}` | `QueryResponse` (Answer + Citations) |
| `GET` | `/api/documents` | List all indexed documents and chunk statistics | None | `List[DocumentStats]` |
| `DELETE` | `/api/documents/{filename}` | Remove a document from the vector store | None | `{"status": "deleted"}` |
| `POST` | `/api/documents/clear` | Purge entire vector index | None | `{"status": "cleared"}` |

### Example Query Request & Response

#### Request
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the liability cap specified in Section 8?",
    "top_k": 4
  }'
```

#### Response
```json
{
  "query": "What is the liability cap specified in Section 8?",
  "answer": "According to Section 8 of the Master Services Agreement, the total aggregate liability for either party is capped at twelve (12) months of service fees paid prior to the incident [Source 1: vendor_agreement.pdf, Page 14]. Neither party shall be liable for indirect, incidental, or consequential damages [Source 2: vendor_agreement.pdf, Page 15].",
  "citations": [
    {
      "citation_id": 1,
      "source": "vendor_agreement.pdf",
      "page": 14,
      "chunk_index": 28,
      "similarity_score": 0.942,
      "excerpt": "8.1 Limitation of Liability. IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED THE TOTAL AMOUNT PAID BY CUSTOMER HEREUNDER IN THE TWELVE (12) MONTHS PRECEDING THE INCIDENT..."
    },
    {
      "citation_id": 2,
      "source": "vendor_agreement.pdf",
      "page": 15,
      "chunk_index": 29,
      "similarity_score": 0.891,
      "excerpt": "8.2 Exclusion of Consequential Damages. In no event shall either party have any liability for any lost profits, loss of data, or indirect, special, incidental, or consequential damages..."
    }
  ],
  "retrieved_chunks_count": 2
}
```

---

## ⚙️ Environment Variables

Configure these settings in your `.env` file:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | *None (Required)* | Google Gemini API key |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model for semantic search |
| `LLM_MODEL` | `gemini-1.5-flash` | Gemini model for answer generation |
| `LLM_TEMPERATURE` | `0.2` | Creativity temperature (`0.0` - `1.0`) |
| `MAX_OUTPUT_TOKENS` | `2048` | Max token budget for synthesis |
| `CHUNK_SIZE` | `1000` | Maximum character length per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap characters between chunks |
| `TOP_K_RESULTS` | `4` | Number of chunks retrieved per query |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma_db` | Persistent vector database directory |
| `UPLOAD_DIR` | `./data/uploads` | Directory for raw uploaded files |
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8000` | API port |

---

## 📸 Screenshots & UI Showcase

```
+---------------------------------------------------------------------------------------+
|  🧠 DocuMind  [ENTERPRISE RAG]                        ● Online (48 Chunks)   Swagger  |
+------------------------------------+--------------------------------------------------+
|  DOCUMENT INGESTION                |  ASK DOCUMENT INTELLIGENCE                       |
|  +------------------------------+  |  +--------------------------------------------+  |
|  |   [Drag & Drop PDF / DOCX]   |  |  | What are the termination notice periods?   |  |
|  +------------------------------+  |  +--------------------------------------------+  |
|                                    |  Prompt Ideas: [Executive Summary] [Liabilities] |
|  INDEXED SOURCES (3)               |                                                  |
|  • MSA_Final.pdf (18 Chunks)   🗑   |  GROUNDED RESPONSE                               |
|  • Security_Policy.docx (12)   🗑   |  Termination requires thirty (30) days written   |
|  • Q3_Financials.txt (18)      🗑   |  notice for convenience [Source 1, Page 12]...   |
|                                    |                                                  |
|                                    |  VERIFIED CITATIONS:                             |
|                                    |  [#1] MSA_Final.pdf - Page 12 (94% Match)        |
|                                    |  [#2] MSA_Final.pdf - Page 13 (88% Match)        |
+------------------------------------+--------------------------------------------------+
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

## 👨‍💻 Author

**Praveen Patil**
- **GitHub**: [@praveen0515](https://github.com/praveen0515)
- **LinkedIn**: [linkedin.com/in/praveen0515](https://linkedin.com/in/praveen0515)

---

<p align="center">
  Built with ❤️ for scalable Enterprise Document Intelligence.
</p>
