
# 🧠 Enhanced RAG System with Ollama, FAISS, and LangChain

This project implements a **Retrieval-Augmented Generation (RAG)** system using:
- **FAISS** for fast vector similarity search,
- **Ollama Embeddings** (via `nomic-embed-text`) for embedding text,
- A **local LLaMA-3 model** served through **Ollama** for natural language question answering,
- **LangChain** to manage document loaders, retrievers, and QA chains.

It supports **country-aware retrieval**, ensuring responses are contextually relevant to your query.

## 🚀 Features

- 📄 Load and chunk PDF documents using LangChain
- 🗺️ Automatically tag content with company and country metadata
- 📦 Store document embeddings in a FAISS vector store
- 🔍 Entity-aware retriever setup (filter by company or country)
- 🧠 QA chains powered by LLaMA-3 via Ollama
- 🤖 Intelligent entity detection and routing of queries

## 🛠️ Installation

Make sure to install the required Python packages:

```bash
pip install langchain langchain-community faiss-cpu pypdf requests
```

Ensure [Ollama](https://ollama.com/) is installed and running with the following models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

## 📁 File Structure

```
.
├── enhanced_rag_system.py    # Main RAG system implementation
├── vector_store/             # Directory to store FAISS index
├── L&T_report.pdf            # Sample PDF for Indian company (L&T)
└── BYD_COMPANY_LIMITED_Report.pdf  # Sample PDF for Chinese company (BYD)
```

## 🧪 Usage

### 1. Run the Script

```bash
python enhanced_rag_system.py
```

You will be prompted to enter queries interactively.

You can also pass custom PDF paths via CLI:

```bash
python enhanced_rag_system.py path/to/LT_report.pdf path/to/BYD_report.pdf
```

### 2. Sample Queries

```
What are L&T's financial results?
What is BYD's operating revenue?
Tell me about Chinese company performance.
```

## 🔄 Rebuilding the Vector Store

To force reloading and rebuilding the vector store, set `force=True` in the `initialize()` method:

```python
rag.initialize(pdf_paths, force=True)
```

## 🧠 How It Works

1. **PDF Loading**: PDFs are loaded and split into chunks.
2. **Metadata Enrichment**: Each chunk is tagged with the company and corresponding country.
3. **Embedding**: Chunks are embedded using `nomic-embed-text`.
4. **Vector Store**: Embeddings are stored in a FAISS vector index.
5. **Retrievers**: Filters are created for both company and country-specific search.
6. **QA Chains**: For each retriever, a RetrievalQA pipeline is created using the local LLaMA-3 model.
7. **Entity Detection**: Incoming queries are scanned to detect whether a specific company or country is mentioned.
8. **Query Routing**: The query is sent to the relevant retriever(s) and answered using the appropriate QA chain.

## 🧩 Dependencies

- Python ≥ 3.8
- [LangChain](https://docs.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- Ollama Models: `llama3`, `nomic-embed-text`



