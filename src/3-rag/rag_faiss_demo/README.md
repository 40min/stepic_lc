# RAG FAISS Demo: Tea Guide Search

A minimal Retrieval-Augmented Generation (RAG) demo using FAISS vector search and BM25 for hybrid querying on Chinese tea guides.

## Features

- Loads tea brewing guide from web page and multiple PDF documents
- Advanced text preprocessing with deduplication and embedding-based similarity filtering
- Creates FAISS semantic search index and BM25 keyword index
- Provides hybrid search combining semantic and keyword matching (60% BM25 + 40% semantic)
- Optimized text chunking (800 chars, 100 overlap) for better context preservation
- Interactive CLI for testing different search modes with real-time comparison

## Requirements

- Python 3.8+
- langchain-community, langchain-text-splitters, langchain-huggingface, faiss-cpu
- PyMuPDF, beautifulsoup4
- **Embedding Model**: cointegrated/rubert-tiny2 (Russian BERT multilingual embeddings)

## Install and Run

Use the Makefile in the root of the repository:

```bash
make install
make tea
make clean-all # to clean up indexes
```

## Search Modes

- `hybrid:query` - Combined BM25 + semantic search (optimized for tea names)
- `bm25:query` - Keyword-only search  
- `semantic:query` - Semantic-only search using multilingual embeddings
- `compare:query` - Compare all modes side by side

## Data Sources

The system loads from multiple sources:

* **Web Source**: [Tea brewing guide](https://tea-mail.by/stati-o-nas/kak-pravilno-zavarivat-kitayskiy-chay/) (brewing techniques)
* **PDF Sources**:
  - `data/tea_guide.pdf` - Chinese tea types and classifications
  - `data/all_you_need_to_know.pdf` - General tea information  
  - `data/locations_ushan.pdf` - Ushan tea locations and regions

## Technical Details

### Text Processing Pipeline
1. **Parallel Loading**: Concurrent loading from all data sources with metadata tagging
2. **Text Cleaning**: Removes artifacts, normalizes whitespace, preserves structure
3. **Deduplication**: 
   - Hash-based exact duplicate removal
   - Embedding-based similarity filtering (threshold: 0.95)
4. **Smart Chunking**: 800-character chunks with 100-character overlap

### Search Architecture
- **FAISS Index**: Semantic search using multilingual embeddings
- **BM25 Index**: Traditional keyword-based retrieval
- **Hybrid Retrieval**: Ensemble retriever with configurable weights (default: 60% BM25 + 40% semantic)

### Embeddings
- **Model**: `cointegrated/rubert-tiny2`
- **Language**: Optimized for Russian and multilingual content
- **Device**: CPU-based encoding with normalized embeddings