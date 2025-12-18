# RAG FAISS Demo: Tea Guide Search

A minimal Retrieval-Augmented Generation (RAG) demo using FAISS vector search and BM25 for hybrid querying on Chinese tea guides.

## Features

- Loads tea brewing guide from web page and PDF document
- Creates FAISS semantic search index and BM25 keyword index
- Provides hybrid search combining semantic and keyword matching
- Interactive CLI for testing different search modes

## Requirements

- Python 3.8+
- langchain-community, langchain-text-splitters, langchain-huggingface, faiss-cpu
- PyMuPDF, beautifulsoup4

## Install and Run

Some can use Makefile in the root of repo

```bash
make install
make tea
make clean-all # to clean up indexes
```

Search modes:
- `hybrid:query` - Combined BM25 + semantic search
- `bm25:query` - Keyword-only search
- `semantic:query` - Semantic-only search
- `compare:query` - Compare all modes

## Data Sources

* [Tea brewing guide](https://tea-mail.by/stati-o-nas/kak-pravilno-zavarivat-kitayskiy-chay/)
* `data/tea_guide.pdf` - Chinese tea types guide