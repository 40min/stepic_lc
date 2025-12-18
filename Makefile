# Makefile for Tea Guide RAG Project
# This Makefile provides commands to run the tea_guide.py script,
# clean the tea_index directory, install dependencies, and more.

.PHONY: help install run-tea clean-tea clean-all tea test-deps

# Default target
.DEFAULT_GOAL := help

# Project directories
TEA_DIR := src/3-rag/rag_faiss_demo
DATA_DIR := $(TEA_DIR)/data
TEA_INDEX_DIR := $(DATA_DIR)/tea_index

# Python executable (use uv run for project environment)
PYTHON := uv run --quiet

## Help - Show this help message
help: 
	@echo "Tea Guide RAG Project - Available Commands:"
	@echo ""
	@echo "Setup and Dependencies:"
	@echo "  install      Install project dependencies using uv"
	@echo "  test-deps    Test if dependencies are correctly installed"
	@echo ""
	@echo "Tea Guide Commands:"
	@echo "  tea          Run the tea guide RAG application"
	@echo "  run-tea      Alias for 'tea' command"
	@echo ""
	@echo "Cleaning Commands:"
	@echo "  clean-tea    Remove the tea_index vector database"
	@echo "  clean-all    Remove all generated data (tea_index and cache)"
	@echo ""
	@echo "Examples:"
	@echo "  make install       # Install dependencies first"
	@echo "  make tea           # Run the tea guide application"
	@echo "  make clean-tea     # Clean the vector database"
	@echo ""
	@echo "The tea guide allows you to query Chinese tea information"
	@echo "using a RAG (Retrieval-Augmented Generation) system."

## Install - Install project dependencies
install:
	@echo "Installing project dependencies..."
	uv sync
	@echo "Dependencies installed successfully!"

## Test Dependencies - Check if key dependencies are available
test-deps:
	@echo "Testing key dependencies..."
	@$(PYTHON) python -c "import langchain_community; print('✓ langchain-community')" || echo "✗ langchain-community missing"
	@$(PYTHON) python -c "import langchain_huggingface; print('✓ langchain-huggingface')" || echo "✗ langchain-huggingface missing"
	@$(PYTHON) python -c "import faiss; print('✓ faiss-cpu')" || echo "✗ faiss-cpu missing"
	@$(PYTHON) python -c "from langchain_community.document_loaders import PyMuPDFLoader; print('✓ pymupdf')" || echo "✗ pymupdf missing"
	@$(PYTHON) python -c "import bs4; print('✓ beautifulsoup4')" || echo "✗ beautifulsoup4 missing"
	@echo "Dependency test completed."

## Tea - Run the tea guide RAG application
tea: run-tea

## Run Tea - Run the tea guide from the correct directory
run-tea:
	@echo "Starting Tea Guide RAG Application..."
	@echo "Working directory: $(TEA_DIR)"
	@echo "Data directory: $(DATA_DIR)"
	@if [ ! -f "$(DATA_DIR)/tea_guide.pdf" ]; then \
		echo "Warning: $(DATA_DIR)/tea_guide.pdf not found!"; \
		echo "Please ensure the tea guide PDF is in the data directory."; \
	fi
	@if [ ! -d "$(TEA_INDEX_DIR)" ]; then \
		echo "Vector database not found, it will be created on first run."; \
	fi
	@echo ""
	cd $(TEA_DIR) && $(PYTHON) tea_guide.py

## Clean Tea Index - Remove the vector database
clean-tea:
	@echo "Cleaning tea index database..."
	@if [ -d "$(TEA_INDEX_DIR)" ]; then \
		rm -rf $(TEA_INDEX_DIR); \
		echo "✓ Removed $(TEA_INDEX_DIR)"; \
	else \
		echo "Tea index directory not found, nothing to clean."; \
	fi

## Setup - One-time setup (install deps + clean)
setup: install clean-tea
	@echo "Setup completed! You can now run 'make tea' to start the application."

## Info - Show project information
info:
	@echo "Tea Guide RAG Project Information:"
	@echo "================================="
	@echo "Script location: $(TEA_DIR)/tea_guide.py"
	@echo "Data directory: $(DATA_DIR)"
	@echo "Vector database: $(TEA_INDEX_DIR)"
	@echo "PDF file: $(DATA_DIR)/tea_guide.pdf"
	@echo ""
	@echo "Dependencies required:"
	@echo "  - langchain-community (document loaders, FAISS)"
	@echo "  - langchain-huggingface (embeddings)"
	@echo "  - langchain-text-splitters (text splitting)"
	@echo "  - faiss-cpu (vector database)"
	@echo "  - pymupdf (PDF loading)"
	@echo "  - beautifulsoup4 (web scraping)"
	@echo ""
	@echo "The application loads tea information from:"
	@echo "  1. Web page: https://tea-mail.by/stati-o-nas/kak-pravilno-zavarivat-kitayskiy-chay/"
	@echo "  2. PDF file: $(DATA_DIR)/tea_guide.pdf"
	@echo ""
	@echo "Vector database features:"
	@echo "  - Multilingual embeddings (intfloat/multilingual-e5-small)"
	@echo "  - Optimized chunking (800 chars, 100 overlap)"
	@echo "  - Interactive querying with similarity search"