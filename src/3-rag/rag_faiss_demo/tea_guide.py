from pathlib import Path
import re
import pickle


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from loaders import load_web_page, load_pdf
from preprosess import (
    clean_text, 
    dedupe_by_embedding, 
    filter_and_dedup, 
)



EMBED_MODEL_NAME = "cointegrated/rubert-tiny2"
# EMBED_MODEL = "intfloat/multilingual-e5-small"


def create_db():
    """Create optimized vector database with BM25 index"""
    html_docs = load_web_page()
    pdf_docs = load_pdf()

    all_docs = html_docs + pdf_docs

    # cleaning
    for doc in all_docs:
        doc.page_content = clean_text(doc.page_content)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    all_docs_filtered = dedupe_by_embedding(filter_and_dedup(all_docs), embedding_model=embedding_model)
    
    print(f"Всего документов: {len(all_docs_filtered)}")

    # OPTIMIZED: Larger chunks with overlap for better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    splitted_docs = text_splitter.split_documents(all_docs_filtered)
    print(f"Было документов: {len(all_docs_filtered)}, стало фрагментов: {len(splitted_docs)}")
    
    # Show sample chunk for verification
    if splitted_docs:
        print(f"\nПример чанка (первые 200 символов):\n{splitted_docs[0].page_content[:200]}...")
        print(f"Метаданные чанка: {splitted_docs[0].metadata}")    
    
    # Create semantic search index (FAISS)
    print("\nСоздание семантического индекса (FAISS)...")
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = FAISS.from_documents(splitted_docs, embed_model)
    vector_store.save_local("data/tea_index")
    print("FAISS индекс сохранен")
    
    # Create BM25 keyword index
    print("\nСоздание BM25 индекса (keyword search)...")
    bm25_retriever = BM25Retriever.from_documents(splitted_docs)
    bm25_retriever.k = 3  # Return top 3 results by default
    
    # Save BM25 index
    with open("data/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("BM25 индекс сохранен")
    
    print("\n✅ Обе базы созданы и сохранены в data/")
    return vector_store, bm25_retriever

def load_db():
    """Load both FAISS and BM25 indexes"""
    print("Загрузка индексов...")
    
    # Load FAISS
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.load_local(
        "data/tea_index", 
        embed_model, 
        allow_dangerous_deserialization=True
    )
    
    # Load BM25
    with open("data/bm25_index.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
    
    print("✅ Индексы загружены")
    return vector_store, bm25_retriever

def hybrid_search(vector_store: FAISS, bm25_retriever: BM25Retriever, 
                  query: str, k: int = 3, bm25_weight: float = 0.5):
    """
    Hybrid search combining BM25 (keyword) and semantic search
    
    Args:
        bm25_weight: 0.0 = pure semantic, 1.0 = pure BM25, 0.5 = balanced
    """
    # Create retrievers
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # Combine with EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, 1 - bm25_weight]  # BM25 weight, Semantic weight
    )
    
    # Get results
    docs = ensemble_retriever.invoke(query)
    return docs[:k]  # Return top k

def db_lookup(vector_store: FAISS, bm25_retriever: BM25Retriever, 
              query: str, k: int = 3, mode: str = 'hybrid', max_to_output: int = 700):
    """
    Search with different modes
    
    Args:
        mode: 'hybrid' (default), 'semantic', 'bm25'
    """
    print(f"\n{'='*50} 🔍 ПОИСК {'='*50}")
    print(f"📝 Запрос: {query}")
    print(f"🎯 Режим: {mode.upper()}")
    print(f"{'='*70}\n")
    
    if mode == 'hybrid':
        # Hybrid: 60% BM25 + 40% semantic (favor keywords for tea names)
        docs_found = hybrid_search(vector_store, bm25_retriever, query, k=k, bm25_weight=0.6)
        # Convert to list of (doc, None) tuples for consistent handling
        docs_found = [(doc, None) for doc in docs_found]
        
    elif mode == 'bm25':
        # Pure keyword search
        bm25_retriever.k = k
        docs = bm25_retriever.invoke(query)
        docs_found = [(doc, None) for doc in docs]
        
    elif mode == 'semantic':
        # Pure semantic search with scores
        docs_found = vector_store.similarity_search_with_score(query, k=k)
    
    else:
        print(f"❌ Неизвестный режим: {mode}")
        return
    
    # Display results
    for i, doc_tuple in enumerate(docs_found, 1):
        doc = doc_tuple[0]
        score = doc_tuple[1] if len(doc_tuple) > 1 and doc_tuple[1] is not None else None
        
        # Add ranking emojis
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i}")
        print(f"{rank_emoji} Результат {i}", end="")
        if score is not None:
            print(f" (релевантность: {score:.4f})", end="")
        print()
        
        source_type = doc.metadata.get('source_type', 'unknown')
        topic = doc.metadata.get('topic', 'unknown')
        source_emoji = {"web": "🌐", "pdf": "📄"}.get(source_type, "❓")
        topic_emoji = {"brewing_guide": "☕", "tea_types": "🍵"}.get(topic, "📋")
        print(f"{source_emoji} Источник: {doc.metadata.get('source_type', 'unknown')} | "
              f"{topic_emoji} Тема: {doc.metadata.get('topic', 'unknown')}")
        if 'page' in doc.metadata:
            print(f"📄 Страница: {doc.metadata['page']}")
        print(f"📋 Полные метаданные: {doc.metadata}")
        
        # Highlight query terms for BM25/hybrid
        content = doc.page_content[:max_to_output]
        
        query_terms = query.lower().split()
        for term in query_terms:
            if len(term) > 2:  # Skip short words                    
                content = re.sub(
                    f'({re.escape(term)})',
                    r'🔥\1🔥',
                    content,
                    flags=re.IGNORECASE
                )
        
        print(f"\n📖 Текст ({len(doc.page_content)} символов):")
        print(content)
        if len(doc.page_content) > max_to_output:
            print(f"✂️ ... [показано {max_to_output} из {len(doc.page_content)} символов]")
        print(f"\n{'-'*50} 🌟 {'-'*50}\n")

def compare_modes(vector_store: FAISS, bm25_retriever: BM25Retriever, query: str):
    """Compare all three search modes"""
    print(f"\n{'#'*40} 🔄 СРАВНЕНИЕ РЕЖИМОВ {'#'*40}")
    print(f"📊 Для запроса: '{query}'")
    print(f"{'#'*70}")
    
    for mode in ['bm25', 'semantic', 'hybrid']:
        db_lookup(vector_store, bm25_retriever, query, k=2, mode=mode, max_to_output=700)
        if mode != 'hybrid':
            input("Нажмите Enter для следующего режима...")

def test_queries(vector_store: FAISS, bm25_retriever: BM25Retriever):
    """Test with sample queries including tea names"""
    test_cases = [
        ("гайвань", "hybrid"),
        ("Железная богиня милосердия", "hybrid"),
        ("как заваривать белый чай", "semantic"),
        ("температура воды для зеленого чая", "semantic"),
    ]
    
    print("\n" + "="*45 + " 🧪 ТЕСТИРОВАНИЕ " + "="*45)
    print("🎯 КАЧЕСТВА ПОИСКА")
    print("="*70)
    
    for query, mode in test_cases:
        print(f"\n🧪 Тест: '{query}' (режим: {mode.upper()})")
        db_lookup(vector_store, bm25_retriever, query, k=2, mode=mode, max_to_output=700)
        input("Нажмите Enter для следующего запроса...")

def main():
    # Check if both indexes exist
    faiss_exists = Path("data/tea_index").exists()
    bm25_exists = Path("data/bm25_index.pkl").exists()
    
    if not (faiss_exists and bm25_exists):
        print("⚠️  Индексы отсутствуют, создаём базу данных...")
        vector_store, bm25_retriever = create_db()
        
        # Run tests after creation
        print("\n" + "="*70)
        response = input("Хотите протестировать базу? (y/n): ").strip().lower()
        if response == 'y':
            test_queries(vector_store, bm25_retriever)
    else:
        print("✅ Индексы найдены, загружаем базу данных")
        vector_store, bm25_retriever = load_db()

    # Interactive mode
    print("\n" + "="*45 + " 🍵 ГИБРИДНЫЙ ПОИСК " + "="*45)
    print("🎮 ДОСТУПНЫЕ РЕЖИМЫ:")
    print("  🔄 'hybrid:запрос'   - BM25 + семантика (для названий чая)")
    print("  🔍 'bm25:запрос'     - только keyword search")
    print("  🧠 'semantic:запрос' - только семантический поиск")
    print("  📊 'compare:запрос'  - сравнить все режимы")
    print("  ⚡ 'запрос'          - semantic по умолчанию")
    print("="*70)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ (Ctrl+C для выхода)")
    print("="*70)
    
    while True:
        try:
            user_input = input("\nВведите запрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nЗавершение работы.")
            break
        
        if not user_input:
            continue
        
        # Parse mode if specified
        if ':' in user_input:
            parts = user_input.split(':', 1)
            if parts[0].strip() in ['hybrid', 'bm25', 'semantic', 'compare']:
                mode = parts[0].strip()
                query = parts[1].strip()
            else:
                mode = 'semantic'
                query = user_input
        else:
            mode = 'semantic'
            query = user_input
        
        if mode == 'compare':
            compare_modes(vector_store, bm25_retriever, query)
        else:
            db_lookup(vector_store, bm25_retriever, query, k=3, mode=mode, max_to_output=700)

if __name__ == "__main__":
    main()
