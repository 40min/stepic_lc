# импорты
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from utils import (
    make_splitter, 
    load_data_from_url,
    clean_wikipedia_text,
)

# схема конфигураций
CONFIGS = [
    # Конфигурации для разных типов поиска
    {"name": "sparse_optimized", "chunk_size": 300, "chunk_overlap": 30, "note": "Оптимизировано для BM25"},
    {"name": "dense_optimized", "chunk_size": 800, "chunk_overlap": 100, "note": "Оптимизировано для векторного поиска"}, 
    {"name": "hybrid_balanced", "chunk_size": 500, "chunk_overlap": 50, "note": "Компромисс для гибридного поиска"},
]

SRC_URL = "https://ru.wikipedia.org/wiki/%D0%98%D0%BD%D0%B4%D0%BE%D0%BD%D0%B5%D0%B7%D0%B8%D1%8F"
QUESTIONS = [
    "Что такое Индонезия?",
    "Какая площадь Индонезии?",
    "Какое население Индонезии?",
    "Какая столица Индонезии?",
    "Какая религия в Индонезии?",
    "Какие языки в Индонезии?",
    "Какие национальности в Индонезии?",
    "Какие города в Индонезии?",
    "Какие острова в Индонезии?",    
    "Какие достопримечательности в Индонезии?",    
]



def run_tests(embedding_model, configs, docs, questions):
    for cfg in configs:
        splitter = make_splitter(cfg)
        chunks = []
        for doc in docs:
            for chunk_text in splitter.split_text(doc.page_content):
                md = (doc.metadata or {}).copy() if hasattr(doc, "metadata") else {}
                chunks.append(Document(page_content=chunk_text, metadata=md))

        print(f"\nКонфигурация: {cfg['name']} "
              f"(chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}), "
            f"всего чанков={len(chunks)}")

        db = FAISS.from_documents(chunks, embedding_model)

        for q in questions:
            k = cfg.get("k", 2)
            docs_and_scores = db.similarity_search_with_score(q, k=k)
            print(f"Q: {q}")
            for doc, score in docs_and_scores:                
                snippet = doc.page_content[:200].replace("\n", " ")
                print(f" - найден фрагмент (score={score:.4f}): {snippet}... \n")

def main():
    print("Загрузка данных...")
    docs = load_data_from_url(SRC_URL)
    docs_cleaned = [Document(page_content=clean_wikipedia_text(doc.page_content), metadata=doc.metadata) for doc in docs]
    print("Загрузка модели...")
    embedding_model = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    print("Запуск тестов...")
    run_tests(embedding_model, CONFIGS, docs_cleaned, QUESTIONS)

if __name__ == "__main__":
    main()
