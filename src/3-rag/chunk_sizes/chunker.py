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
    "столица Индонезии",
    "население Индонезии",
    "какие острова входят в состав Индонезии",
    "история провозглашения независимости Индонезии",
    "президент Индонезии",
    "религиозный состав населения Индонезии",
    "вулканы Индонезии",
    "Кракатау извержение",
    "административное деление провинции Индонезии",
    "экономика Индонезии",
    "индонезийский язык государственный",
    "яванцы крупнейший народ Индонезии",
    "период Нового порядка Сухарто",
    "буддистский храм архитектура",
    "Движение 30 сентября 1965 переворот",
    "олимпийские медали Индонезии",
    "бадминтон",
]


def output_sample_text(text, max_len_of_sample):
    snippet = text[:max_len_of_sample].replace("\n", " ")
    is_cut = len(text) > max_len_of_sample
    print(f"   📄 Пример: {snippet}{"..." if is_cut else ""}")


def run_tests(embedding_model, configs, docs, questions, max_len_of_sample=500):
    # Создаем базы данных для каждой конфигурации
    dbs = []
    for cfg in configs:
        splitter = make_splitter(cfg)
        chunks = []
        for doc in docs:
            for chunk_text in splitter.split_text(doc.page_content):
                md = (doc.metadata or {}).copy() if hasattr(doc, "metadata") else {}
                chunks.append(Document(page_content=chunk_text, metadata=md))

        print(f"📊 Создание БД для конфигурации: {cfg['name']} "
              f"(chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}), "
              f"всего чанков={len(chunks)}")

        db = FAISS.from_documents(chunks, embedding_model)
        dbs.append(db)

    print("\n" + "="*80)
    print("🚀 НАЧАЛО ТЕСТИРОВАНИЯ ВОПРОСОВ")
    print("="*80 + "\n")

    # Инициализация статистики побед для каждой конфигурации
    win_stats = {cfg['name']: 0 for cfg in configs}

    for q in questions:
        print(f"🔍 Вопрос: {q}")
        print("-" * 40)

        results_per_config = []
        for i, cfg in enumerate(configs):
            k = cfg.get("k", 2)
            docs_and_scores = dbs[i].similarity_search_with_score(q, k=k)
            scores = [score for _, score in docs_and_scores]
            avg_score = sum(scores) / len(scores) if scores else float('inf')
            results_per_config.append({
                'config': cfg,
                'scores': scores,
                'docs_and_scores': docs_and_scores,
                'avg_score': avg_score
            })

        # Сортируем по среднему скору (низкий скор - лучше для FAISS distance)
        sorted_results = sorted(results_per_config, key=lambda x: x['avg_score'])

        # Лучшая конфигурация (самый низкий средний скор)
        best = sorted_results[0]
        win_stats[best['config']['name']] += 1
        
        print(f"🏆 Лучшая конфигурация: {best['config']['name']} (avg_score={best['avg_score']:.4f})")
        if best['docs_and_scores']:            
            original_text = best['docs_and_scores'][0][0].page_content            
            output_sample_text(original_text, max_len_of_sample)

        # Худшая конфигурация (самый высокий средний скор)
        worst = sorted_results[-1]
        print(f"👎 Худшая конфигурация: {worst['config']['name']} (avg_score={worst['avg_score']:.4f})")
        if worst['docs_and_scores']:
            original_text = worst['docs_and_scores'][0][0].page_content            
            output_sample_text(original_text, max_len_of_sample)            

        print("\n" + "-"*60 + "\n")

    # Вывод итоговой статистики
    print("\n" + "="*80)
    print("📈 ИТОГОВАЯ СТАТИСТИКА ПОБЕД")
    print("="*80 + "\n")
    
    # Сортируем по количеству побед
    sorted_stats = sorted(win_stats.items(), key=lambda x: x[1], reverse=True)
    
    total_questions = len(questions)
    for rank, (config_name, wins) in enumerate(sorted_stats, 1):
        percentage = (wins / total_questions) * 100
        bar_length = int(percentage / 2)  # Масштаб для визуализации
        bar = "█" * bar_length
        print(f"{rank}. {config_name:20s} | {wins:2d}/{total_questions} побед ({percentage:5.1f}%) {bar}")
    
    print("\n" + "="*80)
    winner = sorted_stats[0]
    print(f"🎉 ПОБЕДИТЕЛЬ: {winner[0]} с {winner[1]} победами из {total_questions} вопросов!")
    print("="*80 + "\n")

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
