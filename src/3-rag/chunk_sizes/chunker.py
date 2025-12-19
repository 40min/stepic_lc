# импорты
import os
import click
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from utils import (
    make_splitter, 
    load_data_from_url,
    clean_wikipedia_text,
)
from llm_assessor import LLMAssessor
from evaluators import ScoreBasedEvaluator, LLMBasedEvaluator

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


def run_tests(
        embedding_model, 
        configs, 
        docs, 
        questions, 
        llm_model,
        api_key,
        max_len_of_sample=500,
        evaluation_mode="score-based",
):
    """
    Run tests with different evaluation modes
    
    Args:
        embedding_model: Embedding model for vector search
        configs: List of chunking configurations
        docs: List of documents to chunk
        questions: List of test questions
        max_len_of_sample: Max length for sample output
        evaluation_mode: "score-based" or "llm-based"
        llm_model: Model name for LLM assessment (optional)
    """
    # Validate evaluation mode
    if evaluation_mode not in ["score-based", "llm-based"]:
        raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")
    
    # Initialize evaluator based on mode
    if evaluation_mode == "llm-based":
        assessor = LLMAssessor(model_name=llm_model, api_key=api_key)
        evaluator = LLMBasedEvaluator(assessor)
        print(f"✅ LLM evaluator initialized")
    else:
        evaluator = ScoreBasedEvaluator()
        print(f"✅ Score-based evaluator initialized")
    
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
        print(f"📊 Evaluation Mode: {evaluation_mode}")
        print("-" * 40)

        # Use evaluator to get results
        k = configs[0].get("k", 2)
        sorted_results = evaluator.evaluate(dbs, configs, q, k=k)
        
        # Update winner tracking and output
        best = sorted_results[0]
        config_name = best['config']['name']
        win_stats[config_name] += 1
        
        # Output results based on mode
        if evaluation_mode == "score-based":
            print(f"🏆 Лучшая конфигурация: {config_name} (avg_score={best['avg_score']:.4f})")
            if best.get('docs_and_scores'):
                original_text = best['docs_and_scores'][0][0].page_content
                output_sample_text(original_text, max_len_of_sample)
        elif evaluation_mode == "llm-based":
            print(f"🏆 Лучшая конфигурация: {config_name} (llm_score={best['llm_score']}/100)")
            print(f"   💡 Reasoning: {best['reasoning']}")
            if best.get('doc'):
                output_sample_text(best['doc'].page_content, max_len_of_sample)
        
        # Show worst configuration
        worst = sorted_results[-1]
        worst_config_name = worst['config']['name']
        if evaluation_mode == "score-based":
            print(f"👎 Худшая конфигурация: {worst_config_name} (avg_score={worst['avg_score']:.4f})")
            if worst.get('docs_and_scores'):
                original_text = worst['docs_and_scores'][0][0].page_content
                output_sample_text(original_text, max_len_of_sample)
        elif evaluation_mode == "llm-based":
            print(f"👎 Худшая конфигурация: {worst_config_name} (llm_score={worst['llm_score']}/100)")
            print(f"   💡 Reasoning: {worst['reasoning']}")
            if worst.get('doc'):
                output_sample_text(worst['doc'].page_content, max_len_of_sample)
        
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

@click.command()
@click.option(
    '--eval-mode',
    type=click.Choice(['score-based', 'llm-based']),
    default='score-based',
    help='Evaluation mode for chunk quality'
)
def main(eval_mode):
    """Evaluate RAG chunking strategies"""
    # Load environment
    load_dotenv()
    
    print("Загрузка данных...")
    docs = load_data_from_url(SRC_URL)
    docs_cleaned = [
        Document(
            page_content=clean_wikipedia_text(doc.page_content), 
            metadata=doc.metadata
        ) 
        for doc in docs
    ]
    
    print("Загрузка модели...")
    embedding_model = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")

    llm_model = os.getenv("OPENROUTER_API_MODEL", "x-ai/grok-4-fast")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print(f"Запуск тестов (режим: {eval_mode})...")
    run_tests(
        embedding_model, 
        CONFIGS, 
        docs_cleaned, 
        QUESTIONS,
        llm_model=llm_model,
        api_key=api_key,
        evaluation_mode=eval_mode,
    )

if __name__ == "__main__":
    main()
