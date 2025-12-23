import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from langchain_openai import ChatOpenAI

from datasets import Dataset
load_dotenv()

# Настройка LLM для оценки
model_name = os.getenv("OPENROUTER_MODEL", "x-ai/grok-code-fast-1")
api_key = os.getenv("OPENROUTER_API_KEY")
ragas_llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)

# Подготовка данных для оценки
data = {
    "user_input": [
        "Когда основана компания?",
        "Кто основатели?"
    ],
    "response": [
        "Компания основана в 1999 году Иваном Ивановым и Марией Петровой.",
        "Основатели — Иван Иванов и Мария Петрова."
    ],
    "retrieved_contexts": [
        ["Компания основана в 1999 году основателями: Иван Иванов и Мария Петрова."],
        ["Основатели компании — Иван Иванов и Мария Петрова, 1999 год."]
    ],
    "reference": [
        "1999 год.",
        "Иван Иванов и Мария Петрова."
    ]
}

dataset = Dataset.from_dict(data)

# Оценка с помощью RAGAS
result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),           # Соответствие контексту (нет галлюцинаций)
        AnswerRelevancy(),        # Релевантность ответа вопросу
        ContextRecall()           # Полнота извлеченного контекста
    ],
    llm=ragas_llm
)

print("=== Результаты RAGAS ===")
scores = result._repr_dict  # type: ignore[attr-defined]
print(f"Faithfulness: {scores['faithfulness']:.3f}")
print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
print(f"Context Recall: {scores['context_recall']:.3f}")
