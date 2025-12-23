import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
llm = ChatOpenAI(model_name=MODEL, temperature=0)

examples = [{"query": "Когда основана компания?", "answer": "1999 год."},
            {"query": "Кто основатели?", "answer": "Иван Иванов и Мария Петрова."}]

predictions = [{"query": "Когда основана компания?", "result": "Компания создана в 1999 году."},
               {"query": "Кто основатели?", "result": "Иван Иванов и Мария Петрова."}]

def evaluate_qa_with_llm(query: str, prediction: str, reference: str) -> dict:
    prompt = f"""Ты эксперт по оценке качества ответов.

Вопрос: {query}
Ожидаемый ответ: {reference}
Полученный ответ: {prediction}

Оцени, насколько полученный ответ соответствует ожидаемому по смыслу.
Ответь ТОЛЬКО одним словом: CORRECT или INCORRECT"""

    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()
    score = 1.0 if verdict == "CORRECT" else 0.0
    
    return score, verdict

print("=== Оценка с помощью LLM ===\n")
scores_llm = []
for example, prediction in zip(examples, predictions):
    score, verdict = evaluate_qa_with_llm(
        query=example["query"],
        prediction=prediction["result"],
        reference=example["answer"]
    )
    scores_llm.append(score)
    print(f"Вопрос: {example['query']}")
    print(f"Полученный ответ: {prediction['result']}")
    print(f"Ожидаемый ответ: {example['answer']}")
    print(f"Оценка: {score} ({verdict})\n")

avg_score_llm = sum(scores_llm) / len(scores_llm)
print(f"Средняя оценка (LLM): {avg_score_llm:.2f}")

