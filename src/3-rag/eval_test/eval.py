import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from eval_data import documents, ground_truth_docs, golden_vs_predicted_answers


def precision_at_k(retriever, k: int) -> float:
    precision_total = 0
    for question, true_docs in ground_truth_docs.items():
        found_docs = retriever.invoke(question)
        found_ids = [d.metadata["source"] for d in found_docs[:k]]
        relevant_found = sum(1 for doc_id in true_docs if doc_id in found_ids)
        precision = relevant_found / k
        precision_total += precision

        print(f"Вопрос: {question}")
        print(f"  Эталонные документы: {true_docs}")
        print(f"  Найденные документы: {found_ids}")
        print(f"  Precision@{k}: {precision:.2f}\n")

    avg_precision = precision_total / len(ground_truth_docs)
    
    return avg_precision


def recall_at_k(retriever, k: int) -> float:
    recall_total = 0
    for question, true_docs in ground_truth_docs.items():
        found_docs = retriever.invoke(question)
        found_ids = [d.metadata["source"] for d in found_docs[:k]]
        relevant_found = sum(1 for doc_id in true_docs if doc_id in found_ids)
        recall = relevant_found / len(true_docs) if true_docs else 0
        recall_total += recall

        print(f"Вопрос: {question}")
        print(f"  Эталонные документы: {true_docs}")
        print(f"  Найденные документы: {found_ids}")
        print(f"  Recall@{k}: {recall:.2f} ({relevant_found}/{len(true_docs)})\n")

    avg_recall = recall_total / len(ground_truth_docs)
    
    return avg_recall


def evaluate_qa_with_llm(
        llm: ChatOpenAI, 
        query: str, 
        prediction: str, 
        reference: str
) -> tuple[float, str]:
    prompt = f"""Ты эксперт по оценке качества ответов.

Вопрос: {query}
Ожидаемый ответ: {reference}
Полученный ответ: {prediction}

Оцени, насколько полученный ответ соответствует ожидаемому по смыслу.
Ответь ТОЛЬКО одним словом: CORRECT или INCORRECT"""

    response = llm.invoke(prompt)
    verdict = response.content.strip().upper() # type: ignore
    score = 1.0 if verdict == "CORRECT" else 0.0
    
    return score, verdict


def evaluate_faithfulness(llm) -> float:    
    scores_llm = []
    for query, answers in golden_vs_predicted_answers.items():
        score, verdict = evaluate_qa_with_llm(
            llm=llm,
            query=query,
            prediction=answers["predicted"],
            reference=answers["golden"],                    
        )
        scores_llm.append(score)
        print(f"Вопрос: {query}")
        print(f"Ожидаемый ответ: {answers['golden']}")
        print(f"Полученный ответ: {answers['predicted']}")
        print(f"Оценка: {score} ({verdict})\n")

    avg_score_llm = sum(scores_llm) / len(scores_llm)

    return avg_score_llm




if __name__ == "__main__":

    K = 2

    # make retriever
    embed_model = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    vector_store = FAISS.from_documents(documents, embed_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": K})

    # evaluate
    avg_precision = precision_at_k(retriever, k=K)
    avg_recall = precision_at_k(retriever, k=K)
    
    print(f"Средняя Precision@{avg_precision}: {avg_precision:.2f}")
    print(f"Средняя Recall@{avg_recall}: {avg_recall:.2f}")


    # evaluate llm
    api_url = os.getenv("API_URL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_API_MODEL", "x-ai/grok-4-fast")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        # base_url=api_url,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,        
    )

    print("=== Оценка с помощью LLM ===\n")
    avg_score_llm = evaluate_faithfulness(llm)
    print(f"Средняя оценка: {avg_score_llm:.2f}")    
