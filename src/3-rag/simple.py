import re

def build_rag_prompt(docs, question):

    # todo: add cleanup of punctuation, keep words only
    question_cleaned = re.sub(r'[^\w\s]', '', question)
    keywords = [word for word in question_cleaned.lower().split() if len(word) > 4]    
    relevant_docs = [doc for doc in docs if any(keyword in doc.lower() for keyword in keywords)]
    if relevant_docs:
        docs_joined = "\n".join(relevant_docs)
        prompt = f"Вопрос: {question}\nКонтекст:\n{docs_joined}\nОтвет:"
    else:        
        prompt = f"Вопрос: {question}\nОтвет:"

    return prompt


docs = [
    "Нейросети используются для распознавания изображений.",
    "Алгоритмы сортировки применяются в информатике",    
]

question = "Для чего применяются нейросети?"

print(build_rag_prompt(docs, question))