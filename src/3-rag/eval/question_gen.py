import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
llm = ChatOpenAI(model_name=MODEL, temperature=0)

documents = [Document(page_content="Фирма ООО 'Одуванчик' основана в 2015 году", metadata={'id': 1}),
             Document(page_content="Директор ООО 'Одуванчик' Смирнов Иван Петрович", metadata={'id': 2}),
             Document(page_content="Адрес ООО 'Одуванчик' ул.Ленина д.5", metadata={'id': 3})]

prompt = ChatPromptTemplate.from_template("""
Ты – помощник, который придумывает вопросы по тексту.
Прочитай текст и сформулируй вопрос, на который этот текст отвечает. Затем дай короткий точный ответ на этот вопрос.

Формат:
{{
  "questions": [...],
  "answers": [...],
  "facts": [...],
  "ground_truth_doc_id": "{doc_id}"
}}

Требования:
- Сгенерируй 2–3 вопроса разной сложности по тексту.
- Дай точные ответы, опираясь только на факты из текста.
- Выдели список ключевых фактов, которые должен найти retriever.
- Не придумывай информацию вне текста.

Текст:
{content}
""")

gen_chain = prompt | llm

examples = []
for doc in documents:
    ex = gen_chain.invoke({"content": doc.page_content, "doc_id": doc.metadata["id"]})
    examples.append(ex.content)

print(examples)
