import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()


api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_API_MODEL", "x-ai/grok-4-fast")

llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)

retriever = WikipediaRetriever(
    wiki_client=None, 
    lang="ru", 
    top_k_results=2,
)

# own retreiver
#
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document

# pages = ["День рождения Алексея Попова 05.09.1997", 
#          "День рождения Данилы Никитина 01.06.1986", 
#          "День рождения Ильи Муромского 02.03.1991"]
# docs = [Document(page_content=t) for t in pages]

# embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vector_store = FAISS.from_documents(docs, embed_model)

# Промпт для RAG
prompt = ChatPromptTemplate.from_template("Ответь на вопрос, используя только следующий контекст:\nКонтекст: {context}\nВопрос: {question}\nОтвет:")

# Функция для форматирования документов
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

query = "Что такое LangChain?"
# retreived =retriever.invoke(query)
response = chain.invoke(query)
print(response.content)
