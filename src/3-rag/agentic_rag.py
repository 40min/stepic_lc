import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_API_MODEL", "x-ai/grok-4-fast")

llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)

documents = [Document(page_content="Фирма ООО 'Одуванчик' основана в 2015 году"),
             Document(page_content="Директор ООО 'Одуванчик' Смирнов Иван Петрович"),
             Document(page_content="Адрес ООО 'Одуванчик' ул.Ленина д.5")]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

search_tool = create_retriever_tool(
    retriever=retriever,
    name="search_knowledge_base",
    description="Ищет информацию в базе фирме ООО 'Одуванчик'"
)

agent = create_agent(
    model=llm,
    system_prompt="Ты полезный ассистент. Когда тебя спрашивают про ООО 'Одуванчик' используй инструмент search_knowledge_base для поиска информации.",
    tools=[search_tool]
).with_config(recursion_limit=11)


response = agent.invoke({"messages": [{"role": "user", "content": "Привет! Как дела]"}]})
print(response['messages'][-1].content) # ответит сразу без ретривера

response = agent.invoke({"messages": [{"role": "user", "content": "Что ты знаешь про ООО 'Одуванчик'?"}]})
print(response['messages'][-1].content) # полезет в ретривер за информацией

                  
