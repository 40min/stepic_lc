
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence


load_dotenv()
model = os.getenv("OPENAI_API_MODEL", "gpt-5-mini")



# Base 
prompt = ChatPromptTemplate.from_template("Переведи на английский: {текст}")

llm = ChatOpenAI(model=model, temperature=0)
chain = RunnableSequence(first=prompt, last=llm)
result = chain.invoke({"текст": "Доброе утро"})

print(result.content)

# with lcel
prompt = ChatPromptTemplate.from_template("Привет. Меня зовут {name}")
basic_chain = prompt | llm

result = basic_chain.invoke({"name": "Вася"})
print(result)

# with RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough

# Просто передаёт входные данные дальше
chain = RunnablePassthrough() | llm
result = chain.invoke("Привет!")  # "Привет!" → llm
print(result)

# with RunnableLambda
from langchain_core.runnables import RunnableLambda

def uppercase(text):
    return text.upper()

chain = RunnableLambda(uppercase) | llm
chain.invoke("hello")  # "HELLO" → llm


# with StrOutputParser
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"текст": "Доброе утро"})
print(result)

# with dicts
# 1. Создать словарь из входных данных
chain = {"name": RunnablePassthrough()} | prompt | llm

result = chain.invoke("Ваня") # → {"name": "Ваня"} → prompt → llm
print(result)

# 2. assign
chain = (
    {"name": RunnablePassthrough()}
    | RunnablePassthrough.assign(second_name=lambda x: "Иванов")
)

result = chain.invoke("Вася") 
print(result)

# 3. pick
chain = (
    {"name": RunnablePassthrough(), "age": lambda x: 25, "city": lambda x: "Москва"}
    | RunnablePassthrough().pick(["name", "city"])
)

result = chain.invoke("Иван")
print(result)
# # вывод: {'name': 'Иван', 'city': 'Москва'}


# with RunnableParallel
from langchain_core.runnables import RunnableParallel

prompt1 = ChatPromptTemplate.from_template("Плюсы {topic}")
prompt2 = ChatPromptTemplate.from_template("Минусы {topic}")

# Выполняются ОДНОВРЕМЕННО
parallel_chain = RunnableParallel(
    pros=prompt1 | llm | StrOutputParser(),
    cons=prompt2 | llm | StrOutputParser()
)

result = parallel_chain.invoke({"topic": "Python"})
print(result)
# result = {"pros": "...", "cons": "..."}

# with RunnableBranch
from langchain_core.runnables import RunnableBranch

prompt_for_questions = ChatPromptTemplate.from_template("вопрос: {text}")
prompt_for_statements = ChatPromptTemplate.from_template("не вопрос: {text}")

def is_question(x):
    return "?" in x["text"]

branch = RunnableBranch(
    (is_question, prompt_for_questions | llm),  # Если вопрос
    prompt_for_statements | llm  # Иначе (default)
)

chain = {"text": RunnablePassthrough()} | branch | StrOutputParser()

result = chain.invoke({"text": "Какая сегодня погода?"})
print(result)
