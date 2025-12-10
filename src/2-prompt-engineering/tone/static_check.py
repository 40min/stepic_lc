import os
from dotenv import load_dotenv
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

model_name = "x-ai/grok-code-fast-1"
api_key = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)

def check_style(text: str) -> str:
    """Проверяет наличие слова 'ты' в ответе"""
    if re.search(r'\bты\b|\bтебя\b|\bтвой\b|\bтвоя\b|\bтвоё\bнажми\bпроверь\b', text, re.IGNORECASE):
        print(f"⚠️ НАРУШЕНИЕ СТИЛЯ: обнаружено обращение на 'ты' в ответе: {text[:100]}...")
    return text 

# Тест 1: правильный стиль
print("Тест 1: правильный стиль")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты помощник компании. Обращайся к клиенту на 'Вы'. Отвечай кратко, не более 100 символов."),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser() | RunnableLambda(check_style)  # ← Проверка стиля
result = chain.invoke({"question": "Здравствуйте, я не могу разобраться с оплатой, куда нажимать?"})
print(f"Ответ: {result}\n")

# Тест 2: неправильный стиль (специально провоцируем)
print("\nТест 2: провоцируем нарушение")
bad_prompt = ChatPromptTemplate.from_messages([
    ("system", "Общайся неформально, на 'ты'. Отвечай кратко, не более 100 символов."),
    ("human", "{question}")
])

bad_chain = bad_prompt | llm | StrOutputParser() | RunnableLambda(check_style)
result = bad_chain.invoke({"question": "Здравствуйте, я не могу разобраться с оплатой, куда нажимать?"})
print(f"Ответ: {result}")
