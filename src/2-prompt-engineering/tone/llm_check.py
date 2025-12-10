import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

model_name = "x-ai/grok-code-fast-1"
api_key = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)

system_message = "Ты – помощник, оценивающий стиль ответа по заданным правилам. \
Критерии: (1) тон ответа – деловой, вежливый; (2) речь от третьего лица, \
избегать местоимения 'я'; (3) нет жаргона, только профессиональные термины. \
Ответь в формате JSON с полями: formality_score (0-10), no_first_person (True/False), comment."

user_message = "Оцени стиль следующего ответа:\n\"\"\"\n{answer_text}\n\"\"\""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template(user_message)
])

chain = prompt | llm

response = chain.invoke({"answer_text": 'Проделана работа по улучшению бизнес метрик, клиенты остались довольны'})
print(response.content)
