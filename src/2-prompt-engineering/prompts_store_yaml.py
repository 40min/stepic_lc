import os
from pyexpat import model
from dotenv import load_dotenv
import yaml
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_openai import ChatOpenAI

load_dotenv()

with open("prompts.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

mode = 'EN-RU'
system_prompt = data["prompts"]["my_chain"]["modes"][mode]["system"]
user_prompt = data["prompts"]["my_chain"]["modes"][mode]["user"]

chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template(user_prompt)
])

model_name = os.getenv("OPENAI_API_MODEL", "gpt-5-nano")
chat_model = ChatOpenAI(
    model=model_name,
    temperature=0,                        
    timeout=15,
)


chain = chat_template | chat_model

result = chain.invoke({"input_text": "Hello, world!"})
print(result.content)
