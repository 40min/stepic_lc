import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL", "gpt-5-mini")

llm = ChatOpenAI(model=MODEL, temperature=0)

example_prompt = PromptTemplate.from_template(
    "Ввод: {input}\nВывод: {output}"
)

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"}
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Дайте антоним каждого слова:",
    suffix="Ввод: {input}\nВывод:",
    input_variables=["input"]
)

formatted_prompt = prompt.format(input="young")
# print(formatted_prompt)
response = llm.invoke(formatted_prompt)

print(response.content) # Вывод: old
