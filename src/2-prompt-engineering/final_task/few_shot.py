import os
import yaml
from pyexpat import model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Load examples from prompts.yaml
with open('prompts.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
examples = data['prompts']['examples']

model_name = "x-ai/grok-code-fast-1"
api_key = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)

def answer_with_examples(input_text: str):
    example_prompt = PromptTemplate.from_template(
        "Как имплементировать: {input} \n Ответ: {output}"
    )    

    prompt_with_examples = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Кратко объясни как реализовать функцию в Python",
        suffix="Как имплементировать: {input} \n Ответ:",
        input_variables=["input"]
    )

    formatted_prompt = prompt_with_examples.format(input=input_text)
    # print(formatted_prompt)    
    response = llm.invoke(formatted_prompt)

    return response.content

def answer_with_simple_prompt(input_text: str):
    prompt = PromptTemplate.from_template(
        "Кратко объясни как реализовать функцию в Python {input}"
    )
    formatted_prompt = prompt.format(input=input_text)
    response = llm.invoke(formatted_prompt)
    return response.content


def __main__():

    delimeter = "=" * 100
    while True:
        
        try:
            input_text = input("Введите запрос: ")
        except (KeyboardInterrupt, EOFError):            
            break
        if not input_text:
            continue

        print("\n")
        print(delimeter)
        print("Ответ с примерами:\n")
        answer = answer_with_examples(input_text)
        print(answer)
        print(delimeter)
        print("\n")

        answer_unstable = answer_with_simple_prompt("генерация простых чисел до N")
        print(delimeter)
        print("Ответ без примеров:\n")
        answer_unstable = answer_with_simple_prompt(input_text)
        print(answer_unstable)
        print(delimeter)

        
    


if __name__ == "__main__":
    __main__()
