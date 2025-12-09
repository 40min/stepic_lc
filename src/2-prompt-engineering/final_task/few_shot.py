import os
from pyexpat import model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

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
        "Ввод: {input}\nВывод: {output}"
    )

    examples = [
        {"input": "Выравнивание вложенного списка", "output": "развернуть рекурсивно, проверяя тип каждого элемента\n ```python \n flatten = lambda lst: [item for sublist in lst for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]```"},
        {"input": "декоратор", "output": "создать функцию, которая принимает функцию в качестве аргумента и возвращает новую функцию, которая оборачивает исходную функцию\n ```python \n  def decorator(func):\n    def wrapper(*args, **kwargs):\n        print(\"Before function call\")\n        result = func(*args, **kwargs)\n        print(\"After function call\")\n        return result\n    return wrapper```"}  
    ]

    prompt_with_examples = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Кратко объясни как реализовать функцию в Python:",
        suffix="Ввод: {input}\nВывод:",
        input_variables=["input"]
    )

    formatted_prompt = prompt_with_examples.format(input=input_text)
    # print(formatted_prompt)
    response = llm.invoke(formatted_prompt)

    return response.content

def answer_with_simple_prompt(input_text: str):
    prompt = PromptTemplate.from_template(
        "Кратко объясни как реализовать функцию в Python: {input}"
    )
    formatted_prompt = prompt.format(input=input_text)
    response = llm.invoke(formatted_prompt)
    return response.content


def __main__():
    while True:
        
        try:
            input_text = input("Введите запрос: ")
        except (KeyboardInterrupt, EOFError):            
            break
        if not input_text:
            continue

        # answer_unstable = answer_with_simple_prompt("генерация простых чисел до N")
        print("=" * 10)
        print("Ответ без примеров:")
        answer_unstable = answer_with_simple_prompt(input_text)
        print(answer_unstable)
        print("=" * 10)

        print("Ответ с примерами:")
        answer = answer_with_examples(input_text)
        print(answer)
        print("=" * 10)
    


if __name__ == "__main__":
    __main__()
