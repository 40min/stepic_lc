import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class WordInfo(BaseModel):
    word: str = Field(description="Слово на русском языке")
    translation: str = Field(description="Перевод слова на английский язык")
    example: str = Field(description="Пример использования слова в предложении")


def _get_llm():
    MODEL = os.getenv("OPENAI_API_MODEL", "gpt-5-mini")
    llm = ChatOpenAI(model=MODEL)
    return llm


def _get_chain():
    llm = _get_llm()

    output_parser = PydanticOutputParser(pydantic_object=WordInfo)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=(
        "Переведи слово на английский язык и придумай пример предложения.\n"
        "Отвечай в формате JSON.\n"
        "Формат ответа:\n{format_instructions}\n"
        "Слово: {user_word}\n"
    ),
        input_variables=["user_word"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser 

    return chain

def _main():
    chain = _get_chain()
    while True:
        try:
            user_text = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nБот: Завершение работы.")
            break
        if not user_text:
            continue

        try:
            result = chain.invoke({"user_word": user_text})
            print(f"Бот: {result.model_dump_json()}")
        except Exception as e:
            print(f"Бот: Произошла ошибка при обработке запроса: {e}")
            print("Пожалуйста, попробуйте еще раз.")


if __name__ == "__main__":
    _main()
