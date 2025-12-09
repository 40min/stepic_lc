import time
import os
from dotenv import load_dotenv


load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from openai import (
    APITimeoutError, 
    APIConnectionError, 
    AuthenticationError
)

import logging

logging.basicConfig(
    filename="chat_session.log", 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)


# Создаём класс для CLI-бота
class CliBot():
    def __init__(self, model_name, system_prompt="Ты полезный ассистент."):
        api_key = os.getenv("OPENROUTER_API_KEY", "")

        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=15,
        )

        # Создаём Хранилище истории
        self.store = {} 

        # Создаем шаблон промпта
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Создаём цепочку (тут используется синтаксис LCEL*)
        self.chain = self.prompt | self.chat_model

        # Создаём цепочку с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain, # Цепочка с историей
            self.get_session_history, # метод для получения истории
            input_messages_key="question", # ключ для вопроса
            history_messages_key="history", # ключ для истории
        )

    # Метод для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def __call__(self, session_id):
        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            logging.info(f"User: {user_text}")

            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                break
            if msg == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                print("Бот: Контекст диалога очищен.")
                continue
            
            try:
                print("Sending request to API...")
                start_time = time.time()
                response = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": session_id}}
                )
                logging.info(f"Bot: {response.content}")
                end_time = time.time()

                bot_reply = response.content.strip()
                print(f"Response time: {end_time - start_time:.2f} seconds")
                print('Бот:', bot_reply, "\n")
            except APITimeoutError as e:
                print("Бот: [Ошибка] Превышено время ожидания ответа.")
                continue
            except APIConnectionError as e:
                print("Бот: [Ошибка] Не удалось подключиться к сервису LLM.")
                continue
            except AuthenticationError as e:
                print("Бот: [Ошибка] Проблема с API‑ключом (неавторизовано).")
                break
            except Exception as e:
                print(f"Бот: [Неизвестная ошибка] {e}")
                continue


if __name__ == "__main__":
    system_prompt = '''Ты полезный ассистент. Еще ты лихой пират в прошлом. Отвечай подробно и по существу с щепоткой соленого морского юмора'''

    bot = CliBot(
        model_name = os.getenv("OPENROUTER_API_MODEL", "no-model"),
    )

    logging.info("=== New session ===")
    bot("user_123")