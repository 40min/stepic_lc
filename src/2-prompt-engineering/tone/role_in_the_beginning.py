import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
import tiktoken

load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL", "gpt-5")

llm = ChatOpenAI(model_name=MODEL, temperature=0.9)

# Кастомная память с автоматическим добавлением системного промпта
class MemoryWithSystemPrepend(BaseChatMessageHistory):
    def __init__(self, system_prompt: str, max_tokens: int = 4000):
        self.system_prompt = system_prompt
        self._messages = []  # Храним только диалог (без system)
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages):
            total = 0
            for msg in messages:
                total += len(self.encoder.encode(msg.content))
            return total
    
    @property
    def messages(self):
        """Подготовка истории: prepend system + усечение до max_tokens"""
        history = [SystemMessage(content=self.system_prompt)] + self._messages
        
        while self.count_tokens(history) > self.max_tokens and len(history) > 1:
            history.pop(1)

        return history

    def add_message(self, message: BaseMessage):
        if not isinstance(message, SystemMessage):
            self._messages.append(message) # добавляем сообщения только в диалог

    def clear(self):
        self._messages = []


# Использование
support_memory = MemoryWithSystemPrepend("Ты опытный ассистент поддержки интернет-провайдера. Ты всегда вежлив и доброжелателен. Отвечай не длиннее 100 символов.")
client_memory = MemoryWithSystemPrepend("Ты раздражённый клиент у которого не работает интернет. Стиль общения грубый. Отвечай не длиннее 100 символов.")

# Простой чат без RunnableWithMessageHistory для наглядности
def simple_chat(user_input: str, memory: MemoryWithSystemPrepend):
    memory.add_message(HumanMessage(content=user_input))
    response = llm.invoke(memory.messages)
    memory.add_message(response)
    
    return response.content

# Демо: длинный диалог
response = "У меня за 3 года не было разрывов, а тут СРАЗУ ТРИ РАЗРЫВА!!!"
print('[1] Клиент:', response)

for i in range(12):
    response = simple_chat(response, support_memory)
    print(f"[{i+1}] Поддержка: {response}\n")
    response = simple_chat(response, client_memory)
    print(f"[{i+2}] Клиент: {response}")

print("\n📊 Что видит модель (первые 6 сообщений в памяти поддержки):")
for msg in support_memory.messages[:6]:
    print(f"  - {msg.__class__.__name__}: {msg.content[:50]}...")
    
print(f"\n💾 Реально хранится сообщений: {len(support_memory._messages)}")
print(f"📤 Отправляется в модель сообщений (+ system): {len(support_memory.messages)}")
