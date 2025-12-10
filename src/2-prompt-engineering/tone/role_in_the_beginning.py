import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

load_dotenv()
model_name = "x-ai/grok-code-fast-1"
api_key = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)

# 🔑 Кастомная память с автоматическим добавлением системного промпта
class MemoryWithSystemPrepend(BaseChatMessageHistory):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self._messages = []  # Храним только диалог (без system)
    
    @property
    def messages(self):
        """При запросе истории добавляем system в начало"""
        return [SystemMessage(content=self.system_prompt)] + self._messages
    
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
    
print(f"\n💾 Реально хранится: {len(support_memory._messages)} сообщений")
print(f"📤 Отправляется в модель: {len(support_memory.messages)} сообщений (+ system)")
