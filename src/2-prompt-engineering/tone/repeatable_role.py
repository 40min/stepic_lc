import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage

load_dotenv()

model_name = "x-ai/grok-code-fast-1"
api_key = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0
)

# Задаём роли и стиль общения через системный промпт
support = "Ты опытный ассистент поддержки интернет провайдера. Ты всегда вежлив и доброжелателен. Отвечай не длиннее 100 символов."
client = "Ты раздражённый клиент у которого не работает интернет. Стиль общения грубый. Отвечай не длиннее 100 символов."

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("{system}"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

chain = prompt_template | llm

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)

# 🔑 КЛЮЧЕВАЯ ФУНКЦИЯ: напоминаем роль внутри истории
def chat_with_reminder(response, remind_every=6):
    response = response if isinstance(response, str) else response.content
    history = get_session_history('support')
    non_system = [m for m in history.messages if not isinstance(m, SystemMessage)]
    
    if len(non_system) > 0 and len(non_system) % remind_every == 0:
        history.add_message(SystemMessage(content=f"НАПОМИНАНИЕ: {support}"))
        print(f"⚠️ Добавлено напоминание роли (сообщение #{len(non_system)})")
    
    response = chain_with_history.invoke(
        {"system": support, "user_input": response},
        {"configurable": {"session_id": 'support'}})
    return response

# Демо: длинный диалог
response = "У меня за 3 года не было разрывов, а тут СРАЗУ ТРИ РАЗРЫВА!!!"
print('[1] Клиент:', response)

for i in range(12):
    response = chat_with_reminder(response)
    print(f'[{i+1}] Поддержка:', response.content, '\n')
   
    response = chain_with_history.invoke(
         {"system": client, "user_input": response},
         {"configurable": {"session_id": 'client'}}
    )
    print(f'[{i+2}] Клиент:', response.content)
    
# Посмотрим структуру истории
print("\n📊 Структура истории поддержки:")
for i, msg in enumerate(get_session_history('support').messages):
    msg_type = msg.__class__.__name__
    preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
    print(f"  {i+1}. {msg_type}: {preview}")
