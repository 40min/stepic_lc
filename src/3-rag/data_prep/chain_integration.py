from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import TokenTextSplitter
import bs4

# 1. СОЗДАНИЕ RUNNABLE ЗАГРУЗЧИКОВ
class LoaderRunnable(RunnableLambda):
    def __init__(self, loader):
        super().__init__(lambda _: list(loader.lazy_load()))
        self.loader = loader

load_pdf = LoaderRunnable(PyMuPDFLoader("docs/document.pdf"))
load_html = LoaderRunnable(WebBaseLoader(
    web_paths=("https://docs.langchain.com/oss/python/langchain/overview",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(id="content")}))


# 2. СОЗДАНИЕ RUNNABLE ОБРАБОТЧИКОВ (Функции обработки тут - это просто заглушки)
def clean_pdf_text(text: str) -> str:
    return text

def clean_html_text(text: str) -> str:
    return text

def normalize_text(text: str) -> str:
    return text.lower()

def apply_func_to_all_docs(func):
    def process_docs(docs):
        for doc in docs:
            doc.page_content = func(doc.page_content)
        return docs
    return process_docs

clean_pdf = RunnableLambda(apply_func_to_all_docs(clean_pdf_text))
clean_html = RunnableLambda(apply_func_to_all_docs(clean_html_text))
normalize_all = RunnableLambda(apply_func_to_all_docs(normalize_text))


# 4. КОМПОЗИЦИЯ ЦЕПОЧКИ
chain = (
    RunnableParallel(pdf=load_pdf | clean_pdf,
                     html=load_html | clean_html)
    | RunnableLambda(lambda x: x["pdf"] + x["html"])
    | normalize_all
)

# 5.1 ПРОСТОЙ ВЫЗОВ
result = chain.invoke(None)

# # 5.2 АСИНХРОННЫЙ ВЫЗОВ
# import asyncio
# async def load_and_split_all_docs():
#     return await chain.ainvoke(None)
# result = asyncio.run(load_and_split_all_docs())

# 6. Смотрим результат
for doc in result:
        print(doc.page_content[:50])
        print('Источник:', doc.metadata['source'], '\n')
