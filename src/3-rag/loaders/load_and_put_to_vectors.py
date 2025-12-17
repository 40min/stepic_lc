from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import bs4


# Загрузка HTML
html_loader = WebBaseLoader(
    web_paths=("https://docs.langchain.com/oss/python/langchain/overview",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(id="content")
    }
)
html_docs = html_loader.load()
print(f"Загружено {len(html_docs)} документов из HTML")

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

text_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=200, chunk_overlap=20)
splitted_docs = text_splitter.split_documents(html_docs)

print(f"Было документов: {len(html_docs)}, стало фрагментов: {len(splitted_docs)}")

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vector = embed_model.embed_query("Пример текста для эмбеддинга")
# print(len(vector), vector[:5])

vector_store = FAISS.from_documents(splitted_docs, embed_model)

query = "Когда основан МГУ?"
docs_found = vector_store.similarity_search(query, k=2)

for doc in docs_found:
    print(doc.metadata, doc.page_content[:50])

# сохранение
# vector_store.save_local("index/my_faiss_index")

# загрузка
# new_store = FAISS.load_local("index/my_faiss_index", embed_model)

# make a retreiver
# retriever = vector_store.as_retriever(search_kwargs={"k": 4})
