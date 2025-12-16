from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

import bs4

def load_web_page():
    # Загрузка HTML
    print("Загружаем страницу ...")
    html_loader = WebBaseLoader(
        web_paths=("https://tea-mail.by/stati-o-nas/kak-pravilno-zavarivat-kitayskiy-chay/",),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(class_="post-info")
        }
)
    html_docs = html_loader.load()
    print(f"Загружено {len(html_docs)} документов из HTML")
    return html_docs

def load_pdf():
    print("Загружаем PDF ...")
    # Загрузка PDF
    loader = PyMuPDFLoader(
        file_path="data/tea_guide.pdf",
        # mode="page",  # "page" или "single"
        extract_images=False    
    )

    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)

    print(f"Загружено {len(docs)} страниц из PDF")
    print(f"Метаданные: {docs[0].metadata}")

    return docs


def create_db():
    html_docs = load_web_page()
    pdf_docs = load_pdf()

    all_docs = html_docs + pdf_docs
    print(f"Всего документов: {len(all_docs)}")

    text_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=200, chunk_overlap=20)
    splitted_docs = text_splitter.split_documents(html_docs)

    print(f"Было документов: {len(all_docs)}, стало фрагментов: {len(splitted_docs)}")

    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splitted_docs, embed_model)
    
    vector_store.save_local("data/tea_index")

    print("База создана и сохранена")
    return vector_store

def load_db() -> FAISS:
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("data/tea_index", embed_model)
    return vector_store

def db_lookup(vector_store: FAISS, query: str, max_to_output: int = 100):
    docs_found = vector_store.similarity_search(query, k=2)
    for doc in docs_found:
        print(doc.metadata, doc.page_content[:max_to_output])

def main():
    if not Path("data/tea_index").exists():
        print("База отсутствует, создаём ...")
        vector_store = create_db()
    else:
        print("База уже создана, загружаем её")
        vector_store = load_db()

    while True:
        try:
            user_text = input("Введите запрос: ").strip()            
        except (KeyboardInterrupt, EOFError):
            print("\nБот: Завершение работы.")
            break
        if not user_text:
            continue

        db_lookup(vector_store, user_text)                

if __name__ == "__main__":
    main()
