from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
import bs4


def load_web_page():
    """Load web page with metadata preservation"""
    print("Загружаем страницу...")
    html_loader = WebBaseLoader(
        web_paths=("https://tea-mail.by/stati-o-nas/kak-pravilno-zavarivat-kitayskiy-chay/",),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(class_="post-info")
        }
    )
    html_docs = html_loader.load()
    
    for doc in html_docs:        
        # Add source type to metadata
        doc.metadata['source_type'] = 'web'
        doc.metadata['topic'] = 'brewing_guide'
        
    print(f"Загружено {len(html_docs)} документов из HTML")
    return html_docs

def load_pdf(file_path: str, topic: str, source_type: str = "pdf"):
    """Load PDF with metadata preservation"""
    print("Загружаем PDF...")
    loader = PyMuPDFLoader(
        file_path=file_path,
        extract_images=False    
    )

    docs = []
    for doc in loader.lazy_load():        
        # Add source type to metadata
        doc.metadata['source_type'] = source_type
        doc.metadata['topic'] = topic
        docs.append(doc)

    print(f"Загружено {len(docs)} страниц из PDF")
    if docs:
        print(f"Пример метаданных: {docs[0].metadata}")

    return docs
