from pathlib import Path
import re

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

import bs4

EMBED_MODEL = "cointegrated/rubert-tiny2"

def clean_text(text: str) -> str:
    """
    Gentle cleaning that preserves structure and context.
    Only removes obvious artifacts while keeping section headers and flow.
    """
    # Remove dates and times
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [APM]{2}', '', text)
    text = re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}', '', text)
    
    # Remove specific PDF header text (only this exact phrase)
    text = text.replace("Виды и сорта китайского чая: полный гид по классификации, вкусам и свойствам", "")
    
    # Remove navigation elements (but keep section titles)
    text = re.sub(r'\b(Наверх|Онлайн-запись|Онлайн-\s*запись)\b', '', text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove page numbers like "1/44" but not regular fractions
    text = re.sub(r'\b\d+/\d+\b(?=\s|$)', '', text)
    
    # Clean whitespace characters
    text = text.replace('\xa0', ' ')
    text = text.replace('\r', '\n')
    text = text.replace('\t', ' ')
    
    # Normalize newlines: max 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up lines but preserve paragraph structure
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

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
        doc.page_content = clean_text(doc.page_content)
        # Add source type to metadata
        doc.metadata['source_type'] = 'web'
        doc.metadata['topic'] = 'brewing_guide'
        
    print(f"Загружено {len(html_docs)} документов из HTML")
    return html_docs

def load_pdf():
    """Load PDF with metadata preservation"""
    print("Загружаем PDF...")
    loader = PyMuPDFLoader(
        file_path="data/tea_guide.pdf",
        extract_images=False    
    )

    docs = []
    for doc in loader.lazy_load():
        doc.page_content = clean_text(doc.page_content)
        # Add source type to metadata
        doc.metadata['source_type'] = 'pdf'
        doc.metadata['topic'] = 'tea_types'
        docs.append(doc)

    print(f"Загружено {len(docs)} страниц из PDF")
    if docs:
        print(f"Пример метаданных: {docs[0].metadata}")

    return docs

def create_db():
    """Create optimized vector database"""
    html_docs = load_web_page()
    pdf_docs = load_pdf()

    all_docs = html_docs + pdf_docs
    print(f"Всего документов: {len(all_docs)}")

    # OPTIMIZED: Larger chunks with overlap for better context
    # 800 chars ≈ 200 tokens, good balance for semantic search
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    splitted_docs = text_splitter.split_documents(all_docs)
    print(f"Было документов: {len(all_docs)}, стало фрагментов: {len(splitted_docs)}")
    
    # Show sample chunk for verification
    if splitted_docs:
        print(f"\nПример чанка (первые 200 символов):\n{splitted_docs[0].page_content[:200]}...")
        print(f"Метаданные чанка: {splitted_docs[0].metadata}")    
    
    print("\nИнициализация модели эмбеддингов...")
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}  # Improves similarity search
    )
    
    print("Создание векторной базы...")
    vector_store = FAISS.from_documents(splitted_docs, embed_model)
    
    # Save with metadata
    vector_store.save_local("data/tea_index")
    print("База создана и сохранена в data/tea_index")
    
    return vector_store

def load_db() -> FAISS:
    """Load existing vector database"""
    print("Загрузка векторной базы...")
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.load_local(
        "data/tea_index", 
        embed_model, 
        allow_dangerous_deserialization=True
    )
    print("База загружена")
    return vector_store

def db_lookup(vector_store: FAISS, query: str, k: int = 3, max_to_output: int = 500):
    """
    db lookup
    
    Args:
        k: Number of results
    """
    print(f"\n{'='*100}")
    print(f"Запрос: {query}")
    print(f"{'='*100}\n")
    
    # Get more results for better coverage
    docs_found = vector_store.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(docs_found, 1):
        print(f"Результат {i} (релевантность: {score:.4f})")
        print(f"Источник: {doc.metadata.get('source_type', 'unknown')} | "
              f"Тема: {doc.metadata.get('topic', 'unknown')}")
        if 'page' in doc.metadata:
            print(f"Страница: {doc.metadata['page']}")
        print(f"\nТекст ({len(doc.page_content)} символов):")
        print(f"{doc.page_content[:max_to_output]}")
        if len(doc.page_content) > max_to_output:
            print(f"... [обрезано, показано {max_to_output} из {len(doc.page_content)} символов]")
        print(f"\n{'-'*100}\n")

def test_queries(vector_store: FAISS):
    """Test with sample queries to verify quality"""
    test_cases = [
        "гайвань",
        "как заваривать белый чай",
        "температура воды для зеленого чая",
        "сколько грамм чая нужно"
    ]
    
    print("\n" + "="*100)
    print("ТЕСТИРОВАНИЕ КАЧЕСТВА ПОИСКА")
    print("="*100)
    
    for query in test_cases:
        db_lookup(vector_store, query, k=2, max_to_output=300)
        input("Нажмите Enter для следующего запроса...")

def main():
    if not Path("data/tea_index").exists():
        print("База отсутствует, создаём...")
        vector_store = create_db()
        
        # Run tests after creation
        print("\n" + "="*100)
        response = input("Хотите протестировать базу? (y/n): ").strip().lower()
        if response == 'y':
            test_queries(vector_store)
    else:
        print("База уже создана, загружаем её")
        vector_store = load_db()

    # Interactive mode
    print("\n" + "="*100)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ (Ctrl+C для выхода)")
    print("="*100)
    
    while True:
        try:
            user_text = input("\nВведите запрос: ").strip()            
        except (KeyboardInterrupt, EOFError):
            print("\n\nЗавершение работы.")
            break
        
        if not user_text:
            continue

        db_lookup(vector_store, user_text, k=3, max_to_output=400)

if __name__ == "__main__":
    main()
