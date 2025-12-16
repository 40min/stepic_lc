from langchain_community.document_loaders import PyPDFLoader

# Инициализация загрузчика с параметрами
loader = PyPDFLoader(
    file_path="docs/document.pdf",
    # mode="page",  # "page" - по страницам (по умолчанию), "single" - весь документ
    # extract_images=False,  # извлекать изображения
)

# Загрузка документов
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

# Или просто:
docs = loader.load()

# Просмотр результата
print(f"Всего документов: {len(docs)}")
print(f"Первая страница:\n{docs[0].page_content[:200]}...")
print(f"Метаданные: {docs[0].metadata}")
