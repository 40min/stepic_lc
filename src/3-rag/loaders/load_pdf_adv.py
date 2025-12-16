from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader(
    file_path="docs/document.pdf",
    # mode="page",  # "page" или "single"
    # extract_images=False,  # извлечение изображений
    # extract_tables="markdown",  # извлечение таблиц в формате markdown
)

docs = []
for doc in loader.lazy_load():
    docs.append(doc)

print(f"Загружено {len(docs)} страниц из PDF")
print(f"Метаданные: {docs[0].metadata}")
