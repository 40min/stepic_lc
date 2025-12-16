from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="docs/train.csv",
    encoding="utf-8",
    csv_args={
        "delimiter": ",",
    }
)
docs = loader.load()

# Каждая строка CSV становится отдельным Document
print(f"Загружено {len(docs)} строк")
print(f"Первая строка: {docs[0].page_content}")
print(f"Метаданные: {docs[0].metadata}")
