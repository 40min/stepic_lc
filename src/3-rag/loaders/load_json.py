from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="docs\report.json",
    jq_schema=".",  # путь внутри JSON для извлечения данных
    text_content=False
)
docs = loader.load()

print(f"Загружено документов: {len(docs)}")
print(f"Первый документ: {docs[0].page_content}")
