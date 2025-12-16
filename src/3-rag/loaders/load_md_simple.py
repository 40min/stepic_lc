from langchain_community.document_loaders import TextLoader

loader = TextLoader("README.md", encoding="utf-8")
docs = loader.load()

print(f"Содержимое: {docs[0].page_content[:200]}")
print(f"Метаданные: {docs[0].metadata}")
