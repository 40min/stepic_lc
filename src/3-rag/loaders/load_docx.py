from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("docs/report.docx")
docs = loader.load()

print(f"Содержимое: {docs[0].page_content[:200]}")
