from langchain_community.document_loaders import WebBaseLoader

# Загрузка одной страницы
# url = "https://docs.langchain.com/oss/python/langchain/overview"
url = "https://hintaopas.fi/product.php?p=13508897"
loader = WebBaseLoader(url)
docs = loader.load()

# remove excessive newlines
page_clean = docs[0].page_content.replace("\n\n", " ")

print(f"Загружено документов: {len(docs)}")  # обычно 1
print(f"Источник: {docs[0].metadata['source']}")
print(f"Первые 2000 символов:\n{page_clean[:2000]}")
