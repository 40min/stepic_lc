from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(
    "README.md",
    mode="elements",  # разбивает по элементам структуры (заголовки, параграфы)
    encoding="utf-8"
)
docs = loader.load()

print(f"Загружено элементов: {len(docs)}")
for i, doc in enumerate(docs[:3]):
    print(f"\nЭлемент {i+1}:")
    print(f"Тип: {doc.metadata.get('category', 'N/A')}")
    print(f"Текст: {doc.page_content[:100]}...")
