from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Загрузка всех .md файлов из директории
loader = DirectoryLoader(
    "docs/",
    glob="**/*.md",  # паттерн для поиска файлов
    loader_cls=TextLoader,  # класс загрузчика для каждого файла
    loader_kwargs={"encoding": "utf-8"},  # параметры для загрузчика
    show_progress=True,  # показывать прогресс
    use_multithreading=True,  # использовать многопоточность
)

docs = loader.load()
print(f"Загружено файлов: {len(docs)}")
