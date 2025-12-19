from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_core.runnables import RunnableLambda
import bs4


class LoaderRunnable(RunnableLambda):
    """Wrapper to make loaders compatible with RunnableParallel"""
    def __init__(self, loader, topic: str, source_type: str = "pdf"):
        def load_and_tag(_):
            docs = list(loader.lazy_load())
            for doc in docs:
                doc.metadata['source_type'] = source_type
                doc.metadata['topic'] = topic
            print(f"Загружено {len(docs)} документов ({topic})")
            print(f"Пример метаданных: {docs[0].metadata}")
            return docs
        super().__init__(load_and_tag)


# Create loader runnables for each data source
load_html = LoaderRunnable(
    WebBaseLoader(
        web_paths=("https://tea-mail.by/stati-o-nas/kak-pravilno-zavarivat-kitayskiy-chay/",),
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_="post-info")}
    ),
    topic="brewing_guide",
    source_type="web"
)

load_pdf_types = LoaderRunnable(
    PyMuPDFLoader(file_path="data/tea_guide.pdf", extract_images=False),
    topic="tea_types"
)

load_pdf_common = LoaderRunnable(
    PyMuPDFLoader(file_path="data/all_you_need_to_know.pdf", extract_images=False),
    topic="common_information"
)

load_pdf_ushan = LoaderRunnable(
    PyMuPDFLoader(file_path="data/locations_ushan.pdf", extract_images=False),
    topic="locations_ushan"
)
