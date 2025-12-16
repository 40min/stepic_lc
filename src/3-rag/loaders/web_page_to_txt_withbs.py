from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_paths=("https://docs.langchain.com/oss/python/langchain/overview",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(id="content")
    }
)

docs = loader.load()
print(docs[0].page_content[:2000])