import re
import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter




def load_data_from_url(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(id="bodyContent")
        }
    )

    docs = loader.load()

    # output some stats about loaded
    print(f"Загружено {len(docs)} документов")    

    return docs




def make_splitter(cfg):
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def clean_wikipedia_text(text: str) -> str:
    """
    Specialized cleaning for Wikipedia text content.
    Removes Wikipedia-specific artifacts while preserving the actual article content.
    """
    # Remove Wikipedia header and version info
    text = re.sub(r"Материал из Википедии — свободной энциклопедии\n", "", text)
    text = re.sub(r"Стабильная версия, проверенная \d{1,2} \w+ \d{4}\.\n", "", text)

    # Remove navigation elements
    text = re.sub(r"Перейти к навигации\n", "", text)
    text = re.sub(r"Перейти к поиску\n", "", text)
    text = re.sub(r"У этого термина существуют и другие значения, см\. [^.]+\.\n", "", text)

    # Remove template references in parentheses
    text = re.sub(r"\([^)]*значения\.\)", "", text)

    # Remove audio/media references
    text = re.sub(r"Прослушать введение в\s*статью\n", "", text)
    text = re.sub(r"Аудиозапись создана на основе версии статьи от \d{1,2} декабря \d{4} года\. Список аудиостатей\n", "", text)
    text = re.sub(r"noicon\n", "", text)
    text = re.sub(r"Медиафайлы на Викискладе\n", "", text)

    # Remove flag/emblem placeholders and other decorative elements
    text = re.sub(r"^Флаг\nГерб\n\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Девиз: «[^»]+»\n$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Гимн: «[^»]+»«[^»]+»\n$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[«»]\s*$", "", text, flags=re.MULTILINE)

    # Remove URL sources
    text = re.sub(r"Источник — https?://[^\n]+\n", "", text)
    text = re.sub(r"Источник — [^\n]+\n", "", text)

    # Remove category lines
    text = re.sub(r"^Категории:.*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Скрытые категории:.*\n", "", text, flags=re.MULTILINE)

    # Remove footnote references like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)

    # Remove template references and links (but keep meaningful content)
    text = re.sub(r"Википедия:[^\n]+\n", "", text)
    text = re.sub(r"Аудиостатьи \([^)]+\)\n", "", text)
    text = re.sub(r"Статьи со ссылками на [^\n]+\n", "", text)

    # Remove reference formatting and special characters
    text = re.sub(r"➤", "", text)
    text = re.sub(r"\xa0", " ", text)  # Non-breaking spaces
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\t", " ", text)

    # Remove Wikipedia section edit links
    text = re.sub(r"\[править \| править код\]", "", text)

    # Remove "Основная статья:" lines
    text = re.sub(r"Основная статья: [^\n]+\n", "", text)

    # Remove "См. также:" references
    text = re.sub(r"См\. также: [^\n]+\n", "", text)

    # Remove map/image references and captions
    text = re.sub(r"^Карта [^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Вулканы [^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Тоба [^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Этно-лингвистические [^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Пример звучания [^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Комодский варан [^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Тропические леса [^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Возвышенности [^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Индонезийские власти [^\n]*\n", "", text, flags=re.MULTILINE)

    # Remove section headers that are just references
    text = re.sub(r"^Список [^\n]*\[англ\.\]\n", "", text, flags=re.MULTILINE)

    # Remove lines starting with specific patterns
    text = re.sub(r"^Герб Индонезии[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Здание в Джакарте[^\n]*\n", "", text, flags=re.MULTILINE)

    # Clean up lists and sections (but preserve content)
    text = re.sub(r"^\s*[•·]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*—\s*", "", text, flags=re.MULTILINE)

    # Remove empty lines at the beginning and normalize newlines
    text = re.sub(r"^\n+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Clean up lines but preserve paragraph structure
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()