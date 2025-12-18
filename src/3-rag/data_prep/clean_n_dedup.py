import re
import hashlib
from typing import List
from langchain_core.documents import Document

def is_table_row(text: str) -> bool:
    return bool(re.search(r'\|.*\|', text)) or \
           bool(re.search(r'\t{2,}', text)) or \
           bool(re.search(r'^\s*[\w\s]+\s{2,}[\w\s]+', text))

def filter_and_dedup(docs: List[Document], min_length: int = 30) -> List[Document]:
    unique_hashes = set()
    filtered = []
    stats = {'duplicates': 0, 'too_short': 0, 'empty': 0}
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            stats['empty'] += 1
            continue
        if len(text) < min_length and not is_table_row(text):
            stats['too_short'] += 1
            continue
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        if h in unique_hashes:
            stats['duplicates'] += 1
            continue
        unique_hashes.add(h)
        filtered.append(doc)
    print(f"Первоначально: {len(docs)} чанков")
    print(f"Удалено дубликатов: {stats['duplicates']}, слишком коротких: {stats['too_short']}, пустых: {stats['empty']}")
    print(f"Осталось: {len(filtered)} чанков")
    return filtered



def is_meaningful(text: str, threshold: float = 0.5) -> bool:
    if not text:
        return False
    ratio = sum(ch.isalpha() for ch in text) / len(text)
    return ratio > threshold  # по умолчанию — хотя бы половина букв

