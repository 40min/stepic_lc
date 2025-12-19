import re

import hashlib
import numpy as np

from langchain_core.documents import Document
from langchain_core.vectorstores.in_memory import InMemoryVectorStore


def clean_text(text: str) -> str:
    """
    Gentle cleaning that preserves structure and context.
    Only removes obvious artifacts while keeping section headers and flow.
    """
    # Remove dates and times
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [APM]{2}", "", text)
    text = re.sub(r"(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}", "", text)
    
    # Remove specific PDF header text (only this exact phrase)
    text = text.replace("Виды и сорта китайского чая: полный гид по классификации, вкусам и свойствам", "")
    
    # Remove navigation elements (but keep section titles)
    text = re.sub(r"\b(Наверх|Онлайн-запись|Онлайн-\s*запись)\b", "", text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    
    # Remove page numbers like "1/44" but not regular fractions
    text = re.sub(r"\b\d+/\d+\b(?=\s|$)", "", text)
    
    # Clean whitespace characters
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    
    # Normalize newlines: max 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Clean up lines but preserve paragraph structure
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)
    
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    
    return text.strip()


def filter_and_dedup(docs: list[Document], min_length: int = 30) -> list[Document]:
    unique_hashes = set()
    filtered = []
    stats = {"duplicates": 0, "too_short": 0, "empty": 0, "not_meaningful": 0}
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            stats["empty"] += 1
            continue
        if len(text) < min_length:
            stats["too_short"] += 1
            continue
        if not is_meaningful(text):
            stats["not_meaningful"] += 1
            continue
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in unique_hashes:
            print(f"Пропускаем дубликат: '{text[:50]}...'")
            stats["duplicates"] += 1
            continue
        unique_hashes.add(h)
        filtered.append(doc)
    print(f"[filter_and_dedup] Первоначально: {len(docs)} чанков")
    print(f"[filter_and_dedup] Удалено дубликатов: {stats["duplicates"]}, слишком коротких: {stats["too_short"]}, пустых: {stats["empty"]}, не содержащих смысла: {stats["not_meaningful"]}")
    print(f"[filter_and_dedup]Осталось: {len(filtered)} чанков")
    return filtered

def is_meaningful(text: str, threshold: float = 0.5) -> bool:
    if not text:
        return False
    ratio = sum(ch.isalpha() for ch in text) / len(text)
    return ratio > threshold  # по умолчанию — хотя бы половина букв


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def dedupe_by_embedding(docs: list[Document], embedding_model, threshold: float = 0.95) -> list[Document]:
    # создаём векторное хранилище
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    kept_docs = []
    embeddings = []

    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        emb = embedding_model.embed_documents([text])[0]

        if not embeddings:
            vector_store.add_documents([doc])
            embeddings.append(emb)
            kept_docs.append(doc)
            continue

        # делаем similarity search вручную: можно просто сравнить со всеми
        sims = [cosine_similarity(np.array(emb), np.array(e)) for e in embeddings]
        max_sim = max(sims)
        if max_sim < threshold:
            vector_store.add_documents([doc])
            embeddings.append(emb)
            kept_docs.append(doc)
        else:
            print(f"Пропускаем дубликат (по embedding): '{text[:50]}...' с sim = {max_sim}")

            
    print(f"[dedupe_by_embedding]Всего документов: {len(kept_docs)}, пропущено дубликатов: {len(docs) - len(kept_docs)}")

    return kept_docs