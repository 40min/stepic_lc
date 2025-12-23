from langchain_core.documents import Document

# 1. Готовые документы
documents = [Document(page_content="Фирма ООО 'Одуванчик' основана в 2015 году", metadata={'source': 'doc_1.pdf'}),
             Document(page_content="Директор ООО 'Одуванчик' Смирнов Иван Петрович", metadata={'source': "doc_2.pdf"}),
             Document(page_content="Адрес ООО 'Одуванчик' ул.Ленина д.5", metadata={'source': "doc_3.pdf"}),
             Document(page_content="Иван Петрович отличный руководитель", metadata={'source': 'doc_4.pdf'}),
             Document(page_content="В 2015 году мы основали нашу компанию", metadata={'source': "doc_5.pdf"}),
             Document(page_content="Наша компания продаёт лучший продукт на рынке", metadata={'source': "doc_6.pdf"}),
             Document(page_content="ул.Ленина д.5 это наш адрес с самого основания", metadata={'source': 'doc_7.pdf'})
             ]

# 2. ФОРМИРОВАНИЕ ground_truth_docs
ground_truth_docs = {
    "Когда основана компания?": ["doc_1.pdf", "doc_5.pdf"],
    "Кто директор компании?": ["doc_2.pdf"],
    "Какой адрес у компании?": ["doc_3.pdf", "doc_7.pdf"]
}

# Загружаем retriever
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embed_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Оценка качества поиска
k = 2

# Precision@k
precision_total = 0
for question, true_docs in ground_truth_docs.items():
    found_docs = retriever.invoke(question)
    found_ids = [d.metadata["source"] for d in found_docs[:k]]
    
    # Считаем, сколько релевантных нашли
    relevant_found = sum(1 for doc_id in true_docs if doc_id in found_ids)
    precision = relevant_found / k
    precision_total += precision
    
    print(f"Вопрос: {question}")
    print(f"  Эталонные документы: {true_docs}")
    print(f"  Найденные документы: {found_ids}")
    print(f"  Precision@{k}: {precision:.2f}\n")

precision_avg = precision_total / len(ground_truth_docs)
print(f"Средняя Precision@{k}: {precision_avg:.2f}")


# Recall@k
recall_total = 0

for question, true_docs in ground_truth_docs.items():

    found_docs = retriever.invoke(question)
    found_ids = [d.metadata["source"] for d in found_docs[:k]]

    relevant_found = sum(1 for doc_id in true_docs if doc_id in found_ids)
    recall = relevant_found / len(true_docs) if true_docs else 0

    recall_total += recall

    print(f"\nВопрос: {question}")
    print(f"  Recall@{k}: {recall:.2f} ({relevant_found}/{len(true_docs)})\n")

recall_avg = recall_total / len(ground_truth_docs)
print(f"Средняя Recall@{k}: {recall_avg:.2f}")
