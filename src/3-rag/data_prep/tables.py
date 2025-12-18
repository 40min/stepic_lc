import pandas as pd

def extract_table_as_dataframe(table_text: str) -> pd.DataFrame:
    """Преобразуем таблицу в DataFrame."""
    lines = table_text.strip().split('\n')
    rows = []
    for line in lines:
        if '|' in line:
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
        else:
            row = [cell.strip() for cell in line.split('\t') if cell.strip()]
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

def table_as_string(df: pd.DataFrame) -> str:
    """Создаём описательный текст для каждой строки."""
    if df.empty:
        return ""
    descriptions = []
    for _, row in df.iterrows():
        desc = ". ".join([f"{col}: {row[col]}" for col in df.columns])
        descriptions.append(desc)
    return "\n".join(descriptions)


table_example = """
| Продукт | Цена | Количество | Статус |
| Ноутбук | 50000 | 5 | В наличии |
| Монитор | 15000 | 3 | Ограничено |
| Клавиатура | 5000 | 10 | В наличии |
"""
df = extract_table_as_dataframe(table_example)

print("📊 Структурированные данные:")
print(df.to_dict(orient='records'))
print("\n📝 Описательный текст:")
print(table_as_string(df))
