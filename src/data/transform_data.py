import pandas as pd
import re
import os

def clean_and_chunk_text(raw_text, max_words=300):
    """
    Curata textul brut si il sparge in bucati semantice bazate pe propozitii,
    incadrandu-se intr-o limita maxima de cuvinte pentru embedding-uri optime.
    """
    if pd.isna(raw_text):
        return []

    # 1. Curatare de zgomot: Eliminam cuvintele exclusiv UPPERCASE (ex: PREFACE)
    # si liniile care incep cu CHAPTER, urmate de numere sau text.
    text = str(raw_text)
    text = re.sub(r'CHAPTER [A-ZIVX]+.*', '', text) # Sterge 'CHAPTER I: THE FREE SPIRIT'
    text = re.sub(r'\b[A-Z]{4,}\b', '', text)       # Sterge cuvintele all-caps lungi (titluri)
    text = re.sub(r'\[.*?\]', '', text)             # Sterge parantezele patrate

    # Normalizam spatiile
    text = re.sub(r'\s+', ' ', text)

    # 2. Spargem textul in propozitii dupa punct si spatiu
    sentences = text.split('. ')

    chunks = []
    current_chunk = []
    current_length = 0

    # 3. Adunam propozitiile in chunks
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words_in_sentence = len(sentence.split())

        # Daca adaugarea propozitiei depaseste limita si avem deja ceva in chunk, salvam
        if current_length + words_in_sentence > max_words and current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += words_in_sentence

    # Adaugam ce a mai ramas la final
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

def build_processed_dataset():
    # Update cu numele exact al fisierului din folderul tau raw
    input_path = os.path.join("artifacts", "raw", "Nietzsche_works_corpus.csv")
    output_path = os.path.join("artifacts", "processed", "indexed_books.csv")

    print(f"[*] Citesc datele brute din {input_path}...")
    df_raw = pd.read_csv(input_path)

    processed_rows = []
    chunk_id_counter = 1

    # Iteram prin fiecare carte din dataset-ul original
    for index, row in df_raw.iterrows():
        book_title = row.get('book_title', 'Unknown Title')
        raw_text = row.get('text', '')

        print(f"    - Procesez cartea: {book_title}")

        # Generam lista de chunks din textul brut
        chunks = clean_and_chunk_text(raw_text, max_words=300)

        # Cream un rand nou pentru fiecare chunk
        for chunk in chunks:
            # Setam metadatele programatic
            author = "Friedrich Nietzsche"
            school = "Existentialism / Nihilism"
            content_to_embed = f"Title: {book_title}. Author: {author}. Text: {chunk}"

            processed_rows.append({
                "id": chunk_id_counter,
                "title": book_title,
                "author": author,
                "school": school,
                "summary": chunk,
                "key_concepts": "existentialism, philosophy, will to power, morality",
                "content_to_embed": content_to_embed
            })
            chunk_id_counter += 1

    # Salvam noul dataset
    df_processed = pd.DataFrame(processed_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    print(f"[*] Gata! Am transformat cartile in {len(df_processed)} chunks.")
    print(f"[*] Salvat in {output_path}")

if __name__ == "__main__":
    build_processed_dataset()