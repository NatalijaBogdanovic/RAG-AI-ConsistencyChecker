import os
import re
import numpy as np
import openai
import faiss
import shutil
import pickle
import nltk  # --- IZMENA: Importujemo NLTK ---
from sklearn.metrics.pairwise import cosine_similarity
from embedding import embed_query_cohere, embed_doc_cohere
from utils import OPENROUTER_API_KEY

# --- IZMENA: Tiho preuzimanje 'punkt' modela za NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab') # Dodata provera i za punkt_tab
    print("✅ NLTK 'punkt' i 'punkt_tab' resursi su već preuzeti.")
except LookupError:
    print("⚠️ NLTK resursi nisu kompletni. Preuzimam 'punkt' i 'punkt_tab'...")
    nltk.download('punkt')
    nltk.download('punkt_tab') # Eksplicitno preuzimanje i za punkt_tab
    print("✅ Preuzimanje NLTK resursa završeno.")

# FAISS fajlovi na disku
FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_METADATA_FILE = "faiss_metadata.pkl"
openai.api_key = OPENROUTER_API_KEY

# Globalne varijable za smeštanje embeddinga i čankova
embeddings = np.array([])
sentences = np.array([])
all_metadatas = []  # --- IZMENA: Lista za čuvanje metapodataka (npr. ime fajla) ---
all_embeddings = []
all_ids = []
index = None


# --- IZMENA: Uklonjene duplirane 'chunk_text' funkcije ---
# --- Sada imamo jednu funkciju za učitavanje DOKUMENATA (konteksta) ---
def load_and_chunk_docs(folder, sentences_per_chunk=3, overlap=1):
    """
    Učitava dokumente, deli ih na rečenice pomoću NLTK,
    grupiše ih u čankove i čuva metapodatke (ime fajla).
    """
    print("CHUNKING DOKUMENATA (KONTEKSTA):")
    dataset_with_metadata = []

    for file_name in os.listdir(folder):
        if not (file_name.endswith('.txt') or file_name.endswith('.csv')):
            continue

        path = os.path.join(folder, file_name)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. Pouzdano deljenje na rečenice pomoću NLTK
        all_sentences = nltk.sent_tokenize(text, language='english')  # 'punkt' je dobar i za srpski, iako piše english

        # 2. Čišćenje rečenica
        cleaned_sentences = [s.replace('\n', ' ').strip() for s in all_sentences if s.strip()]

        # 3. Grupisanje rečenica u čankove
        i = 0
        while i < len(cleaned_sentences):
            end = i + sentences_per_chunk
            chunk_text = " ".join(cleaned_sentences[i:end])

            # Čuvamo metapodatke
            metadata = {"source_file": file_name, "start_sentence_index": i}

            dataset_with_metadata.append((chunk_text, metadata))

            # Pomeramo prozor
            step = sentences_per_chunk - overlap
            i += max(1, step)  # Osiguravamo da se uvek pomerimo bar za 1

    print(f"CHUNKING DOKUMENATA DONE. Ukupno čankova: {len(dataset_with_metadata)}")
    # Vraća listu torki: [(text1, meta1), (text2, meta2), ...]
    return dataset_with_metadata


# --- IZMENA: Jasna funkcija za učitavanje UPITA ---
def load_and_chunk_query(folder):
    """
    Učitava fajlove iz 'query' foldera i deli ih na
    pojedinačne rečenice (svaka rečenica je jedan upit).
    """
    print("CHUNKING UPITA:")
    query_list = []
    for file in os.listdir(folder):
        if not (file.endswith('.txt') or file.endswith('.csv')):
            continue

        path = os.path.join(folder, file)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. Pouzdano deljenje na rečenice
        all_sentences = nltk.sent_tokenize(text, language='english')

        # 2. Čišćenje i dodavanje svake rečenice kao posebnog upita
        cleaned_sentences = [s.replace('\n', ' ').strip() for s in all_sentences if s.strip()]
        query_list.extend(cleaned_sentences)

    print(f"CHUNKING UPITA DONE. Ukupno upita: {len(query_list)}")
    return query_list  # Vraća običnu listu stringova


def load_faiss_database():
    """Funkcija za učitavanje FAISS baze ako postoji"""
    global all_embeddings, all_ids, sentences, all_metadatas, index
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE):
        try:
            print("📌 FAISS baza pronađena! Učitavam podatke...")
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open(FAISS_METADATA_FILE, "rb") as f:
                metadata = pickle.load(f)

            all_embeddings = metadata["embeddings"]
            all_ids = metadata["ids"]
            sentences = np.array(metadata["texts"])
            # --- IZMENA: Učitavamo i metapodatke ---
            # Fallback (get) za slučaj da pokrećete na starom .pkl fajlu
            all_metadatas = metadata.get("metadatas", [{} for _ in all_ids])

            print("📌 FAISS baza učitana sa diska!")
            return index
        except Exception as e:
            print(f"Greška pri učitavanju FAISS baze: {e}. Kreiram novu.")
            return None
    else:
        print("⚠️ FAISS baza ne postoji. Kreiram novu...")
        return None


def create_database():
    global sentences, all_embeddings, all_ids, all_metadatas, index
    datasource_dir = os.path.join(os.getcwd(), 'datasource')

    # --- IZMENA: Funkcija sada vraća torke (text, metadata) ---
    dataset_tuples = load_and_chunk_docs(datasource_dir)

    # Ako nema novih fajlova u 'datasource' folderu
    if not dataset_tuples:
        index = load_faiss_database()  # Pokušaj učitati staru bazu
        if index is None:
            print("⚠️ Nema FAISS baze i nema novih dokumenata u 'datasource' folderu. Baza je prazna.")
            d = 1024  # Standardna dimenzija za Cohere
            index = faiss.IndexFlatIP(d)
        else:
            print("📌 Baza je već ažurna. Nema novih fajlova za indeksiranje.")
        return  # Završi funkciju ako nema šta da se doda

    # --- IZMENA: Razdvajamo tekstove od metapodataka ---
    new_texts = [item[0] for item in dataset_tuples]
    new_metas = [item[1] for item in dataset_tuples]
    new_sentences_np = np.array(new_texts)  # Ovo su sada samo tekstualni čankovi

    # Proveravamo da li FAISS baza već postoji
    index = load_faiss_database()

    print(f"Indeksiranje {len(new_sentences_np)} novih segmenata...")

    # Kreiramo embeddinge samo za nove podatke
    new_embeddings_list = embed_doc_cohere(new_texts)  # Šaljemo listu stringova
    new_embeddings = np.array(new_embeddings_list).astype('float32')

    # Normalizacija vektora
    faiss.normalize_L2(new_embeddings)

    if index is None:
        # Kreiramo novi FAISS indeks
        d = new_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(new_embeddings)

        all_embeddings = new_embeddings.tolist()
        all_ids = list(range(len(new_texts)))
        sentences = new_sentences_np
        all_metadatas = new_metas
    else:
        # Dopunjujemo postojeći FAISS indeks
        index.add(new_embeddings)

        all_embeddings.extend(new_embeddings.tolist())
        start_id = len(all_ids)
        all_ids.extend(list(range(start_id, start_id + len(new_texts))))
        sentences = np.append(sentences, new_sentences_np)
        all_metadatas.extend(new_metas)

    # Čuvamo FAISS bazu na disk
    faiss.write_index(index, FAISS_INDEX_FILE)

    # --- IZMENA: Čuvamo i metapodatke ---
    metadata_to_save = {
        "embeddings": all_embeddings,
        "ids": all_ids,
        "texts": sentences.tolist(),
        "metadatas": all_metadatas  # Čuvamo i listu metapodataka
    }
    with open(FAISS_METADATA_FILE, "wb") as f:
        pickle.dump(metadata_to_save, f)
    print("📌 FAISS baza sačuvana/ažurirana na disku!")

    # Pomeramo obrađene fajlove
    datasource_added_dir = os.path.join(os.getcwd(), 'datasource_added')
    if not os.path.exists(datasource_added_dir):
        os.makedirs(datasource_added_dir)

    # --- IZMENA: Pomeramo samo fajlove koji su zaista bili obrađeni ---
    processed_files = set(meta['source_file'] for meta in new_metas)
    for filename in processed_files:
        source_file = os.path.join(datasource_dir, filename)
        destination_file = os.path.join(datasource_added_dir, filename)
        if os.path.exists(source_file):  # Provera da li fajl još uvek postoji
            shutil.move(source_file, destination_file)


def database_embeddings():
    return all_embeddings


# --- IZMENA: Funkcija koja vraća tekst ČANKA i njegove METAPODATKE ---
def database_chunks_with_metadata(indexes):
    """Vraća listu torki (text, metadata) za date indekse."""
    global sentences, all_metadatas
    results = []
    for idx in indexes:
        if 0 <= idx < len(sentences):
            text = sentences[idx]
            meta = all_metadatas[idx] if 0 <= idx < len(all_metadatas) else {}
            results.append((text, meta))
    return results


def load_query():
    return load_and_chunk_query(os.path.join(os.getcwd(), 'query'))


def embed_query(query):
    embeddings_list = embed_query_cohere(query)
    query_embedding = np.array(embeddings_list).astype('float32')
    faiss.normalize_L2(query_embedding)
    return query_embedding


def find_similarities(query_embedded, top_n):
    if query_embedded.shape[1] != index.d:
        raise ValueError(
            f"Dimensionality mismatch: query_embedded has shape {query_embedded.shape[1]}, expected {index.d}")
    distances, indices = index.search(query_embedded, top_n)
    return distances, indices


def find_most_relevant_text(query, top_n=3, similarity_threshold=0.5):
    """
    Pronalazi najrelevantnije čankove i formatira ih
    tako da uključuju metapodatke (ime fajla).
    """
    global index
    if index is None or index.ntotal == 0:
        print("Greška: FAISS index nije inicijalizovan ili je prazan.")
        return "Nema relevantnih podataka (baza je prazna)."

    query_embedding = embed_query([query])
    distances, indices = find_similarities(query_embedding, top_n)

    relevant_chunks = []

    if len(indices.shape) == 2:
        indices_list = indices[0]
        similarities_list = distances[0]

        for i in range(len(indices_list)):
            idx = indices_list[i]
            sim = similarities_list[i]

            if sim > similarity_threshold:
                # --- IZMENA: Preuzimamo čank i metapodatke ---
                retrieved_chunks = database_chunks_with_metadata([idx])
                if retrieved_chunks:
                    text, meta = retrieved_chunks[0]
                    # --- IZMENA: Formatiramo izlaz da uključi izvor ---
                    source = meta.get('source_file', 'Nepoznat izvor')
                    formatted_chunk = f"[Izvor: {source}]\n{text}"
                    relevant_chunks.append(formatted_chunk)
            else:
                break

    if not relevant_chunks:
        return "Nema relevantnih podataka."

    # Spaja sve pronađene relevantne čankove
    return "\n---\n".join(relevant_chunks)