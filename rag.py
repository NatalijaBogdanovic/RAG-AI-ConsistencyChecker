from baza import find_most_relevant_text
from baza import create_database, load_query, embed_query, find_similarities
from llm import chat, check, chat_haiku
import os
import csv
import re
import time
import sys

create_database()  # pravimo memorijsku bazu iz dokumenata u folderu
print('DATABASE CREATED')

query_chunks = load_query()

csv_bp_file = open(os.path.join("response", "class0.csv"), "w", newline="", encoding='utf-8-sig')
csv_bp_writer = csv.writer(csv_bp_file, quoting=csv.QUOTE_ALL, delimiter=';')
csv_bp_writer.writerow(["Sentence", "Explanation"])

csv_p_file = open(os.path.join("response", "class1.csv"), "w", newline="", encoding='utf-8-sig')
csv_p_writer = csv.writer(csv_p_file, quoting=csv.QUOTE_ALL, delimiter=';')

# --- Ажурирамо заглавље CSV фајла ---
csv_p_writer.writerow([
    "Sentence",
    "Relevant_Context",
    "Response_Cohere",
    "Latency_Cohere",
    "Response_Haiku",
    "Latency_Haiku"
])

for i, chunk in enumerate(query_chunks):
    print("PROVERA " + str(i) + " of " + str(len(query_chunks)))

    try:
        response_class_obj = check(chunk)
        response_class_text = response_class_obj.text
    except Exception as e:
        print(f"Greška prilikom poziva check() API-ja: {e}")
        response_class_text = "0 - Greška pri klasifikaciji"  # Tretiraj kao 0

    if not response_class_text:
        classification = "0"
        explanation = "Prazan odgovor od klasifikatora."
    else:
        match = re.search(r'\b(1|0)\b', response_class_text)
        if match:
            classification = match.group(1)
        else:
            classification = "0"

        explanation = response_class_text.strip()
    # --- (Крај 'check' дела) ---

    if classification == "1":
        # 1. Get context
        most_relevant_text = find_most_relevant_text(chunk, top_n=3, similarity_threshold=0.5)

        # 2. Call Model A (Cohere) + merenje vremena
        try:
            start_time_cohere = time.time()  # Početak merenja
            response_chat_obj = chat(chunk, most_relevant_text)
            response_cohere = response_chat_obj.text
            latency_cohere = time.time() - start_time_cohere  # Kraj merenja
        except Exception as e:
            print(f"Greška prilikom poziva Cohere chat() API-ja: {e}")
            response_cohere = f"Greška: {e}"
            latency_cohere = 0

        # 3. Call Model B (Haiku) + merenje vremena
        try:
            start_time_haiku = time.time()  # Početak merenja
            # chat_haiku vraća string direktno
            response_haiku = chat_haiku(chunk, most_relevant_text)
            latency_haiku = time.time() - start_time_haiku  # Kraj merenja
        except Exception as e:
            print(f"Greška prilikom poziva Haiku chat() API-ja: {e}")
            response_haiku = f"Greška: {e}"
            latency_haiku = 0

        # 4. Write ALL data to CSV
        csv_p_writer.writerow([
            chunk,
            most_relevant_text,
            response_cohere,
            latency_cohere,
            response_haiku,
            latency_haiku
        ])

    else:
        # Tretiramo klasu "0"
        csv_bp_writer.writerow([chunk, explanation])


csv_bp_file.close()
csv_p_file.close()

print("Obrada završena.")