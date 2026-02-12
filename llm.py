from utils import client, call_openrouter
import re

def chat(input_query, text):
    """
    Proverava konzistentnost 'input_query' u odnosu na 'text' koristeći Cohere Chat API.
    """

    # Preamble je sistemska poruka koja definiše zadatak za model
    preamble = """
Ti si AI koji proverava tačnost date izjave na osnovu priloženog konteksta.
- Ako je izjava tačna, odgovori sa "Konzistentno" i navedi obrazloženje.
- Ako je izjava netačna, odgovori sa: "Kontradiktorno" I pruži tačan odgovor i obrazloženje.
- Ako ne znaš, napiši "Ne znam" a obrazloženje je "Nema relevantnih podataka".

Postaraj se da tvoj odgovor bude **strogo zasnovan na priloženom kontekstu baze podataka**.
    """

    # Kontekst 'text' prosleđujemo kao 'documents'
    # Ovo je bolja praksa za RAG nego samo ubacivanje u prompt
    documents = []
    if isinstance(text, list):
        documents = [{"snippet": doc} for doc in text if isinstance(doc, str)]
    elif isinstance(text, str):
        documents = [{"snippet": text}]

    # Ako je kontekst prazan ili pogrešan, dodajemo podrazumevani
    if not documents:
        documents = [{"snippet": "Nema relevantnih podataka."}]

    response = client.chat(
        model="command-r-plus-08-2024",  # Ažuriran naziv modela
        preamble=preamble,
        message=input_query,
        documents=documents,
        max_tokens=200,
        temperature=0.4
    )

    # Vraćamo ceo 'response' objekat, jer rag.py očekuje da pristupi .text
    return response


def check(input_query):
    """
    Klasifikuje 'input_query' kao 1 (proverljivo) ili 0 (neproverljivo) koristeći Cohere Chat API.
    """

    preamble = """
Ti si veštačka inteligencija koja klasifikuje političke izjave da bi utvrdila da li ih treba proveriti.

Klasifikuj kao **"1"** ako izjava:
    - Sadrži **proverljivu činjenicu, statistiku, obećanje ili predviđanje** u vezi sa predsedničkim dužnostima.
    - Govori o **upravljanju, ekonomiji, politici, migracijama, zakonima ili diplomatiji, odnosima sa drugim zemljama**.
    - Je **specifična tvrdnja koja se može proveriti podacima**.

Klasifikuj kao **"0"** ako izjava:
    - Je **nejasna ili se tiče privatnog života**.
    - Je **motivacioni ili lični iskaz**, a ne činjenična tvrdnja.
    - **Ne može biti proverena objektivnim izvorima**.

Vrati samo `"1"` ili `"0"`, praćeno kratkim objašnjenjem na srpskom.
    """

    response = client.chat(
        model="command-r-plus-08-2024",  # Ažuriran naziv modela
        preamble=preamble,
        message=input_query,
        max_tokens=50,  # 50 je dovoljno za "1" ili "0" i kratko objašnjenje
        temperature=0.1
    )

    # Vraćamo ceo 'response' objekat
    return response


def chat_haiku(input_query, text):
    """
    Proverava konzistentnost 'input_query' u odnosu na 'text' koristeći Claude 3 Haiku.
    """

    # Definišemo sistemski prompt za Haiku
    # Važno: Haiku (preko OpenRouter) ne prima 'preamble' i 'documents' kao Cohere.
    # Moramo sve spojiti u jedan veliki prompt.

    prompt_template = """
Ti si AI koji proverava tačnost date izjave na osnovu priloženog konteksta.
- Ako je izjava tačna, odgovori sa "Konzistentno" i navedi obrazloženje.
- Ako je izjava netačna, odgovori sa: "Kontradiktorno" I pruži tačan odgovor i obrazloženje.
- Ako ne znaš, napiši "Ne znam" a obrazloženje je "Nema relevantnih podataka".

Postaraj se da tvoj odgovor bude **strogo zasnovan na priloženom kontekstu baze podataka**.

KONTEKST:
{context}

TVRDNJA:
{query}

ODGOVOR:
"""
    # 'text' je već string koji vraća find_most_relevant_text
    final_prompt = prompt_template.format(context=text, query=input_query)

    # call_openrouter vraća string
    return call_openrouter(final_prompt)

