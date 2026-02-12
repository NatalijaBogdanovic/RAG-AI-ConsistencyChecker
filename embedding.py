from utils import client
def embed_query_cohere(query): # embedduje upit
    response = client.embed( # vraca embedding za query
        model="embed-multilingual-v3.0",  # Specify the multilingual model
        texts=query,
        input_type="search_query"
    )
    return response.embeddings

def embed_doc_cohere(source_sentences): #embedduje dokumente
    response = client.embed(    # embedding niza recenica, vraca niz embeddinga
        model="embed-multilingual-v3.0",  #model
        texts=source_sentences,
        input_type="search_document" 
    )
    return response.embeddings


