import time

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer 

def query_question(question, collection_milvus, model):

    question_embedding = model.encode([question])

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    result = collection_milvus.search(question_embedding, "embeddings", search_params, limit=5, output_fields=["content", "pagelabel"])


    # getting the content from the search query 
    """
    for hits in result:
        for hit in hits:
            print(hit.entity.get("content"))
            print(f"page: https://www.trumpf.com/filestorage/TRUMPF_US/Landingpages/TruLaser_2030_fiber/pdfs/TruLaser-2030-Pre-Install-Manual.pdf#page={hit.entity.get('pagelabel')}")
            print("------------")

    # make the question to a vector
    """
    return result


def search_db(question):
    # num_entities = 3000
    # dim = 768 # dim of the embeddings
    collection_name = "test_new_bookv2"

    # connecting to localhost db
    connections.connect("default", host="localhost", port="19530")

    collection_milvus = Collection(name=collection_name)  # Get the collection object
    collection_milvus.load()  # Load the collection into memory

    # get embedding model
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    collection_milvus.load()

    results = query_question(question, collection_milvus, model)



    return results



print("\n")
found = search_db("What is the Error Number 40300?")

for hits in found:
    for hit in hits:
        print("\n")
        print(hit.entity.get("content"))