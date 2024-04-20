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
import json


version_name = "test_v19"

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 768

print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")


collection_name = "test_v19"
collection_milvus = Collection(name=collection_name)  # Get the collection object
collection_milvus.load()  # Load the collection into memory



# get embedding model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
dimension = dim
normalize_embeddings = True



# -------------------
# Query Question...

print(fmt.format("Start loading"))
collection_milvus.load()


# asking milvus

question = input("ask question:\n")

question_embedding = model.encode([question])

print(fmt.format("Start searching based on vector similarity"))

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

# start_time = time.time()
result = collection_milvus.search(question_embedding, "embeddings", search_params, limit=3, output_fields=["content", "pagelabel"])
# end_time = time.time()

for hits in result:
    for hit in hits:
        print(hit.entity.get("content"))
        print(f"page: https://www.trumpf.com/filestorage/TRUMPF_US/Landingpages/TruLaser_2030_fiber/pdfs/TruLaser-2030-Pre-Install-Manual.pdf#page={hit.entity.get('pagelabel')}")
        print("------------")
# print(search_latency_fmt.format(end_time - start_time))

# make the question to a vector
