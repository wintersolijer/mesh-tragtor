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

from PyPDF2 import PdfReader   # or any other PDF processing library
from sentence_transformers import SentenceTransformer 

version_name = "test_v19"

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 768

print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

fields = [
    # FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=False, auto_id=False, max_length=500),
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, is_primary=False, auto_id=False, max_length=2048),
    FieldSchema(name="pagelabel", dtype=DataType.INT64, is_primary=False, auto_id=False),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")

print(fmt.format("Create collection `hello_milvus`"))
hello_milvus = Collection(version_name, schema, consistency_level="Strong")

# Function to extract text from PDF and convert to embeddings
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text_arr = []
        page_numbers = []
        for page_num in range(len(reader.pages)):
            page_numbers.append(page_num+1)
            text_arr.append(reader.pages[page_num].extract_text())
        return text_arr, page_numbers




# Sample usage:
pdf_path = "TruLaser-2030-Pre-Install-Manual.pdf"  # Replace with your PDF file path
pdf_text_arr, page_num_arr = process_pdf(pdf_path)

# Convert text to embeddings
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
dimension = dim
normalize_embeddings = True


# embeddings_text = []
# for text in pdf_text_arr:
#     embedding = model.encode([text], normalize_embeddings=normalize_embeddings, dimension=dimension)
#     embeddings_text.append(embedding)

embeddings_text = model.encode(pdf_text_arr)


# pdf_embeddings = model.encode([pdf_text])

entities = [
    pdf_text_arr,
    page_num_arr, 
    embeddings_text,
]

insert_result = hello_milvus.insert(entities)


print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)


# -------------------
# Query Question...

print(fmt.format("Start loading"))
hello_milvus.load()


# asking milvus

question = "What is Authorized personnel?"
question_embedding = model.encode([question])

print(fmt.format("Start searching based on vector similarity"))

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = hello_milvus.search(question_embedding, "embeddings", search_params, limit=3, output_fields=["content"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, random field: {hit.entity.get('content')}")
        print("------------")
print(search_latency_fmt.format(end_time - start_time))

# make the question to a vector
