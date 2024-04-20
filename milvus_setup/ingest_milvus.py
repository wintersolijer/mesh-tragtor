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

version_name = "test_new_bookv2"

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

print(fmt.format(f"Create collection {version_name}"))
hello_milvus = Collection(version_name, schema, consistency_level="Strong")

# Function to extract text from PDF and convert to embeddings
# def process_pdf(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PdfReader(file)
#         text_arr = []
#         page_numbers = []
#         for page_num in range(len(reader.pages)):
#             page_split_num = 4
#             page_text_arr = split_text_into_parts(reader.pages[page_num].extract_text())
#             page_text = ' '.join(page_text_arr)
            
            
#             splitted_text = split_text_into_parts(page_text, page_split_num)
#             for i in range(page_split_num):
#                 page_numbers.append(page_num+1)
#                 text_arr.append(splitted_text[i])
            
#         return text_arr, page_numbers
    
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text_arr = []
        page_numbers = []
        for page_num in range(len(reader.pages)):
            page_numbers.append(page_num+1)
            page_text_arr = split_text_into_parts(reader.pages[page_num].extract_text())
            page_text = ' '.join(page_text_arr)
            text_arr.append(page_text)
                
                
            
        return text_arr, page_numbers

def split_text_into_parts(text, num_parts=4):
    words = text.split()
    total_words = len(words)

    part_size = total_words // num_parts

    parts = []
    # Create each part
    for i in range(num_parts):
        # Calculate the start index of this part
        start_index = i * part_size
        # Calculate the end index of this part
        if i == num_parts - 1:
            # Last part includes any remaining words
            end_index = total_words
        else:
            end_index = start_index + part_size
        # Extract the words for this part and join them into a string
        parts.append(' '.join(words[start_index:end_index]))
    return parts


# Sample usage:
pdf_path = "TRUMPF_Manual_TruConvert_DC_1030.pdf"  # Replace with your PDF file path
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



"""

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

"""