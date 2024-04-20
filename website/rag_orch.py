from openai_llm import ask_question
from ask_vectordb import search_db

def doRAG(question):
    
    results = search_db(question)
    
    context_passage = ""
    pages = []
    
    
    for hits in results:
        for hit in hits:
            context_passage += f"\n\n{hit.entity.get('content')}" 
            pages.append(hit.entity.get('pagelabel'))
    
    prompt = f"Anser the following question: {question}. With the given context here: {context_passage}. Only use information from the context to answer the question."
    
    llm_response = ask_question(prompt)
    
    response_json = {
        "llm_response": llm_response,
        "pagelabel": pages
    }
    
    return response_json


if __name__ == '__main__':
    print("\n\n\n")
    print(doRAG("What is the Error Number 40300?"))