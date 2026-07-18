from Retrival_pipeline import rag_retriever

from google import genai
import os

GEMINI_API_KEY="your_api_key_here"
client = genai.Client(api_key=GEMINI_API_KEY)


def rag_simple(query,retriver,llm,top_k=3):
    retrieved_docs = retriver.retrieve(query, top_K=top_k)
    context = "\n\n".join([doc['content'] for doc in retrieved_docs]) if retriver else ""
    if not context:
        return "no relevent context is found"
    prompt = f"Use the following context to answer the question concisely.Context:{context} Question {query}"

    response = client.models.generate_content(model="gemini-3.5-flash",contents=prompt)
    return response.text

answer = rag_simple("What is attention mechanism ",rag_retriever,client)
print(answer)

