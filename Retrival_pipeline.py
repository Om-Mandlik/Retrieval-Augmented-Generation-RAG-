from Ingestion_pipeline import VectorStore,Embedding_Manager,vectorstore,embedding_manager
from typing import List, Dict, Any
class RAG_Retrival:
    def __init__(self,vector_store: VectorStore,embedding_manager: Embedding_Manager):
      self.vector_store = vector_store
      self.embedding_manager = embedding_manager
    
    def retrieve(self,query:str,top_K:int = 5 , score_threshold: float = 0.0)->List[Dict[str,Any]]:
        print(f"Retriving document for query:'{query}'")
        print(f"Top K: {top_K},Score threshold:{score_threshold}")
        query_embedding = self.embedding_manager.genrate_embedding([query])[0]
        try:
           results = self.vector_store.collection.query(
              query_embeddings=[query_embedding.tolist()],
              n_results=top_K
           )
           retrieved_docs = []
           if results['documents'] and results['documents'][0]:
              documents = results['documents'][0]
              metadatas = results['metadatas'][0]
              distances = results['distances'][0]
              ids = results['ids'][0]
              for i ,(doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                 similarity_score = 1 - distance
                 if similarity_score >= score_threshold:
                    retrieved_docs.append({
                       'id':doc_id,
                       'content':document,
                       'similarity_score': similarity_score,
                       'distance': distance,
                       'rank':i+1
                    })
                    print(f"Retrived {len(retrieved_docs)} document after filtering")
                 if not retrieved_docs:
                    print("No documents met the similarity threshold.")
           return retrieved_docs
        except Exception as e:
           print(f"Error during retrival {e}")

rag_retriever = RAG_Retrival(vectorstore,embedding_manager)

print(rag_retriever.retrieve("What is attention is all you need"))