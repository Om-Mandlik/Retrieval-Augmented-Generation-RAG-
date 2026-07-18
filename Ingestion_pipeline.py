import os
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List,Dict,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity  

def process_all_pdfs(pdfDirectory):
    all_documents = []
    pdf_Dir = Path(pdfDirectory)
    pdf_files = list(pdf_Dir.glob("**/*.pdf"))
    print(f"Found{len(pdf_files)}PDF Files in Directory")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")
        except Exception as e:
            print(f"Error{e}")
    print(f"\nTotal Document Loaded:{len(all_documents)}")
    return all_documents

all_pdf_documents = process_all_pdfs("./Data/pdfs")

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ",""],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function =len
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"\n\nSplit {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs

chunks = split_documents(all_pdf_documents)

class Embedding_Manager:
    def __init__(self,model_name: str= "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    def _load_model(self):
        try:
            print(f"Loading Embedding Model {self.model_name} ")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded Succesfully. Embedding Dimension:{self.model.get_embedding_dimension()}")    
        except Exception as e:
            print(f"Error Loading Model:{e}")
            raise
    def genrate_embedding(self,texts:List[str]) ->np.ndarray:
        if not self.model:
            raise ValueError("Model not found")
        print(f"Generating Embedding for {len(texts)} texts")
        embeddings = self.model.encode(texts,show_progress_bar=True)
        print(f"Generated Embedding with shape :{embeddings.shape}")
        return embeddings
    
embedding_manager = Embedding_Manager()

class VectorStore:
    def __init__(self,collection_name:str="pdf_documents",persist_directory:str="../Data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initalize_store()
    def _initalize_store(self):
        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"PDF document Embedding for RAG"}
            )
            print(f"Vector store initalized.Collection:{self.collection_name}")
            print(f"Exisiting documents in collection:{self.collection.count()}")
        except Exception as e:
            print(f"Error initalizing vectore store {e}")
            raise
    def add_documents(self,documents:List[Any],embedding:np.ndarray):
        if(len(documents)!=len(embedding)):
            raise ValueError(f"Number of documents must match number of embedding")
        print(f"Adding {len(documents)}documents to Vector store")
        ids = []
        metadatas = []
        documents_text = []
        embedding_list = []
        for i,(doc,embedding) in enumerate(zip(documents,embedding)):
            doc_id = f"{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())
        try:
            self.collection.add(
                ids = ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Succesfully added {len(documents)} in vector store")
            print(f"total document in collection {self.collection.count()}")
        except Exception as e:
            print(f"Error adding doc in vector store {e}")
            raise

vectorstore = VectorStore()

texts = [doc.page_content for doc in chunks]

embeddings = embedding_manager.genrate_embedding(texts)

vectorstore.add_documents(chunks,embeddings)

    