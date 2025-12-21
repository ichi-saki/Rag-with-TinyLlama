import sys
import torch
import numpy as np
import chromadb
import PyPDF2
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from pathlib import Path

class RAG:
    def __init__(self, chunk_size, chunk_overlap, k_value):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_value = k_value
        self.similarity_threshold = 0.7
        
        self.init_models()

    def init_models(self):
        print('Initializing models..')
        print('Loading TinyLlama..')
        self.tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.bfloat16, device_map='auto')

        print('Loading embeddings model..')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        print('Setting up vector database..')
        self.chroma_client = chromadb.PersistentClient(path='./chroma_db')

        self.collection_name = 'cs_handbook'
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            print(f'Loading collection: {self.collection_name}')
        except:
            self.collection = self.chroma_client.create_collection(name=self.collection_name, metadata={'hnsw:space': 'cosine'})
            print(f'created new collection: {self.collection_name}')
    
    def get_text(self):
        text = ""
        with open('data/cpsc-handbook-2022.pdf', 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"

            return text
        
    def get_chunks(self, text):
        chunks = []
        start = 0
        i = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

            print(f'Length of chunks: {len(chunks)} chunks created')
        return chunks
    
    def create_vector_index(self):
        print('\nCreating vector index..')

        if self.collection.count() > 0:
            print(f'collection has {self.collection.count()} items')
            return 
        
        text = self.get_text()
        chunks = self.get_chunks(text)

        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(embeddings=[embedding.tolist()],
                                documents=[chunk],
                                metadatas=[{"chunk_id": i,
                                            "source": "cpsc-handbook-2022.pdf",
                                            "length": len(chunk)}],
                                            ids=[f"chunk_{i}"])
        print(f'Indexed {len(chunks)} chunks')
        self.chunks = chunks
