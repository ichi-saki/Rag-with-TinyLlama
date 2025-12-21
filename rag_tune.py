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
    