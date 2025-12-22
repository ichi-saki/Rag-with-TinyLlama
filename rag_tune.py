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

    def retrieve(self, query, k=None):
        k = k or self.k_value

        query_emb = self.embedding_model.encode([query])[0]

        results = self.collection.query(query_embeddings=[query_emb.tolist()],
                                        n_results=k,
                                        include=['documents', 'metadatas', 'distances'])
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []

        similarities = [1 - (dist / 2) for dist in distances]

        filtered_docs = []
        filtered_scores = []

        for doc, score in zip(documents, similarities):
            if score >= self.similarity_threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)

        return filtered_docs, filtered_scores
    
    def create_prompt(self, query, context_chunks):
        if context_chunks:
            text = "\n".join([f'[Context {i+1}] {chunk[:300]}..' if len(chunk) > 300 else chunk for i, chunk in enumerate(context_chunks)])

            prompt = f"""<|system|>
                You are a helpful assistant for CSU Fullerton Computer Science students.
                Answer the question based ONLY on the following context from the CS Handbook.
                If the information is not in the context, say "I cannot find that information."
                Context:
                {text}
                </s>
                <|user|>
                Question: {query}
                Please answer based ONLY on the provided context.
                </s>
                <|assistant|>
                Based on the CS Handbook, """
        else:
            prompt = f"""<|system|>
                You are a helpful assistant for CSU Fullerton Computer Science students.
                Answer the question to the best of your ability about computer science programs.
                </s>
                <|user|>
                {query}
                </s>
                <|assistant|>
                """
        
        return prompt

    def generate_response(self, query, **gen_kwargs):
        context_chunks, scores = self.retrieve(query)
        print(f'Retrieved {len(context_chunks)} relevant chunks')

        prompt = self.create_prompt(query, context_chunks)

        params = {
            "max_new_tokens": 300,
            "temperature": 0.1, 
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        generation_config = {**params, **gen_kwargs}

        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        device = next(self.llm.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.llm.generate(**inputs, **generation_config)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if '<|assistant|>' in response:
            response = response.split('<|assistant|>')[-1].strip()
        else:
            response = response[len(prompt):].strip()
        
        return response
