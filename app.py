from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import numpy as np
from scipy.spatial.distance import cosine
import json

# Define the FastAPI app
app = FastAPI()

# Define the request body schema using Pydantic
class SearchRequest(BaseModel):
    docs: List[str]
    query: str

# URL for the external embedding API
EMBEDDING_API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6Imxpc2EubWlyYW5kYUBncmFtZW5lci5jb20ifQ.nvcT6zt6b65Hf-rJE1Q0bwn4KrAeGzGZ6lCi5RP3IhY"  # Replace with your actual API key

# Function to call the external embedding API
def get_embeddings(texts: List[str]) -> List[List[float]]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "text-embedding-3-small",  # or the specific model you want to use
        "input": texts
    }
    
    response = requests.post(EMBEDDING_API_URL, headers=headers, json=data)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch embeddings from the API.")
    
    embeddings = response.json().get("data", [])
    return [embedding['embedding'] for embedding in embeddings]

# Function to calculate cosine similarity
def calculate_cosine_similarity(query_embedding, doc_embeddings):
    similarities = []
    for doc_embedding in doc_embeddings:
        similarity = 1 - cosine(query_embedding, doc_embedding)
        similarities.append(similarity)
    return similarities

# POST endpoint for semantic search
@app.post("/similarity")
async def similarity_search(request: SearchRequest):
    # Combine the docs and query into a list of texts
    texts = request.docs + [request.query]

    # Get embeddings for the documents and the query
    embeddings = get_embeddings(texts)
    
    # The query embedding is the last one
    query_embedding = embeddings[-1]
    
    # Document embeddings are all but the last one
    doc_embeddings = embeddings[:-1]
    
    # Calculate cosine similarities between the query and each document
    similarities = calculate_cosine_similarity(query_embedding, doc_embeddings)
    
    # Get the indices of the top 3 most similar documents
    sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    top_3_indices = sorted_indices[:3]
    
    # Get the top 3 most similar documents
    top_3_docs = [request.docs[i] for i in top_3_indices]
    
    # Return the result
    return {"matches": top_3_docs}

# To enable CORS for all origins, methods, and headers
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]  # Allow all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST"],  # Allow OPTIONS and POST methods
    allow_headers=["*"],  # Allow all headers
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
