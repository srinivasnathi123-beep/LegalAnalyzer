import os
import pickle
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def reciprocal_rank_fusion(dense_results: list, sparse_results: list, top_n=5, k=60) -> list:
    rrf_scores = {}
    chunk_lookup = {}
    
    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        chunk_lookup[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        chunk_lookup[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [chunk_lookup[doc_id] for doc_id in sorted_ids[:top_n]]

def hybrid_search(query: str, metadata_filter: dict = None, top_k=20, top_n=5) -> list:
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "legal-contracts-index")
    
    # Sparse Local Load
    bm25_data_path = "./data/bm25_index.pkl"
    if not os.path.exists(bm25_data_path):
        return []
        
    with open(bm25_data_path, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    chunks = data["chunks"]
    
    # Filter chunks by metadata locally
    candidate_chunks = []
    for idx, chunk in enumerate(chunks):
        matched = True
        if metadata_filter:
            for key, value in metadata_filter.items():
                if chunk["metadata"].get(key) != value:
                    matched = False
                    break
        if matched:
            candidate_chunks.append((idx, chunk))
            
    # 1. Sparse Search
    tokenized_query = query.lower().split(" ")
    sparse_results = []
    if candidate_chunks:
        scores = bm25.get_scores(tokenized_query)
        # Sort candidates by BM25 score
        ranked_candidates = sorted(candidate_chunks, key=lambda x: scores[x[0]], reverse=True)
        for rank, (idx, chunk) in enumerate(ranked_candidates[:top_k]):
            sparse_results.append({
                "id": f"{chunk['metadata']['contract_id']}_{chunk['metadata']['chunk_index']}",
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
            
    # 2. Dense Search
    dense_results = []
    if pinecone_api_key and pinecone_api_key not in ["mock_key_or_fill_it", "your_pinecone_api_key_here"]:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(index_name)
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            response = client.embeddings.create(
                input=[query.replace("\n", " ")],
                model="Qwen/Qwen3-Embedding-8B"
            )
            query_vector = response.data[0].embedding
            
            # Convert filters to Pinecone structure
            pc_filter = {}
            if metadata_filter:
                for key, val in metadata_filter.items():
                    pc_filter[key] = {"$eq": val}
                    
            matches = index.query(
                vector=query_vector,
                top_k=top_k,
                filter=pc_filter if pc_filter else None,
                include_metadata=True
            )
            for match in matches.matches:
                dense_results.append({
                    "id": match.id,
                    "text": match.metadata["text"],
                    "metadata": match.metadata
                })
        except Exception as e:
            print(f"Error querying Pinecone: {e}. Falling back to sparse search results for dense.")
            dense_results = list(sparse_results)
    else:
        # If offline, just reuse sparse results to simulate dense retrieval match
        dense_results = list(sparse_results)
        
    # 3. Fuse
    return reciprocal_rank_fusion(dense_results, sparse_results, top_n=top_n)

if __name__ == "__main__":
    res = hybrid_search("limitation of liability")
    print(f"Search found {len(res)} results.")
