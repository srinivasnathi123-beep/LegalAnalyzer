import os
import re
import pickle
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

def chunk_text(text: str, filename: str) -> list:
    # Split by Section headers or double newlines using non-newline whitespace matching
    pattern = r'(SECTION[ \t]+\d+(?:\.[ \t]*[A-Z \t]+)?\.?|ARTICLE[ \t]+[IVXLCDM]+(?:\.[ \t]*[A-Z \t]+)?\.?|[\d\.]+[ \t]+[A-Z \t]{4,}\.?)'
    parts = re.split(pattern, text)
    
    chunks = []
    current_section = "PREAMBLE"
    
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue
            
        if re.match(pattern, part):
            current_section = part
            if i + 1 < len(parts):
                content = parts[i+1].strip()
                i += 2
            else:
                content = ""
                i += 1
        else:
            content = part
            i += 1
            
        if content:
            chunks.append({
                "text": f"Section: {current_section}\n{content}",
                "metadata": {
                    "contract_id": filename,
                    "contract_type": "NDA" if "nda" in filename else "SOW" if "sow" in filename else "Vendor Agreement",
                    "section_name": current_section,
                    "chunk_index": len(chunks)
                }
            })
    return chunks

def get_embedding(text: str, client: OpenAI) -> list:
    # Fallback dummy embedding if offline/mock key
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key or api_key in ["mock_key_or_fill_it", "your_nebius_api_key_here"]:
        return [0.0] * 4096
        
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model="Qwen/Qwen3-Embedding-8B"
    )
    return response.data[0].embedding

def ingest_all(contracts_dir="./contracts", data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "legal-contracts-index")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Pinecone Init
    pc = None
    if pinecone_api_key and pinecone_api_key not in ["mock_key_or_fill_it", "your_pinecone_api_key_here"]:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=4096,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            index = pc.Index(index_name)
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}. Falling back to local/offline mode.")
            pc = None
            
    all_chunks = []
    if not os.path.exists(contracts_dir):
        print(f"Contracts directory {contracts_dir} does not exist.")
        return
        
    for filename in os.listdir(contracts_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(contracts_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            all_chunks.extend(chunk_text(text, filename))
            
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Upload vectors and build BM25 corpora
    bm25_corpus = []
    for chunk in all_chunks:
        text_content = chunk["text"]
        bm25_corpus.append(text_content)
        
        if pc:
            try:
                vector = get_embedding(text_content, client)
                index.upsert(vectors=[(
                    f"{chunk['metadata']['contract_id']}_{chunk['metadata']['chunk_index']}",
                    vector,
                    {
                        "text": text_content,
                        "contract_id": chunk["metadata"]["contract_id"],
                        "contract_type": chunk["metadata"]["contract_type"],
                        "section_name": chunk["metadata"]["section_name"]
                    }
                )])
            except Exception as e:
                print(f"Error during upsert: {e}. Skipping Pinecone upsert for this chunk.")
    
    # Build and Save BM25
    tokenized_corpus = [doc.lower().split(" ") for doc in bm25_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(os.path.join(data_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)
        
    print("Ingestion & Indexing complete.")

if __name__ == "__main__":
    ingest_all()
