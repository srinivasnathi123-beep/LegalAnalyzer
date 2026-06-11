# Legal Contract Risk and Compliance Analyzer

An automated system designed to ingest, retrieve, and evaluate legal contracts (such as NDAs, SOWs, and Vendor Agreements) for compliance gaps and high-risk clauses. The engine combines **Pinecone Dense Search** and a **Local BM25 Sparse Index** using **Reciprocal Rank Fusion (RRF)**, backed by a **Pydantic Validation Agent** powered by **Llama-3.3-70B** on **Nebius AI Studio** (with a robust offline regex rules-engine fallback).

---

## Key Features

- **Structural Section Chunking**: Intelligently splits contracts by section headers and clause boundaries rather than arbitrary token counts.
- **RRF Hybrid Search**: Merges keyword-based BM25 sparse results with dense vector similarity scores using Reciprocal Rank Fusion (RRF).
- **Pydantic Validation Agent**: Leverages LLM schema validation to output clean, structured risk profiles.
- **Resilient Offline Fallback**: Automatically activates a local regular-expression-based matching engine if API keys are absent or endpoints are unreachable.
- **CLI Chat Interface**: Includes an interactive console using `rich` terminal panels to run queries, summarize contracts, or isolate search filters.
- **Verification Harness**: Validates parsing, retrieval, and schema serialization with preset evaluation queries.

---

## Directory Structure

```text
LegalAnalyzer/
├── contracts/               # Generated synthetic contract text files (.txt)
├── data/                    # Serialized local BM25 index and chunk repository
├── docs/                    # Design specifications and architecture document
├── tests/                   # Pytest unit tests for chunking, RRF, and fallback
├── .env.template            # Example template for setting API keys
├── .env                     # Local configuration credentials (gitignored)
├── requirements.txt         # Project package dependencies
├── generate_contracts.py    # Script to create 50 synthetic legal agreements
├── ingest.py                # Text parser, chunking, embedding, and indexing pipeline
├── search.py                # Hybrid search implementation using BM25 and RRF
├── agent.py                 # Pydantic schemas, LLM analyzer, and local rules engine fallback
├── chat.py                  # Interactive CLI chat console interface
├── verify.py                # Verification harness to run end-to-end queries
├── gamma_deck_spec.md       # Pre-formatted presentation deck spec for Gamma AI
└── README.md                # This instructions file
```

---

## Setup Instructions

### 1. Create and Activate a Virtual Environment
From your project workspace root, open your terminal (PowerShell or CMD) and run:

```powershell
# Create the virtual environment
python -m venv venv

# Activate it:
# Option A: PowerShell
.\venv\Scripts\Activate.ps1

# Option B: Command Prompt (CMD)
.\venv\Scripts\activate.bat
```

### 2. Install Project Dependencies
Run the following to install all necessary dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pinecone
```

### 3. Environment Variables Configuration
Duplicate `.env.template` to a new file named `.env` in the root folder, and enter your actual credentials:

```text
NEBIUS_API_KEY=your_actual_nebius_api_key_here
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
PINECONE_API_KEY=your_actual_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=legal-contracts-index
```

> **Note**: If `NEBIUS_API_KEY` or `PINECONE_API_KEY` are left empty, are missing, or contain placeholders, the system will automatically degrade gracefully to local rules-matching and offline mockup mode.

---

## How to Run the Pipelines

Run the following scripts in sequence inside your terminal:

### Step 1: Generate Mock Contracts
Creates 50 synthetic contracts containing structured agreements (NDAs, SOWs, and Vendor Agreements) containing high-density risks.
```powershell
python generate_contracts.py
```

### Step 2: Chunk and Ingest Contracts
Reads contract files, splits them into structural section chunks, generates vector embeddings using **Qwen3-Embedding-8B (4096 dimensions)**, upserts them to **Pinecone** (if keys exist), and serializes the local **BM25** index.
```powershell
python ingest.py
```

### Step 3: Run Verification Harness
Evaluates retrieval capability and verifies Pydantic compliance extraction across 10 preset search topics.
```powershell
python verify.py
```

### Step 4: Run the CLI Console
Starts the interactive CLI panel console to query and inspect contracts.
```powershell
python chat.py
```

---

## CLI Chat Commands

Once `chat.py` is running, type any of the following commands in the prompt:

| Command | Action | Example |
| :--- | :--- | :--- |
| `/summarize <file_name>` | Generates a full, structured risk profile report for that contract. | `/summarize contract_001_nda.txt` |
| `/file <file_name> <query>` | Restricts search to a single contract file to answer a specific question. | `/file contract_002_sow.txt What is the governing law?` |
| `<query>` | Performs a global hybrid RRF search across all 50 contracts in the library. | `limitation of liability cap auto-renew` |
| `/exit` | Exits the interactive chat shell loop. | `/exit` |

---

## Models In Use

The application uses modern high-performance endpoints hosted via **Nebius AI Studio**:
- **LLM Reasoner**: `meta-llama/Llama-3.3-70B-Instruct`
- **Embedding Model**: `Qwen/Qwen3-Embedding-8B` (Output Dimension: `4096`)
