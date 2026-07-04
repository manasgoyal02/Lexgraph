# 🚀 LexGraph

> **Hybrid GraphRAG for Intelligent Contract Analysis**

LexGraph is an AI-powered contract intelligence system that combines **semantic retrieval**, **knowledge graphs (Neo4j)**, and **Large Language Models (LLMs)** to answer legal questions with high-quality evidence retrieval. Instead of relying solely on vector similarity, LexGraph enriches retrieval using graph relationships, clause types, and confidence-aware reasoning.

---

## ✨ Key Features

* 📄 Upload and analyze **PDF** or **TXT** contracts
* 🧩 Automatic contract parsing and clause segmentation
* 🏷️ Clause-type classification for targeted retrieval
* 🕸️ Knowledge graph construction using **Neo4j**
* 🔍 Hybrid retrieval combining:

  * Semantic vector search (FAISS)
  * Clause-type-aware retrieval
  * Multi-hop graph expansion
* 🤖 Confidence-aware LLM answer generation
* 🛡️ Unsafe claim detection and evidence-based guardrails
* 💬 Continuous contract chat with conversational context
* 📌 Supporting clause citations with every response

---

# 🏗️ Architecture

```text
                Contract (PDF / TXT)
                        │
                        ▼
              Clause Parsing & Extraction
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  Semantic Embeddings             Clause Classification
        │                               │
        └───────────────┬───────────────┘
                        ▼
              Neo4j Knowledge Graph
                        │
                        ▼
              Hybrid GraphRAG Retrieval
                        │
                        ▼
          Confidence-aware LLM Answer
                        │
                        ▼
        Answer + Supporting Clause References
```

---

# 📂 Project Structure

```text
LexGraph/
│
├── app.py                         # Streamlit application
├── requirements.txt
├── README.md
│
├── tests/
│   └── test_clause_parser.py
│
└── utils/
    ├── answer_generator.py
    ├── clause_parser.py
    ├── embedder.py
    ├── extractor.py
    ├── graph_builder.py
    ├── llm_extractor.py
    ├── neo4j_handler.py
    ├── pdf_loader.py
    ├── query_engine.py
    └── vector_store.py
```

---

# ⚙️ Technology Stack

| Category       | Technologies                        |
| -------------- | ----------------------------------- |
| Language       | Python                              |
| UI             | Streamlit                           |
| Vector Search  | FAISS                               |
| Graph Database | Neo4j                               |
| LLM API        | OpenAI Compatible APIs / OpenRouter |
| Embeddings     | HuggingFace Models                  |
| Retrieval      | Hybrid GraphRAG                     |

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/manasgoyal02/Lexgraph.git
cd Lexgraph
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

### LLM Configuration

```env
OPENAI_API_KEY=your_key
MODEL_NAME=your_model
```

### Optional Configuration

```env
OPENAI_BASE_URL=https://openrouter.ai/api/v1

HF_TOKEN=your_huggingface_token

MODEL_NAME_EMBEDDING=qwen/qwen3-embedding-0.6b
MODEL_NAME_EMBEDDING_FALLBACK=nvidia/llama-nemotron-embed-vl-1b-v2:free
```

### Neo4j Configuration

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

---

# 🔄 Retrieval Pipeline

LexGraph follows a multi-stage retrieval pipeline:

1. Parse uploaded contract into clauses.
2. Generate semantic embeddings.
3. Detect clause types.
4. Build or update the Neo4j knowledge graph.
5. Classify the user's legal query.
6. Select retrieval strategy:

   * High confidence → Clause-aware hybrid retrieval
   * Medium confidence → Balanced retrieval
   * Low confidence → Semantic-first retrieval
7. Expand retrieved context through graph neighbors.
8. Recover missing governing clauses when applicable.
9. Generate an evidence-grounded response using the LLM.
10. Return the answer along with supporting clauses.

---

# 🛡️ Reliability & Safety

### Confidence-Aware Answering

The system evaluates retrieval quality before generating a response and adjusts answer confidence accordingly.

### Unsafe Claim Guard

LexGraph avoids unsupported legal conclusions by:

* Blocking unsupported hard-negative claims
* Rewriting uncertain responses into evidence-backed statements
* Restricting answers to retrieved context only

---

# 🧪 Testing

Run the parser tests:

```bash
python tests/test_clause_parser.py
```

Compile important modules:

```bash
python -m py_compile app.py
python -m py_compile utils/query_engine.py
python -m py_compile utils/answer_generator.py
```

---

# 📌 Future Improvements

* Automated evaluation benchmark for legal QA
* Retrieval quality metrics
* Graph visualization dashboard
* Multi-document contract reasoning
* Clause citation scoring
* Support for additional legal document formats

---

# 📜 License

This project is intended for educational and research purposes.
