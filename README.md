# LexGraph

Contract intelligence system with hybrid GraphRAG:
- semantic retrieval
- clause-type-aware retrieval
- graph neighborhood expansion (multi-hop)
- confidence-aware LLM answering
- unsafe-claim guardrails
- continuous contract chat

## Features

- Upload `PDF` or `TXT` contracts in Streamlit.
- Parse and split contracts into clauses.
- Extract legal entities and clause-type signals.
- Build/sync contract graph in Neo4j.
- Retrieve context via:
  - clause-type classification
  - semantic vector search
  - graph expansion (nearby and concept-linked clauses)
  - missing-clause recovery for expected governing clause families
- Generate final legal answers from LLM using retrieved context only.
- Return supporting clauses with each answer.
- Continuous chat with retrieval over same contract.

## Project Structure

- `app.py` - Streamlit application
- `utils/query_engine.py` - GraphRAG orchestration and retrieval flow
- `utils/answer_generator.py` - LLM synthesis, confidence policy, unsafe-claim guard
- `utils/graph_builder.py` - Neo4j graph construction
- `utils/vector_store.py` - FAISS vector index build/load/search
- `utils/clause_parser.py` - clause segmentation
- `utils/extractor.py` - entity and rule extraction
- `utils/pdf_loader.py` - PDF text extraction
- `utils/neo4j_handler.py` - Neo4j client wrapper

## Prerequisites

- Python 3.12+
- Neo4j database (for full GraphRAG path)
- API key for LLM provider compatible with OpenAI SDK

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create `.env` in project root.

Required for LLM answering:

```env
OPENAI_API_KEY=your_key
MODEL_NAME=your_model_name
```

Optional:

```env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
HF_TOKEN=your_huggingface_token
MODEL_NAME_EMBEDDING=qwen/qwen3-embedding-0.6b
MODEL_NAME_EMBEDDING_FALLBACK=nvidia/llama-nemotron-embed-vl-1b-v2:free
```

Required for Neo4j GraphRAG:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## Run

```bash
python -m streamlit run app.py
```

## Retrieval and Answering Flow

1. Parse contract and build clause corpus.
2. Classify user query into clause type(s) with confidence.
3. Route retrieval:
   - high confidence: typed-priority hybrid
   - medium confidence: balanced hybrid
   - low confidence: semantic-priority
4. Merge typed + semantic candidates.
5. Expand with graph context (adaptive multi-hop).
6. Run missing-clause detection for governing clause families and recover missing types when possible.
7. Send reduced evidence context to LLM (not full contract dump).
8. Return:
   - final legal answer
   - supporting clauses

## Safety and Reliability Layers

- Confidence-aware answering:
  - computes answer confidence from retrieval quality and completeness
  - applies cautious response policy on medium/low confidence
- Unsafe-claim guard:
  - blocks hard negative/conclusive statements unless evidence is strong
  - rewrites risky claims to evidence-scoped language

## Notes

- If Neo4j is unavailable, app has fallback behavior, but full graph expansion and typed graph retrieval require Neo4j.
- First embedding model load can be slow due to model download/cache warm-up.

## Testing

Current tests:

```bash
python tests/test_clause_parser.py
```

Compile checks:

```bash
python -m py_compile app.py
python -m py_compile utils/query_engine.py
python -m py_compile utils/answer_generator.py
```

## Suggested Next Step

Add an automated evaluation set (20-50 legal queries with expected clause references) to track retrieval/citation quality over changes.

