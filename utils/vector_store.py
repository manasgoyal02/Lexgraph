import os
import pickle
from pathlib import Path
from datetime import datetime, timezone

import faiss
import numpy as np

from utils.embedder import get_embedding, get_embeddings

BASE_DIR = Path("output/vector_index")


def _slugify(value):
    text = str(value or "all_contracts").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in safe.split("_") if part) or "all_contracts"


def _paths(contract_name=None):
    slug = _slugify(contract_name)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = BASE_DIR / f"{slug}.faiss"
    meta_path = BASE_DIR / f"{slug}.pkl"
    return index_path, meta_path


def build_clause_index(clauses, contract_name=None):
    if not clauses:
        return False

    texts = [str(clause.get("clause_text", "")).strip() for clause in clauses]
    valid_rows = [(idx, text) for idx, text in enumerate(texts) if text]
    if not valid_rows:
        return False

    valid_texts = [row[1] for row in valid_rows]
    embeddings = np.array(get_embeddings(valid_texts), dtype="float32")
    if embeddings.size == 0:
        return False

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    metadata = []
    for source_idx, _ in valid_rows:
        clause = clauses[source_idx]
        metadata.append(
            {
                "contract_name": clause.get("contract_name"),
                "clause_id": clause.get("clause_id"),
                "clause_idx": clause.get("clause_idx"),
                "clause_number": clause.get("clause_number", ""),
                "clause_text": clause.get("clause_text", ""),
                "clause_type": clause.get("clause_type"),
            }
        )

    index_path, meta_path = _paths(contract_name=contract_name)
    faiss.write_index(index, str(index_path))
    payload = {
        "meta_version": 2,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_marker": None,
        "rows": metadata,
    }
    with open(meta_path, "wb") as file_obj:
        pickle.dump(payload, file_obj)
    return True


def load_clause_index(contract_name=None):
    index_path, meta_path = _paths(contract_name=contract_name)
    if not index_path.exists() or not meta_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as file_obj:
        payload = pickle.load(file_obj)

    if isinstance(payload, dict) and "rows" in payload:
        metadata = payload.get("rows", [])
    else:
        # Backward compatibility with old metadata format (list only).
        metadata = payload if isinstance(payload, list) else []
    return index, metadata


def has_clause_index(contract_name=None):
    index_path, meta_path = _paths(contract_name=contract_name)
    return index_path.exists() and meta_path.exists()


def set_index_source_marker(contract_name=None, source_marker=None):
    index_path, meta_path = _paths(contract_name=contract_name)
    if not index_path.exists() or not meta_path.exists():
        return False

    with open(meta_path, "rb") as file_obj:
        payload = pickle.load(file_obj)

    if isinstance(payload, dict) and "rows" in payload:
        payload["source_marker"] = source_marker
        if not payload.get("built_at"):
            payload["built_at"] = datetime.now(timezone.utc).isoformat()
    else:
        payload = {
            "meta_version": 2,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_marker": source_marker,
            "rows": payload if isinstance(payload, list) else [],
        }

    with open(meta_path, "wb") as file_obj:
        pickle.dump(payload, file_obj)
    return True


def get_index_state(contract_name=None):
    index_path, meta_path = _paths(contract_name=contract_name)
    if not index_path.exists() or not meta_path.exists():
        return {"exists": False, "built_at": None, "source_marker": None, "row_count": 0}

    with open(meta_path, "rb") as file_obj:
        payload = pickle.load(file_obj)

    if isinstance(payload, dict) and "rows" in payload:
        rows = payload.get("rows", [])
        return {
            "exists": True,
            "built_at": payload.get("built_at"),
            "source_marker": payload.get("source_marker"),
            "row_count": len(rows),
        }

    rows = payload if isinstance(payload, list) else []
    return {
        "exists": True,
        "built_at": None,
        "source_marker": None,
        "row_count": len(rows),
    }


def search_similar_clauses(question, top_k=5, contract_name=None):
    index, metadata = load_clause_index(contract_name=contract_name)
    if index is None or metadata is None:
        return []

    query_vec = get_embedding(question)
    if not query_vec:
        return []

    query_embedding = np.array([query_vec], dtype="float32")
    k = min(max(int(top_k), 1), len(metadata))
    scores, indices = index.search(query_embedding, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            row = dict(metadata[idx])
            row["semantic_score"] = float(scores[0][rank])
            results.append(row)
    return results
