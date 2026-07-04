import hashlib
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st

from utils.answer_generator import generate_answer
from utils.clause_parser import split_into_clauses
from utils.embedder import get_embedding, get_embeddings
from utils.extractor import extract_basic_entities
from utils.graph_builder import create_contract_graph
from utils.pdf_loader import extract_text_from_pdf, strip_heading_markers
from utils.query_engine import ask_question


st.set_page_config(page_title="LexGraph", layout="wide")


@st.cache_data(show_spinner=False)
def load_text_from_upload(filename: str, file_bytes: bytes) -> tuple[str, str, str]:
    os.makedirs("data/sample_contracts", exist_ok=True)
    safe_filename = Path(filename).name
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file_path = os.path.join("data/sample_contracts", safe_filename)

    should_write = True
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as existing:
                existing_hash = hashlib.sha256(existing.read()).hexdigest()
            should_write = existing_hash != file_hash
        except Exception:
            should_write = True

    if should_write:
        with open(file_path, "wb") as f:
            f.write(file_bytes)

    if safe_filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    return safe_filename, text, file_hash


@st.cache_data(show_spinner=False)
def process_document(text: str):
    clauses = split_into_clauses(text, structured=True)
    clause_texts = [c["text"] for c in clauses]
    clean_text = strip_heading_markers(text)
    extracted = extract_basic_entities(clean_text, clause_texts, use_llm=True)
    extracted["all_clauses"] = clauses
    return extracted


@st.cache_data(show_spinner=False)
def build_semantic_clause_rows(contract_name: str, clause_records: tuple[tuple[str, str], ...]):
    rows = []
    texts = []
    for idx, (number, text) in enumerate(clause_records, start=1):
        clean_text = str(text or "").strip()
        if not clean_text:
            continue
        rows.append(
            {
                "contract_name": contract_name,
                "clause_id": f"{contract_name}::clause::{idx}",
                "clause_idx": idx,
                "clause_number": number if str(number).strip() else str(idx),
                "clause_text": clean_text,
            }
        )
        texts.append(clean_text)

    if not texts:
        return [], []

    embeddings = get_embeddings(texts)
    if not embeddings:
        return [], []

    return rows, embeddings


def semantic_retrieve(question: str, rows: list[dict], embeddings: list[list[float]], top_k: int):
    if not rows or not embeddings:
        return []

    query_vec = get_embedding(question)
    if not query_vec:
        return []

    embs = np.array(embeddings, dtype="float32")
    q = np.array(query_vec, dtype="float32")

    if embs.ndim != 2 or q.ndim != 1 or embs.shape[1] != q.shape[0]:
        return []

    # Cosine similarity with numerical guard.
    embs_norm = np.linalg.norm(embs, axis=1)
    q_norm = np.linalg.norm(q)
    denom = np.maximum(embs_norm * max(float(q_norm), 1e-8), 1e-8)
    scores = (embs @ q) / denom

    k = min(max(int(top_k), 1), len(rows))
    best_idx = np.argsort(scores)[::-1][:k]

    out = []
    for i in best_idx:
        row = dict(rows[int(i)])
        row["semantic_score"] = float(scores[int(i)])
        row["source"] = "semantic_local"
        out.append(row)
    return out


def run_semantic_llm_answer(
    question: str,
    contract_name: str,
    rows: list[dict],
    embeddings: list[list[float]],
    top_k: int,
    chat_history: list[dict] | None = None,
):
    # Primary path: GraphRAG engine (typed routing + semantic + nearby-node expansion).
    try:
        query_result = ask_question(
            question=question,
            contract_name=contract_name,
            top_k=top_k,
            chat_history=chat_history,
        )
        hits = query_result.get("clauses", [])
        if hits:
            answer_payload = generate_answer(query_result, chat_history=chat_history)
            return hits, answer_payload, query_result
    except Exception:
        pass

    # Fallback path: local semantic retrieval if graph path is unavailable.
    hits = semantic_retrieve(question, rows, embeddings, top_k=top_k)
    if not hits:
        return None, None, None

    query_result = {
        "question": question,
        "detected_type": None,
        "clauses": hits,
        "retrieval_mode": "semantic_local_fallback",
        "type_confidence": 0.0,
    }
    answer_payload = generate_answer(query_result, chat_history=chat_history)
    return hits, answer_payload, query_result


st.title("LexGraph")
st.caption("Semantic retrieval + LLM-first answer generation.")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Contract", type=["pdf", "txt"])
    top_k = st.slider("Semantic Top K", min_value=2, max_value=12, value=6, step=1)
    auto_sync_neo4j = st.toggle("Auto-sync Neo4j (once/file)", value=True)

if uploaded_file is None:
    st.info("Upload a PDF or TXT contract to start.")
    st.stop()

safe_filename, text, file_hash = load_text_from_upload(uploaded_file.name, uploaded_file.getvalue())

if st.session_state.get("chat_contract_marker") != f"{safe_filename}:{file_hash}":
    st.session_state["chat_contract_marker"] = f"{safe_filename}:{file_hash}"
    st.session_state["qa_chat_history"] = []

with st.spinner("Parsing contract..."):
    extracted = process_document(text)

# Persist extracted JSON only when input changes.
os.makedirs("output/extracted_json", exist_ok=True)
json_path = os.path.join("output/extracted_json", safe_filename + ".json")
json_marker = f"{safe_filename}:{file_hash}:semantic_llm"
if st.session_state.get("json_marker") != json_marker:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)
    st.session_state["json_marker"] = json_marker

all_clauses = extracted.get("all_clauses", [])
clause_records = tuple((str(c.get("number", "")), str(c.get("text", ""))) for c in all_clauses)

with st.spinner("Building semantic index..."):
    clause_rows, clause_embeddings = build_semantic_clause_rows(safe_filename, clause_records)

m1, m2, m3 = st.columns(3)
m1.metric("Contract", safe_filename)
m2.metric("Clauses Indexed", len(clause_rows))
m3.metric("Answer Mode", "LLM (Default)")

st.markdown("### Neo4j Graph")
neo4j_marker = f"{safe_filename}:{file_hash}"
if auto_sync_neo4j and st.session_state.get("neo4j_sync_attempt_marker") != neo4j_marker:
    st.session_state["neo4j_sync_attempt_marker"] = neo4j_marker
    try:
        with st.spinner("Auto-syncing contract graph to Neo4j..."):
            create_contract_graph(extracted, safe_filename)
        st.session_state["neo4j_sync_success_marker"] = neo4j_marker
        st.success("Auto-sync complete for this uploaded file.")
    except Exception as exc:
        st.warning(f"Auto-sync skipped/failed: {exc}")

if st.button("Sync Contract Graph to Neo4j", use_container_width=True):
    try:
        with st.spinner("Writing contract graph to Neo4j..."):
            create_contract_graph(extracted, safe_filename)
        st.session_state["neo4j_sync_success_marker"] = neo4j_marker
        st.success("Graph synced to Neo4j successfully.")
    except Exception as exc:
        st.warning(f"Neo4j sync failed: {exc}")

st.markdown("### Ask a Question")
question = st.text_input("Question", placeholder="Example: Summarize termination and liability obligations.")

if st.button("Get Answer", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not clause_rows:
        st.warning("No clauses were indexed. Upload a readable contract.")
    else:
        with st.spinner("Running semantic retrieval + LLM..."):
            hits, answer_payload, query_result = run_semantic_llm_answer(
                question=question,
                contract_name=safe_filename,
                rows=clause_rows,
                embeddings=clause_embeddings,
                top_k=top_k,
            )

        if not hits or not answer_payload or not query_result:
            st.warning("Semantic retrieval did not return context. Try a more specific legal query.")
        else:
            st.markdown("### Final Legal Answer")
            st.write(answer_payload.get("answer", "No answer generated."))

            with st.expander("Retrieval Diagnostics"):
                st.write("Retrieval Mode:", query_result.get("retrieval_mode"))
                st.write("Detected Type:", query_result.get("detected_type"))
                st.write("Detected Types:", query_result.get("detected_types"))
                st.write("Type Confidence:", query_result.get("type_confidence"))
                st.write("Typed Matches:", query_result.get("typed_count"))
                st.write("Semantic Matches:", query_result.get("semantic_count"))
                st.write("Graph Expanded:", query_result.get("graph_expanded_count"))
                st.write("Missing-Clause Recovery:", query_result.get("missing_clause_recovery_count"))
                missing_check = query_result.get("missing_clause_check") or {}
                st.write("Missing-Clause Check Complete:", missing_check.get("is_complete"))
                st.write("Expected Governing Types:", missing_check.get("expected_types"))
                st.write("Missing Governing Types:", missing_check.get("missing_types"))

            st.markdown("### Supporting Clauses")
            citations = answer_payload.get("citations", [])
            if citations:
                for idx, row in enumerate(citations, start=1):
                    st.markdown(f"**[{idx}] Clause {row.get('clause_number', 'Unknown')}**")
                    snippet = row.get("clause_text", "")
                    snippet = snippet[:500] + "..." if len(snippet) > 500 else snippet
                    st.code(snippet)
            else:
                st.info("No supporting clauses available.")

            with st.expander("LLM Status"):
                st.write("LLM Called:", answer_payload.get("llm_called"))
                st.write("LLM Success:", answer_payload.get("llm_success"))
                st.write("LLM Provider:", answer_payload.get("llm_provider"))
                st.write("LLM Model:", answer_payload.get("llm_model_used"))
                st.write("LLM Error:", answer_payload.get("llm_error"))

st.markdown("### Continuous Chat")
if st.button("Clear Chat"):
    st.session_state["qa_chat_history"] = []
    st.rerun()

chat_history = st.session_state.get("qa_chat_history", [])
for msg in chat_history:
    with st.chat_message(msg.get("role", "assistant")):
        st.write(msg.get("content", ""))
        if msg.get("citations"):
            with st.expander("Supporting Clauses"):
                for idx, row in enumerate(msg["citations"], start=1):
                    st.markdown(f"**[{idx}] Clause {row.get('clause_number', 'Unknown')}**")
                    snippet = row.get("clause_text", "")
                    snippet = snippet[:500] + "..." if len(snippet) > 500 else snippet
                    st.code(snippet)

chat_prompt = st.chat_input("Ask follow-up questions about this contract...")
if chat_prompt:
    st.session_state["qa_chat_history"].append({"role": "user", "content": chat_prompt})
    prior_history = st.session_state["qa_chat_history"][:-1]

    with st.spinner("Running semantic retrieval + LLM..."):
        hits, answer_payload, query_result = run_semantic_llm_answer(
            question=chat_prompt,
            contract_name=safe_filename,
            rows=clause_rows,
            embeddings=clause_embeddings,
            top_k=top_k,
            chat_history=prior_history,
        )

    if not hits or not answer_payload:
        assistant_text = "I could not find relevant context for that follow-up. Try adding a legal keyword from the contract."
        st.session_state["qa_chat_history"].append({"role": "assistant", "content": assistant_text, "citations": []})
    else:
        st.session_state["qa_chat_history"].append(
            {
                "role": "assistant",
                "content": answer_payload.get("answer", "No answer generated."),
                "citations": hits,
                "debug": {
                    "retrieval_mode": (query_result or {}).get("retrieval_mode"),
                    "detected_type": (query_result or {}).get("detected_type"),
                    "detected_types": (query_result or {}).get("detected_types"),
                    "type_confidence": (query_result or {}).get("type_confidence"),
                },
            }
        )
    st.rerun()

with st.expander("Quick Overview"):
    st.markdown("**Parties**")
    st.write(extracted.get("parties") or "No parties detected")
    st.markdown("**Jurisdiction**")
    st.write(extracted.get("jurisdiction") or "No jurisdiction detected")
    st.markdown("**Clause Type Map**")
    st.json(extracted.get("clause_types", {}))
