from utils.neo4j_handler import Neo4jHandler
from utils.vector_store import (
    search_similar_clauses,
    build_clause_index,
    has_clause_index,
    load_clause_index,
    get_index_state,
    set_index_source_marker,
)
from utils.embedder import get_last_debug

CLAUSE_TYPE_KEYWORDS = {
    "confidentiality": ("confidential", "privacy", "disclosure", "nda"),
    "liability": ("liability", "damages", "loss", "limit of liability"),
    "termination": ("termination", "terminate", "end contract", "expire", "breach"),
    "payment": ("payment", "invoice", "fee", "pricing", "amount due", "due date"),
    "governing_law": ("governing law", "jurisdiction", "applicable law"),
    "dispute_resolution": ("dispute", "arbitration", "mediation", "court"),
    "force_majeure": ("force majeure", "act of god"),
    "assignment": ("assignment", "assign", "transfer rights"),
    "warranty": ("warranty", "warranties", "guarantee"),
    "ip": ("intellectual property", "ip", "copyright", "patent", "trademark"),
    "indemnity": ("indemnity", "indemnify", "hold harmless"),
}

# Expected governing clause families for each primary legal question type.
# This is heuristic and meant to reduce missed critical context before LLM synthesis.
EXPECTED_GOVERNING_TYPES = {
    "termination": ("termination", "liability", "indemnity", "dispute_resolution"),
    "liability": ("liability", "indemnity", "warranty"),
    "indemnity": ("indemnity", "liability", "dispute_resolution"),
    "payment": ("payment", "termination"),
    "dispute_resolution": ("dispute_resolution", "governing_law"),
    "governing_law": ("governing_law", "dispute_resolution"),
    "warranty": ("warranty", "liability", "indemnity"),
    "assignment": ("assignment", "termination"),
    "force_majeure": ("force_majeure", "termination"),
    "ip": ("ip", "confidentiality", "liability"),
    "confidentiality": ("confidentiality", "termination", "indemnity"),
}


def _build_conversation_context(chat_history, max_turns=4):
    if not chat_history:
        return ""
    turns = chat_history[-max_turns:]
    lines = []
    for row in turns:
        role = str(row.get("role", "")).strip().lower()
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    return "\n".join(lines)


def classify_question(question, chat_history=None):
    text = str(question or "").lower()
    chat_text = _build_conversation_context(chat_history)
    if chat_text:
        text = f"{text}\n{chat_text.lower()}"

    scores = {}
    for clause_type, keywords in CLAUSE_TYPE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        if score > 0:
            scores[clause_type] = score

    if not scores:
        return {
            "primary_type": None,
            "candidate_types": [],
            "confidence": 0.0,
            "scores": {},
        }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_type, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Confidence heuristic:
    # - strong when top score is >=2 and separated from second.
    # - medium when top score is 1 and close to second.
    if top_score >= 2 and top_score >= second_score + 1:
        confidence = 0.85
    elif top_score >= 2:
        confidence = 0.7
    elif top_score == 1 and second_score == 0:
        confidence = 0.55
    else:
        confidence = 0.4

    candidate_types = [t for t, _ in ranked[:2]]

    return {
        "primary_type": primary_type,
        "candidate_types": candidate_types,
        "confidence": float(confidence),
        "scores": scores,
    }


def detect_clause_type(question):
    classification = classify_question(question)
    return classification.get("primary_type")


def _fetch_clause_corpus(contract_name=None, limit=3000):
    db = Neo4jHandler()
    try:
        if contract_name:
            query = """
            MATCH (c:Contract {name: $contract_name})-[:HAS_CLAUSE]->(cl:Clause)
            OPTIONAL MATCH (cl)-[:HAS_TYPE]->(ct:ClauseType)
            RETURN c.name AS contract_name,
                   cl.id AS clause_id,
                   cl.idx AS clause_idx,
                   cl.number AS clause_number,
                   cl.text AS clause_text,
                   collect(DISTINCT ct.name) AS clause_types
            ORDER BY cl.idx ASC
            LIMIT $limit
            """
            params = {"contract_name": contract_name, "limit": int(limit)}
        else:
            query = """
            MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:Clause)
            OPTIONAL MATCH (cl)-[:HAS_TYPE]->(ct:ClauseType)
            RETURN c.name AS contract_name,
                   cl.id AS clause_id,
                   cl.idx AS clause_idx,
                   cl.number AS clause_number,
                   cl.text AS clause_text,
                   collect(DISTINCT ct.name) AS clause_types
            ORDER BY c.name ASC, cl.idx ASC
            LIMIT $limit
            """
            params = {"limit": int(limit)}
        return db.run_query(query, params)
    finally:
        db.close()


def _get_graph_source_marker(contract_name=None):
    db = Neo4jHandler()
    try:
        if contract_name:
            query = """
            MATCH (c:Contract {name: $contract_name})
            RETURN toString(c.updated_at) AS marker
            """
            rows = db.run_query(query, {"contract_name": contract_name})
            if rows:
                return rows[0].get("marker")
            return None

        query = """
        MATCH (c:Contract)
        RETURN toString(max(c.updated_at)) AS marker
        """
        rows = db.run_query(query)
        if rows:
            return rows[0].get("marker")
        return None
    finally:
        db.close()


def _is_index_stale(contract_name=None):
    state = get_index_state(contract_name=contract_name)
    if not state.get("exists"):
        return True
    graph_marker = _get_graph_source_marker(contract_name=contract_name)
    index_marker = state.get("source_marker")
    if not graph_marker:
        return False
    return str(index_marker or "") != str(graph_marker)


def refresh_semantic_index(contract_name=None, limit=3000):
    corpus = _fetch_clause_corpus(contract_name=contract_name, limit=limit)
    graph_marker = _get_graph_source_marker(contract_name=contract_name)
    built = build_clause_index(corpus, contract_name=contract_name) if corpus else False
    marker_updated = set_index_source_marker(contract_name=contract_name, source_marker=graph_marker) if built else False
    return {
        "contract_scope": contract_name,
        "corpus_count": len(corpus),
        "index_built": built,
        "index_marker_updated": marker_updated,
        "graph_source_marker": graph_marker,
    }


def get_clauses_by_type(clause_type, contract_name=None, top_k=10):
    db = Neo4jHandler()
    try:
        if contract_name:
            query = """
            MATCH (c:Contract {name: $contract_name})-[:HAS_CLAUSE]->(cl:Clause)-[:HAS_TYPE]->(ct:ClauseType {name: $clause_type})
            RETURN c.name AS contract_name,
                   cl.id AS clause_id,
                   cl.idx AS clause_idx,
                   cl.number AS clause_number,
                   cl.text AS clause_text,
                   ct.name AS clause_type
            ORDER BY cl.idx ASC
            LIMIT $top_k
            """
            params = {
                "contract_name": contract_name,
                "clause_type": str(clause_type).strip().lower(),
                "top_k": int(top_k),
            }
        else:
            query = """
            MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:Clause)-[:HAS_TYPE]->(ct:ClauseType {name: $clause_type})
            RETURN c.name AS contract_name,
                   cl.id AS clause_id,
                   cl.idx AS clause_idx,
                   cl.number AS clause_number,
                   cl.text AS clause_text,
                   ct.name AS clause_type
            ORDER BY c.name ASC, cl.idx ASC
            LIMIT $top_k
            """
            params = {
                "clause_type": str(clause_type).strip().lower(),
                "top_k": int(top_k),
            }
        return db.run_query(query, params)
    finally:
        db.close()


def get_clauses_by_types(clause_types, contract_name=None, total_k=10):
    clause_types = [str(x).strip().lower() for x in clause_types if str(x).strip()]
    if not clause_types:
        return []

    per_type = max(2, int(total_k) // len(clause_types))
    out = []
    seen = set()
    for ctype in clause_types:
        rows = get_clauses_by_type(ctype, contract_name=contract_name, top_k=per_type)
        for row in rows:
            key = row.get("clause_id") or f"{row.get('contract_name')}::{row.get('clause_idx')}"
            if key in seen:
                continue
            seen.add(key)
            row2 = dict(row)
            row2["matched_clause_type"] = ctype
            out.append(row2)
    return out[: max(int(total_k), 1)]


def _merge_results(typed_results, semantic_results, top_k=8, classification=None):
    merged = {}
    classification = classification or {}
    predicted_types = set(classification.get("candidate_types") or [])
    confidence = float(classification.get("confidence") or 0.0)

    for row in typed_results:
        key = row.get("clause_id") or f"{row.get('contract_name')}::{row.get('clause_idx')}"
        new_row = dict(row)
        new_row["source"] = "typed"
        merged[key] = new_row

    for row in semantic_results:
        key = row.get("clause_id") or f"{row.get('contract_name')}::{row.get('clause_idx')}"
        if key in merged:
            merged[key]["source"] = "hybrid"
            if "semantic_score" in row:
                merged[key]["semantic_score"] = row.get("semantic_score")
        else:
            new_row = dict(row)
            new_row["source"] = "semantic"
            merged[key] = new_row

    def rank_key(row):
        source = row.get("source")
        semantic = float(row.get("semantic_score") or 0.0)
        clause_type = str(row.get("clause_type") or row.get("matched_clause_type") or "").lower()
        type_bonus = 0.0
        if predicted_types and clause_type in predicted_types:
            type_bonus = 0.12 if confidence >= 0.7 else 0.06
        source_bonus = 0.10 if source in ("typed", "hybrid") else 0.0
        idx = row.get("clause_idx") if row.get("clause_idx") is not None else 10**9
        return -(semantic + type_bonus + source_bonus), idx

    ranked = sorted(merged.values(), key=rank_key)
    return ranked[: max(int(top_k), 1)]


def _dedupe_clause_rows(rows):
    deduped = {}
    for row in rows:
        key = row.get("clause_id") or f"{row.get('contract_name')}::{row.get('clause_idx')}"
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())


def _collect_present_clause_types(rows):
    present = set()
    for row in rows:
        ctype = row.get("clause_type")
        if ctype:
            present.add(str(ctype).strip().lower())
        ctype2 = row.get("matched_clause_type")
        if ctype2:
            present.add(str(ctype2).strip().lower())
    return present


def _run_missing_clause_detection(primary_type, candidate_types, clauses):
    seed_types = [t for t in [primary_type] + list(candidate_types or []) if t]
    expected = set()
    for t in seed_types:
        expected.update(EXPECTED_GOVERNING_TYPES.get(t, (t,)))
    if not expected:
        return {
            "enabled": False,
            "question_types": seed_types,
            "expected_types": [],
            "present_types": [],
            "missing_types": [],
            "is_complete": True,
        }

    present = _collect_present_clause_types(clauses)
    missing = sorted(t for t in expected if t not in present)
    return {
        "enabled": True,
        "question_types": seed_types,
        "expected_types": sorted(expected),
        "present_types": sorted(present),
        "missing_types": missing,
        "is_complete": len(missing) == 0,
    }


def expand_with_graph_context(seed_clauses, contract_name=None, clause_type=None, expansion_k=4, neighbor_window=1):
    if not seed_clauses or expansion_k <= 0:
        return []

    seed_ids = [row.get("clause_id") for row in seed_clauses if row.get("clause_id")]
    if not seed_ids:
        return []

    db = Neo4jHandler()
    try:
        params = {
            "seed_ids": seed_ids,
            "neighbor_window": int(max(neighbor_window, 1)),
            "limit": int(expansion_k),
            "contract_name": contract_name,
        }

        neighbor_filter = "WHERE c.name = $contract_name " if contract_name else "WHERE 1=1 "

        # Adaptive hop budget:
        # - small seed set and narrow expansion => keep latency lower with 2 hops
        # - broader seed set / larger expansion => allow 3 hops for deeper context
        max_hops = 3 if len(seed_ids) >= 3 or int(expansion_k) >= 5 else 2

        # Hop-1: immediate clause neighbors by index distance.
        query_neighbors_h1 = f"""
        UNWIND $seed_ids AS sid
        MATCH (seed:Clause {{id: sid}})<-[:HAS_CLAUSE]-(c:Contract)-[:HAS_CLAUSE]->(cl:Clause)
        {neighbor_filter}
        AND cl.id <> sid
        AND abs(coalesce(cl.idx, 0) - coalesce(seed.idx, 0)) <= $neighbor_window
        OPTIONAL MATCH (cl)-[:HAS_TYPE]->(ct:ClauseType)
        RETURN c.name AS contract_name,
               cl.id AS clause_id,
               cl.idx AS clause_idx,
               cl.number AS clause_number,
               cl.text AS clause_text,
               head(collect(DISTINCT ct.name)) AS clause_type,
               1 AS graph_hops,
               min(abs(coalesce(cl.idx, 0) - coalesce(seed.idx, 0))) AS graph_distance
        ORDER BY graph_distance ASC, cl.idx ASC
        LIMIT $limit
        """
        neighbors_h1 = db.run_query(query_neighbors_h1, params)

        # Hop-2/3 over concept graph:
        # Clause -> concept -> Clause (hop 2)
        # Clause -> concept -> Clause -> concept -> Clause (hop 3)
        query_concept_h2 = f"""
        UNWIND $seed_ids AS sid
        MATCH (seed:Clause {{id: sid}})<-[:HAS_CLAUSE]-(c:Contract)
        {neighbor_filter}
        MATCH (seed)-[:HAS_OBLIGATION|HAS_CONDITION|HAS_REMEDY|HAS_AMOUNT|HAS_DEADLINE]->(concept)
        MATCH (cl:Clause)-[:HAS_OBLIGATION|HAS_CONDITION|HAS_REMEDY|HAS_AMOUNT|HAS_DEADLINE]->(concept)
        WHERE cl.id <> sid
        OPTIONAL MATCH (cl)-[:HAS_TYPE]->(ct:ClauseType)
        RETURN c.name AS contract_name,
               cl.id AS clause_id,
               cl.idx AS clause_idx,
               cl.number AS clause_number,
               cl.text AS clause_text,
               head(collect(DISTINCT ct.name)) AS clause_type,
               2 AS graph_hops,
               2 AS graph_distance
        ORDER BY cl.idx ASC
        LIMIT $limit
        """
        concept_h2 = db.run_query(query_concept_h2, params)

        concept_h3 = []
        if max_hops >= 3:
            query_concept_h3 = f"""
            UNWIND $seed_ids AS sid
            MATCH (seed:Clause {{id: sid}})<-[:HAS_CLAUSE]-(c:Contract)
            {neighbor_filter}
            MATCH (seed)-[:HAS_OBLIGATION|HAS_CONDITION|HAS_REMEDY|HAS_AMOUNT|HAS_DEADLINE]->(concept1)
            MATCH (mid:Clause)-[:HAS_OBLIGATION|HAS_CONDITION|HAS_REMEDY|HAS_AMOUNT|HAS_DEADLINE]->(concept1)
            WHERE mid.id <> sid
            MATCH (mid)-[:HAS_OBLIGATION|HAS_CONDITION|HAS_REMEDY|HAS_AMOUNT|HAS_DEADLINE]->(concept2)
            MATCH (cl:Clause)-[:HAS_OBLIGATION|HAS_CONDITION|HAS_REMEDY|HAS_AMOUNT|HAS_DEADLINE]->(concept2)
            WHERE cl.id <> sid AND cl.id <> mid.id
            OPTIONAL MATCH (cl)-[:HAS_TYPE]->(ct:ClauseType)
            RETURN c.name AS contract_name,
                   cl.id AS clause_id,
                   cl.idx AS clause_idx,
                   cl.number AS clause_number,
                   cl.text AS clause_text,
                   head(collect(DISTINCT ct.name)) AS clause_type,
                   3 AS graph_hops,
                   3 AS graph_distance
            ORDER BY cl.idx ASC
            LIMIT $limit
            """
            concept_h3 = db.run_query(query_concept_h3, params)

        same_type = []
        if clause_type:
            params_type = {
                "seed_ids": seed_ids,
                "clause_type": str(clause_type).strip().lower(),
                "limit": int(expansion_k),
                "contract_name": contract_name,
            }
            contract_filter = "AND c.name = $contract_name " if contract_name else ""
            query_same_type = f"""
            MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:Clause)-[:HAS_TYPE]->(ct:ClauseType {{name: $clause_type}})
            WHERE NOT cl.id IN $seed_ids
            {contract_filter}
            RETURN c.name AS contract_name,
                   cl.id AS clause_id,
                   cl.idx AS clause_idx,
                   cl.number AS clause_number,
                   cl.text AS clause_text,
                   ct.name AS clause_type,
                   1 AS graph_hops,
                   null AS graph_distance
            ORDER BY cl.idx ASC
            LIMIT $limit
            """
            same_type = db.run_query(query_same_type, params_type)
    finally:
        db.close()

    expanded = _dedupe_clause_rows(neighbors_h1 + concept_h2 + concept_h3 + same_type)

    # Apply hop-based decay and keep top rows by adjusted graph relevance.
    # Lower hop count gets higher priority.
    for row in expanded:
        row["source"] = "graph_expansion"
        hops = int(row.get("graph_hops") or 2)
        if hops <= 1:
            decay = 1.0
        elif hops == 2:
            decay = 0.82
        else:
            decay = 0.68
        row["graph_hop_decay"] = float(decay)

    expanded = sorted(
        expanded,
        key=lambda r: (
            -(r.get("graph_hop_decay") or 0.0),
            r.get("graph_hops") if r.get("graph_hops") is not None else 9,
            r.get("clause_idx") if r.get("clause_idx") is not None else 10**9,
        ),
    )
    return expanded[: int(expansion_k)]


def ask_question(question, contract_name=None, retrieval_mode="hybrid", top_k=8, chat_history=None):
    context_text = _build_conversation_context(chat_history)
    retrieval_text = question if not context_text else f"{question}\n\nConversation context:\n{context_text}"

    classification = classify_question(question, chat_history=chat_history)
    primary_type = classification.get("primary_type")
    candidate_types = classification.get("candidate_types")
    confidence = float(classification.get("confidence") or 0.0)

    if confidence >= 0.75:
        typed_k = max(2, int(top_k))
        semantic_k = max(2, int(top_k) // 2)
        mode = "typed_priority"
    elif confidence >= 0.4:
        typed_k = max(2, int(top_k) // 2)
        semantic_k = max(2, int(top_k))
        mode = "hybrid_balanced"
    else:
        typed_k = 0
        semantic_k = max(2, int(top_k) + 2)
        mode = "semantic_priority"

    typed_results = []
    if candidate_types and typed_k > 0:
        typed_results = get_clauses_by_types(candidate_types, contract_name=contract_name, total_k=typed_k)

    corpus_count = None
    index_was_stale = _is_index_stale(contract_name=contract_name)
    if not has_clause_index(contract_name=contract_name) or index_was_stale:
        refresh_result = refresh_semantic_index(contract_name=contract_name)
        corpus_count = refresh_result.get("corpus_count")
    else:
        refresh_result = None

    semantic_results = search_similar_clauses(
        question=retrieval_text,
        top_k=semantic_k,
        contract_name=contract_name,
    )

    if corpus_count is None:
        _, metadata = load_clause_index(contract_name=contract_name)
        corpus_count = len(metadata) if metadata else 0

    base_limit = max(int(top_k), 1)
    clauses = _merge_results(
        typed_results,
        semantic_results,
        top_k=base_limit,
        classification=classification,
    )

    expansion_k = max(2, int(top_k) // 2)
    if confidence >= 0.75:
        expansion_k += 1

    expanded = expand_with_graph_context(
        seed_clauses=clauses,
        contract_name=contract_name,
        clause_type=primary_type,
        expansion_k=expansion_k,
        neighbor_window=1,
    )
    clauses = _dedupe_clause_rows(clauses + expanded)

    missing_check = _run_missing_clause_detection(primary_type, candidate_types, clauses)
    supplemental_rows = []
    if missing_check.get("enabled") and missing_check.get("missing_types"):
        supplemental_k = max(1, int(top_k) // 3)
        for missing_type in missing_check.get("missing_types", [])[:2]:
            supplemental_rows.extend(
                get_clauses_by_type(
                    missing_type,
                    contract_name=contract_name,
                    top_k=supplemental_k,
                )
            )
        if supplemental_rows:
            for row in supplemental_rows:
                row["source"] = "missing_clause_recovery"
            clauses = _dedupe_clause_rows(clauses + supplemental_rows)
            missing_check = _run_missing_clause_detection(primary_type, candidate_types, clauses)

    clauses = clauses[: max(base_limit + expansion_k + len(supplemental_rows), 1)]

    embed_debug = get_last_debug()

    return {
        "question": question,
        "retrieval_question": retrieval_text,
        "detected_type": primary_type,
        "detected_types": candidate_types,
        "type_confidence": confidence,
        "type_scores": classification.get("scores") or {},
        "retrieval_mode": mode,
        "contract_scope": contract_name,
        "embedding_provider_used": embed_debug.get("provider"),
        "embedding_model_used": embed_debug.get("model"),
        "embedding_dim": embed_debug.get("dim"),
        "embedding_last_error": embed_debug.get("last_error"),
        "corpus_count": corpus_count,
        "index_refreshed": bool(refresh_result),
        "index_was_stale": bool(index_was_stale),
        "typed_count": len(typed_results),
        "semantic_count": len(semantic_results),
        "graph_expanded_count": len(expanded),
        "missing_clause_recovery_count": len(supplemental_rows),
        "missing_clause_check": missing_check,
        "clauses": clauses,
    }


def get_explainability_paths(clauses, contract_name=None, per_clause_limit=4):
    clause_ids = [row.get("clause_id") for row in clauses if row.get("clause_id")]
    if not clause_ids:
        return []

    db = Neo4jHandler()
    try:
        params = {
            "clause_ids": clause_ids,
            "limit": int(max(per_clause_limit, 1)),
            "contract_name": contract_name,
        }
        contract_filter = "AND c.name = $contract_name" if contract_name else ""

        query = f"""
        UNWIND $clause_ids AS cid
        MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:Clause {{id: cid}})
        WHERE 1=1 {contract_filter}
        OPTIONAL MATCH (cl)-[r1]->(n1)
        WITH c, cl, r1, n1
        ORDER BY type(r1)
        WITH c, cl, collect({{
            path_type: 'one_hop',
            rel1: type(r1),
            node1_labels: labels(n1),
            node1_text: coalesce(n1.text, coalesce(n1.name, ''))
        }})[..$limit] AS one_hop_rows
        UNWIND one_hop_rows AS row
        RETURN c.name AS contract_name,
               cl.id AS clause_id,
               cl.number AS clause_number,
               cl.idx AS clause_idx,
               row.path_type AS path_type,
               row.rel1 AS rel1,
               row.node1_labels AS node1_labels,
               row.node1_text AS node1_text,
               null AS rel2,
               null AS node2_labels,
               null AS node2_text
        UNION
        UNWIND $clause_ids AS cid
        MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:Clause {{id: cid}})
        WHERE 1=1 {contract_filter}
        OPTIONAL MATCH (cl)-[r1]->(n1)-[r2]->(n2)
        WITH c, cl, r1, n1, r2, n2
        ORDER BY type(r1), type(r2)
        WITH c, cl, collect({{
            path_type: 'two_hop',
            rel1: type(r1),
            node1_labels: labels(n1),
            node1_text: coalesce(n1.text, coalesce(n1.name, '')),
            rel2: type(r2),
            node2_labels: labels(n2),
            node2_text: coalesce(n2.text, coalesce(n2.name, ''))
        }})[..$limit] AS two_hop_rows
        UNWIND two_hop_rows AS row
        RETURN c.name AS contract_name,
               cl.id AS clause_id,
               cl.number AS clause_number,
               cl.idx AS clause_idx,
               row.path_type AS path_type,
               row.rel1 AS rel1,
               row.node1_labels AS node1_labels,
               row.node1_text AS node1_text,
               row.rel2 AS rel2,
               row.node2_labels AS node2_labels,
               row.node2_text AS node2_text
        """
        rows = db.run_query(query, params)
    finally:
        db.close()

    cleaned = []
    for row in rows:
        if not row.get("rel1"):
            continue
        cleaned.append(row)
    return cleaned
