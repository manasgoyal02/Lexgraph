from utils.neo4j_handler import Neo4jHandler
import hashlib


def _concept_id(contract_name, clause_id, kind, text):
    fingerprint = hashlib.md5(f"{contract_name}|{clause_id}|{kind}|{text}".encode("utf-8")).hexdigest()
    return f"{contract_name}::{kind}::{fingerprint}"

def _normalize_clauses(contract_name, clauses):
    normalized = []
    for idx, clause in enumerate(clauses, start=1):
        if isinstance(clause, dict):
            text = str(clause.get("text", "")).strip()
            number = str(clause.get("number", "")).strip()
        else:
            text = str(clause).strip()
            number = ""

        if not text:
            continue

        clause_id = f"{contract_name}::clause::{idx}"
        normalized.append(
            {
                "clause_id": clause_id,
                "idx": idx,
                "number": number,
                "text": text,
            }
        )
    return normalized


def create_contract_graph(extracted_data, contract_name):
    db = Neo4jHandler()
    try:
        db.run_write("CREATE CONSTRAINT contract_name_unique IF NOT EXISTS FOR (c:Contract) REQUIRE c.name IS UNIQUE")
        db.run_write("CREATE CONSTRAINT party_name_unique IF NOT EXISTS FOR (p:Party) REQUIRE p.name IS UNIQUE")
        db.run_write("CREATE CONSTRAINT jurisdiction_name_unique IF NOT EXISTS FOR (j:Jurisdiction) REQUIRE j.name IS UNIQUE")
        db.run_write("CREATE CONSTRAINT clause_id_unique IF NOT EXISTS FOR (cl:Clause) REQUIRE cl.id IS UNIQUE")
        db.run_write("CREATE CONSTRAINT clause_type_name_unique IF NOT EXISTS FOR (ct:ClauseType) REQUIRE ct.name IS UNIQUE")
        db.run_write("CREATE CONSTRAINT obligation_id_unique IF NOT EXISTS FOR (o:Obligation) REQUIRE o.id IS UNIQUE")
        db.run_write("CREATE CONSTRAINT deadline_id_unique IF NOT EXISTS FOR (d:Deadline) REQUIRE d.id IS UNIQUE")
        db.run_write("CREATE CONSTRAINT amount_id_unique IF NOT EXISTS FOR (a:Amount) REQUIRE a.id IS UNIQUE")
        db.run_write("CREATE CONSTRAINT condition_id_unique IF NOT EXISTS FOR (cn:Condition) REQUIRE cn.id IS UNIQUE")
        db.run_write("CREATE CONSTRAINT remedy_id_unique IF NOT EXISTS FOR (r:Remedy) REQUIRE r.id IS UNIQUE")

        parties = extracted_data.get("parties", [])
        jurisdictions = extracted_data.get("jurisdiction", [])
        clauses = _normalize_clauses(contract_name, extracted_data.get("all_clauses", []))
        clause_types = extracted_data.get("clause_types", {})
        obligations = extracted_data.get("obligations", [])
        deadlines = extracted_data.get("deadlines", [])
        amounts = extracted_data.get("amounts", [])
        conditions = extracted_data.get("conditions", [])
        remedies = extracted_data.get("remedies", [])
        rule_relations = extracted_data.get("rule_relations", [])
        clause_id_by_idx = {c["idx"]: c["clause_id"] for c in clauses}

        db.run_write(
            """
            MERGE (c:Contract {name: $contract_name})
            SET c.updated_at = datetime()
            """,
            {"contract_name": contract_name},
        )

        # Refresh contract-specific graph slice to avoid stale/noisy accumulation.
        db.run_write(
            """
            MATCH (c:Contract {name: $contract_name})
            OPTIONAL MATCH (c)<-[rs:SIGNED]-(:Party)
            DELETE rs
            WITH c
            OPTIONAL MATCH (c)-[rg:GOVERNED_BY]->(:Jurisdiction)
            DELETE rg
            WITH c
            OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
            DETACH DELETE cl
            """,
            {"contract_name": contract_name},
        )

        if parties:
            db.run_write(
                """
                MERGE (c:Contract {name: $contract_name})
                WITH c
                UNWIND $parties AS party_name
                MERGE (p:Party {name: party_name})
                MERGE (p)-[:SIGNED]->(c)
                """,
                {"contract_name": contract_name, "parties": parties},
            )

        if jurisdictions:
            db.run_write(
                """
                MERGE (c:Contract {name: $contract_name})
                WITH c
                UNWIND $jurisdictions AS jurisdiction_name
                MERGE (j:Jurisdiction {name: jurisdiction_name})
                MERGE (c)-[:GOVERNED_BY]->(j)
                """,
                {"contract_name": contract_name, "jurisdictions": jurisdictions},
            )

        if clauses:
            db.run_write(
                """
                MERGE (c:Contract {name: $contract_name})
                WITH c
                UNWIND $clauses AS row
                MERGE (cl:Clause {id: row.clause_id})
                SET cl.idx = row.idx,
                    cl.number = row.number,
                    cl.text = row.text
                MERGE (c)-[:HAS_CLAUSE]->(cl)
                """,
                {"contract_name": contract_name, "clauses": clauses},
            )

            # Preserve clause order in graph for better neighborhood traversal.
            db.run_write(
                """
                MATCH (c:Contract {name: $contract_name})-[:HAS_CLAUSE]->(cl:Clause)
                WITH cl ORDER BY cl.idx ASC
                WITH collect(cl) AS cls
                UNWIND range(0, size(cls)-2) AS i
                WITH cls[i] AS curr, cls[i+1] AS nxt
                MERGE (curr)-[:NEXT_CLAUSE]->(nxt)
                """,
                {"contract_name": contract_name},
            )

            clause_type_rows = []
            for clause in clauses:
                idx_key = str(clause["idx"])
                for label in clause_types.get(idx_key, []):
                    cleaned = str(label).strip().lower()
                    if cleaned:
                        clause_type_rows.append({"clause_id": clause["clause_id"], "type_name": cleaned})

            if clause_type_rows:
                db.run_write(
                    """
                    UNWIND $rows AS row
                    MATCH (cl:Clause {id: row.clause_id})
                    MERGE (ct:ClauseType {name: row.type_name})
                    MERGE (cl)-[:HAS_TYPE]->(ct)
                    """,
                    {"rows": clause_type_rows},
                )

            def _build_rows(items, kind):
                rows = []
                for item in items:
                    idx = item.get("clause_idx")
                    text = str(item.get("text", "")).strip()
                    clause_id = clause_id_by_idx.get(idx)
                    if not clause_id or not text:
                        continue
                    rows.append({"id": _concept_id(contract_name, clause_id, kind, text), "clause_id": clause_id, "text": text})
                return rows

            obligation_rows = _build_rows(obligations, "obligation")
            deadline_rows = _build_rows(deadlines, "deadline")
            amount_rows = _build_rows(amounts, "amount")
            condition_rows = _build_rows(conditions, "condition")
            remedy_rows = _build_rows(remedies, "remedy")

            if obligation_rows:
                db.run_write(
                    """
                    UNWIND $rows AS row
                    MATCH (cl:Clause {id: row.clause_id})
                    MERGE (o:Obligation {id: row.id})
                    SET o.text = row.text
                    MERGE (cl)-[:HAS_OBLIGATION]->(o)
                    """,
                    {"rows": obligation_rows},
                )

            if deadline_rows:
                db.run_write(
                    """
                    UNWIND $rows AS row
                    MATCH (cl:Clause {id: row.clause_id})
                    MERGE (d:Deadline {id: row.id})
                    SET d.text = row.text
                    MERGE (cl)-[:HAS_DEADLINE]->(d)
                    """,
                    {"rows": deadline_rows},
                )

            if amount_rows:
                db.run_write(
                    """
                    UNWIND $rows AS row
                    MATCH (cl:Clause {id: row.clause_id})
                    MERGE (a:Amount {id: row.id})
                    SET a.text = row.text
                    MERGE (cl)-[:HAS_AMOUNT]->(a)
                    """,
                    {"rows": amount_rows},
                )

            if condition_rows:
                db.run_write(
                    """
                    UNWIND $rows AS row
                    MATCH (cl:Clause {id: row.clause_id})
                    MERGE (cn:Condition {id: row.id})
                    SET cn.text = row.text
                    MERGE (cl)-[:HAS_CONDITION]->(cn)
                    """,
                    {"rows": condition_rows},
                )

            if remedy_rows:
                db.run_write(
                    """
                    UNWIND $rows AS row
                    MATCH (cl:Clause {id: row.clause_id})
                    MERGE (r:Remedy {id: row.id})
                    SET r.text = row.text
                    MERGE (cl)-[:HAS_REMEDY]->(r)
                    """,
                    {"rows": remedy_rows},
                )

            relation_rows = []
            for rel in rule_relations:
                clause_idx = rel.get("clause_idx")
                source_text = str(rel.get("source_text", "")).strip()
                target_text = str(rel.get("target_text", "")).strip()
                source_type = str(rel.get("source_type", "")).strip().lower()
                target_type = str(rel.get("target_type", "")).strip().lower()
                relation = str(rel.get("relation", "")).strip().upper()
                clause_id = clause_id_by_idx.get(clause_idx)

                if not clause_id or not source_text or not target_text or not source_type or not target_type or not relation:
                    # Allow clause-based relation endpoints where text may be empty.
                    if not clause_id or not source_type or not target_type or not relation:
                        continue

                if source_type == "clause":
                    source_id = clause_id
                else:
                    if not source_text:
                        continue
                    source_id = _concept_id(contract_name, clause_id, source_type, source_text)

                if target_type == "clause":
                    target_id = clause_id
                else:
                    if not target_text:
                        continue
                    target_id = _concept_id(contract_name, clause_id, target_type, target_text)

                relation_rows.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation": relation,
                    }
                )

            rel_queries = {
                "DEPENDS_ON": """
                    UNWIND $rows AS row
                    MATCH (s {id: row.source_id}), (t {id: row.target_id})
                    MERGE (s)-[:DEPENDS_ON]->(t)
                """,
                "TRIGGERS": """
                    UNWIND $rows AS row
                    MATCH (s {id: row.source_id}), (t {id: row.target_id})
                    MERGE (s)-[:TRIGGERS]->(t)
                """,
                "EXCEPTION_TO": """
                    UNWIND $rows AS row
                    MATCH (s {id: row.source_id}), (t {id: row.target_id})
                    MERGE (s)-[:EXCEPTION_TO]->(t)
                """,
                "SURVIVES": """
                    UNWIND $rows AS row
                    MATCH (s {id: row.source_id}), (t {id: row.target_id})
                    MERGE (s)-[:SURVIVES]->(t)
                """,
                "OBLIGES": """
                    UNWIND $rows AS row
                    MATCH (s {id: row.source_id}), (t {id: row.target_id})
                    MERGE (s)-[:OBLIGES]->(t)
                """,
            }

            if relation_rows:
                grouped = {}
                for row in relation_rows:
                    grouped.setdefault(row["relation"], []).append(row)
                for rel_name, rows in grouped.items():
                    query = rel_queries.get(rel_name)
                    if query and rows:
                        db.run_write(query, {"rows": rows})
    finally:
        db.close()
