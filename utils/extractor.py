import re
from utils.llm_extractor import extract_parties_and_clause_types

def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\n\r,.;:-")


def _looks_like_party(candidate: str) -> bool:
    cleaned = _normalize_spaces(candidate)
    if not cleaned or len(cleaned) < 2:
        return False

    lower = cleaned.lower()
    blocked_phrases = (
        "whereas",
        "parties hereto",
        "between the parties",
        "registered in the office",
        "resolved amicably",
    )
    if any(phrase in lower for phrase in blocked_phrases):
        return False

    # Avoid article-like fragments: "a Lease", "an exchange is that".
    if re.match(r"^(a|an)\s+[a-z]", lower):
        return False

    # Keep candidates that look like names or organizations.
    if re.search(r"[A-Z]", cleaned):
        return True

    org_tokens = ("llp", "llc", "ltd", "limited", "inc", "corp", "company", "co.")
    return any(token in lower for token in org_tokens)


def _dedupe_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _extract_clause_knowledge(clause_text: str):
    obligations = []
    deadlines = []
    amounts = []
    conditions = []
    remedies = []

    text = clause_text.strip()
    if not text:
        return obligations, deadlines, amounts, conditions, remedies

    # Obligation-like statements.
    obligation_patterns = [
        r"(?i)\b(?:shall|must|agrees?\s+to|is\s+required\s+to)\b[^.;\n]*",
    ]
    for pattern in obligation_patterns:
        for match in re.findall(pattern, text):
            item = _normalize_spaces(match)
            if len(item) >= 12:
                obligations.append(item)

    # Deadlines and temporal requirements.
    deadline_patterns = [
        r"(?i)\bwithin\s+\d+\s+(?:business\s+)?days?\b[^.;\n]*",
        r"(?i)\bno\s+later\s+than\b[^.;\n]*",
        r"(?i)\bon\s+or\s+before\b[^.;\n]*",
        r"(?i)\bby\s+(?:\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{2,4})\b[^.;\n]*",
    ]
    for pattern in deadline_patterns:
        for match in re.findall(pattern, text):
            item = _normalize_spaces(match)
            if len(item) >= 8:
                deadlines.append(item)

    # Monetary amounts.
    amount_patterns = [
        r"(?i)\b(?:USD|INR|EUR|GBP|Rs\.?|₹|\$)\s?\d[\d,]*(?:\.\d+)?\b",
        r"(?i)\b\d[\d,]*(?:\.\d+)?\s?(?:USD|INR|EUR|GBP)\b",
        r"(?i)\b\d+(?:\.\d+)?\s?%\b",
    ]
    for pattern in amount_patterns:
        for match in re.findall(pattern, text):
            item = _normalize_spaces(match)
            if item:
                amounts.append(item)

    # Conditions / triggers.
    condition_patterns = [
        r"(?i)\bif\b[^.;\n]*",
        r"(?i)\bunless\b[^.;\n]*",
        r"(?i)\bexcept\b[^.;\n]*",
        r"(?i)\bnotwithstanding\b[^.;\n]*",
        r"(?i)\bprovided\s+that\b[^.;\n]*",
        r"(?i)\bsubject\s+to\b[^.;\n]*",
        r"(?i)\bin\s+the\s+event\s+that\b[^.;\n]*",
    ]
    for pattern in condition_patterns:
        for match in re.findall(pattern, text):
            item = _normalize_spaces(match)
            if len(item) >= 8:
                conditions.append(item)

    # Remedies / enforcement outcomes.
    remedy_patterns = [
        r"(?i)\bmay\s+terminate\b[^.;\n]*",
        r"(?i)\bentitled\s+to\b[^.;\n]*",
        r"(?i)\bshall\s+be\s+liable\b[^.;\n]*",
        r"(?i)\binjunctive\s+relief\b[^.;\n]*",
        r"(?i)\bdamages\b[^.;\n]*",
        r"(?i)\bspecific\s+performance\b[^.;\n]*",
    ]
    for pattern in remedy_patterns:
        for match in re.findall(pattern, text):
            item = _normalize_spaces(match)
            if len(item) >= 8:
                remedies.append(item)

    return (
        _dedupe_preserve_order(obligations),
        _dedupe_preserve_order(deadlines),
        _dedupe_preserve_order(amounts),
        _dedupe_preserve_order(conditions),
        _dedupe_preserve_order(remedies),
    )


def _extract_rule_relations_for_clause(clause_idx, clause_text, obligations, deadlines, amounts, conditions, remedies):
    def _bounded_pairs(left_items, right_items, limit=6):
        pairs = []
        for left in left_items:
            for right in right_items:
                pairs.append((left, right))
                if len(pairs) >= limit:
                    return pairs
        return pairs

    relations = []
    text_lower = clause_text.lower()

    # Clause obliges obligations.
    for obligation in obligations[:6]:
        relations.append(
            {
                "clause_idx": clause_idx,
                "source_type": "clause",
                "source_text": "",
                "relation": "OBLIGES",
                "target_type": "obligation",
                "target_text": obligation,
            }
        )

    # Obligation depends on condition.
    if obligations and conditions:
        for obligation, condition in _bounded_pairs(obligations, conditions, limit=6):
            relations.append(
                {
                    "clause_idx": clause_idx,
                    "source_type": "obligation",
                    "source_text": obligation,
                    "relation": "DEPENDS_ON",
                    "target_type": "condition",
                    "target_text": condition,
                }
            )

    # Condition triggers remedy.
    if conditions and remedies:
        for condition, remedy in _bounded_pairs(conditions, remedies, limit=6):
            relations.append(
                {
                    "clause_idx": clause_idx,
                    "source_type": "condition",
                    "source_text": condition,
                    "relation": "TRIGGERS",
                    "target_type": "remedy",
                    "target_text": remedy,
                }
            )

    # Exception conditions to obligations.
    exception_tokens = ("except", "unless", "notwithstanding", "save that")
    exception_conditions = [c for c in conditions if any(tok in c.lower() for tok in exception_tokens)]
    if obligations and exception_conditions:
        for condition, obligation in _bounded_pairs(exception_conditions, obligations, limit=4):
            relations.append(
                {
                    "clause_idx": clause_idx,
                    "source_type": "condition",
                    "source_text": condition,
                    "relation": "EXCEPTION_TO",
                    "target_type": "obligation",
                    "target_text": obligation,
                }
            )

    # Obligations that survive termination.
    if obligations and any(token in text_lower for token in ("survive", "survives", "survival", "termination")):
        for obligation in obligations[:3]:
            relations.append(
                {
                    "clause_idx": clause_idx,
                    "source_type": "obligation",
                    "source_text": obligation,
                    "relation": "SURVIVES",
                    "target_type": "clause",
                    "target_text": "",
            }
        )

    # Deduplicate per clause.
    seen = set()
    deduped = []
    for row in relations:
        key = (
            row["clause_idx"],
            row["source_type"],
            row["source_text"],
            row["relation"],
            row["target_type"],
            row["target_text"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped[:20]


def extract_basic_entities(text, clauses, use_llm=True):
    clause_texts = []
    for clause in clauses:
        if isinstance(clause, dict):
            clause_text = str(clause.get("text", "")).strip()
        else:
            clause_text = str(clause).strip()
        if clause_text:
            clause_texts.append(clause_text)

    extracted = {
        "parties": [],
        "jurisdiction": [],
        "confidentiality_clauses": [],
        "liability_clauses": [],
        "termination_clauses": [],
        "clause_types": {},
        "obligations": [],
        "deadlines": [],
        "amounts": [],
        "conditions": [],
        "remedies": [],
        "rule_relations": [],
        "all_clauses": clauses,
    }

    party_patterns = [
        r"(?is)\bthis\s+agreement\s+is\s+(?:made\s+)?between\s+(.+?)\s+and\s+(.+?)(?:[\.,;\n]|$)",
        r"(?is)\bby\s+and\s+between\s+(.+?)\s+and\s+(.+?)(?:[\.,;\n]|$)",
        r"(?is)\bbetween\s+(.+?)\s+and\s+(.+?)(?:[\.,;\n]|$)",
    ]

    for pattern in party_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            left = _normalize_spaces(match[0])
            right = _normalize_spaces(match[1])
            if _looks_like_party(left):
                extracted["parties"].append(left)
            if _looks_like_party(right):
                extracted["parties"].append(right)

    # Deterministic legal header patterns: "Party A: <Entity>".
    party_header_matches = re.findall(
        r"(?im)^\s*Party\s+[A-Z0-9]+\s*[:\-]\s*(.+?)\s*$",
        text,
    )
    for candidate in party_header_matches:
        entity = _normalize_spaces(candidate)
        if _looks_like_party(entity):
            extracted["parties"].append(entity)

    # Pick up explicitly defined party labels, e.g. "ABC Ltd (the "Lessor")".
    alias_matches = re.findall(r'(?i)\(the\s+"?([A-Za-z][A-Za-z\s]{1,40})"?\)', text)
    for alias in alias_matches:
        alias_clean = _normalize_spaces(alias)
        if _looks_like_party(alias_clean):
            extracted["parties"].append(alias_clean)

    jurisdiction_matches = re.findall(
        r"(?i)governed\s+by\s+the\s+laws?\s+of\s+([^.;\n]+)",
        text,
    )
    for match in jurisdiction_matches:
        value = _normalize_spaces(match)
        # Normalize common truncation from \"the People's Republic of China\".
        if re.search(r"people'?s\s+republic\s+of\s+china", value, re.IGNORECASE):
            extracted["jurisdiction"].append("People's Republic of China")
        else:
            extracted["jurisdiction"].append(value)

    for idx, clause in enumerate(clause_texts, start=1):
        clause_lower = clause.lower()
        labels = []

        if "confidential" in clause_lower:
            extracted["confidentiality_clauses"].append(clause)
            labels.append("confidentiality")

        if "liability" in clause_lower or "indemnity" in clause_lower:
            extracted["liability_clauses"].append(clause)
            labels.append("liability")

        if "terminate" in clause_lower or "termination" in clause_lower:
            extracted["termination_clauses"].append(clause)
            labels.append("termination")

        if labels:
            extracted["clause_types"][str(idx)] = sorted(set(labels))

        obligations, deadlines, amounts, conditions, remedies = _extract_clause_knowledge(clause)
        for value in obligations:
            extracted["obligations"].append({"clause_idx": idx, "text": value})
        for value in deadlines:
            extracted["deadlines"].append({"clause_idx": idx, "text": value})
        for value in amounts:
            extracted["amounts"].append({"clause_idx": idx, "text": value})
        for value in conditions:
            extracted["conditions"].append({"clause_idx": idx, "text": value})
        for value in remedies:
            extracted["remedies"].append({"clause_idx": idx, "text": value})

        extracted["rule_relations"].extend(
            _extract_rule_relations_for_clause(
                clause_idx=idx,
                clause_text=clause,
                obligations=obligations,
                deadlines=deadlines,
                amounts=amounts,
                conditions=conditions,
                remedies=remedies,
            )
        )

    if use_llm:
        llm_result = extract_parties_and_clause_types(text, clause_texts)
        for party in llm_result.get("parties", []):
            party_clean = _normalize_spaces(party)
            if _looks_like_party(party_clean):
                extracted["parties"].append(party_clean)

        llm_clause_types = llm_result.get("clause_types", {})
        for clause_id, labels in llm_clause_types.items():
            if not isinstance(clause_id, int):
                continue
            if clause_id < 1 or clause_id > len(clause_texts):
                continue

            clause_text = clause_texts[clause_id - 1]
            normalized_labels = sorted(set(str(label).strip().lower() for label in labels if str(label).strip()))
            if not normalized_labels:
                continue

            existing = extracted["clause_types"].get(str(clause_id), [])
            extracted["clause_types"][str(clause_id)] = sorted(set(existing + normalized_labels))

            if "confidentiality" in normalized_labels:
                extracted["confidentiality_clauses"].append(clause_text)
            if "liability" in normalized_labels or "indemnity" in normalized_labels:
                extracted["liability_clauses"].append(clause_text)
            if "termination" in normalized_labels:
                extracted["termination_clauses"].append(clause_text)

    extracted["parties"] = _dedupe_preserve_order(extracted["parties"])
    extracted["jurisdiction"] = _dedupe_preserve_order(extracted["jurisdiction"])
    extracted["confidentiality_clauses"] = _dedupe_preserve_order(extracted["confidentiality_clauses"])
    extracted["liability_clauses"] = _dedupe_preserve_order(extracted["liability_clauses"])
    extracted["termination_clauses"] = _dedupe_preserve_order(extracted["termination_clauses"])

    # Dedupe structured rows while preserving order.
    for key in ("obligations", "deadlines", "amounts", "conditions", "remedies"):
        seen = set()
        deduped = []
        for row in extracted[key]:
            marker = (row.get("clause_idx"), row.get("text"))
            if marker not in seen:
                seen.add(marker)
                deduped.append(row)
        extracted[key] = deduped

    # Dedupe relations globally.
    seen_rel = set()
    deduped_rel = []
    for row in extracted["rule_relations"]:
        marker = (
            row.get("clause_idx"),
            row.get("source_type"),
            row.get("source_text"),
            row.get("relation"),
            row.get("target_type"),
            row.get("target_text"),
        )
        if marker not in seen_rel:
            seen_rel.add(marker)
            deduped_rel.append(row)
    extracted["rule_relations"] = deduped_rel

    return extracted
