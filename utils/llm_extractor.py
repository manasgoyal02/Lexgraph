import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SUPPORTED_CLAUSE_TYPES = {
    "confidentiality",
    "liability",
    "termination",
    "indemnity",
    "governing_law",
    "payment",
    "dispute_resolution",
    "force_majeure",
    "warranty",
    "assignment",
    "ip",
    "other",
}


def _build_client():
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME")
    if not api_key or not model_name:
        return None, None

    base_url = os.getenv("OPENAI_BASE_URL")
    # OpenRouter keys usually start with sk-or-; use OpenRouter base URL by default.
    if not base_url and api_key.startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url), model_name
    return OpenAI(api_key=api_key), model_name


def _parse_json_payload(content: str) -> dict:
    if not content:
        return {}

    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def extract_parties_and_clause_types(text: str, clauses: list[str]) -> dict:
    client, model_name = _build_client()
    if not client or not model_name:
        return {"parties": [], "clause_types": {}}

    if not clauses:
        return {"parties": [], "clause_types": {}}

    clause_rows = [{"clause_id": idx + 1, "text": clause[:1200]} for idx, clause in enumerate(clauses)]
    payload = {
        "document_intro_excerpt": text[:3000],
        "clauses": clause_rows[:120],  # safety bound for token size
    }

    system_prompt = (
        "You are a legal information extractor. "
        "Return only strict JSON with keys: parties, clause_types. "
        "parties must be an array of legal entity names. "
        "clause_types must be an array of objects: "
        '{"clause_id": <int>, "types": [<string>, ...]}. '
        "Allowed types: confidentiality, liability, termination, indemnity, governing_law, "
        "payment, dispute_resolution, force_majeure, warranty, assignment, ip, other. "
        "Do not include explanations."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.5,
            timeout=20,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        content = response.choices[0].message.content if response.choices else "{}"
        data = _parse_json_payload(content)
    except Exception:
        return {"parties": [], "clause_types": {}}

    parties_raw = data.get("parties", [])
    parties = [str(p).strip() for p in parties_raw if isinstance(p, str) and str(p).strip()]

    clause_types_map: dict[int, list[str]] = {}
    for row in data.get("clause_types", []):
        if not isinstance(row, dict):
            continue
        clause_id = row.get("clause_id")
        if not isinstance(clause_id, int):
            continue
        labels = []
        for label in row.get("types", []):
            if not isinstance(label, str):
                continue
            norm = label.strip().lower().replace(" ", "_")
            if norm in SUPPORTED_CLAUSE_TYPES:
                labels.append(norm)
        if labels:
            clause_types_map[clause_id] = sorted(set(labels))

    return {"parties": parties, "clause_types": clause_types_map}
