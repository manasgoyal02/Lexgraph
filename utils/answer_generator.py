import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _build_client():
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME")
    if not api_key or not model_name:
        return None, None, None

    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and api_key.startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url), model_name, "openrouter_or_custom"
    return OpenAI(api_key=api_key), model_name, "openai_default"


def _format_citation(clause):
    clause_number = clause.get("clause_number") or clause.get("clause_idx") or "Unknown"
    return {
        "contract_name": clause.get("contract_name"),
        "clause_id": clause.get("clause_id"),
        "clause_number": clause_number,
        "clause_text": clause.get("clause_text", ""),
    }


def _build_chat_context(chat_history, max_turns=4):
    if not chat_history:
        return []
    turns = chat_history[-max_turns:]
    out = []
    for row in turns:
        role = str(row.get("role", "")).strip().lower()
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        out.append({"role": role if role in ("user", "assistant") else "user", "content": content})
    return out


def _fallback_answer(question, clause_type, clauses):
    if not clauses:
        return "No relevant clauses found for this query."

    key_points = []
    for clause in clauses[:4]:
        clause_number = clause.get("clause_number") or clause.get("clause_idx") or "Unknown"
        clause_text = str(clause.get("clause_text", "")).strip()
        sentence = clause_text.split(".")[0].strip() if clause_text else ""
        if not sentence:
            continue
        short = sentence[:180] + "..." if len(sentence) > 180 else sentence
        key_points.append(f"[Clause {clause_number}] {short}")

    if not key_points:
        return "Relevant context was found, but no readable clause text was available for summarization."

    heading = "Context-based summary (LLM unavailable):"
    return heading + "\n" + "\n".join(key_points)


def _clamp01(value):
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _compute_answer_confidence(query_result, clauses, llm_success):
    type_conf = _clamp01(query_result.get("type_confidence") or 0.0)
    typed_count = int(query_result.get("typed_count") or 0)
    semantic_scores = []
    for row in clauses[:6]:
        score = row.get("semantic_score")
        if score is None:
            continue
        semantic_scores.append(_clamp01(score))
    semantic_signal = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
    evidence_signal = _clamp01(len(clauses) / 6.0)
    typed_signal = _clamp01(typed_count / 3.0)

    base = (0.35 * type_conf) + (0.35 * semantic_signal) + (0.20 * evidence_signal) + (0.10 * typed_signal)

    missing_check = query_result.get("missing_clause_check") or {}
    if missing_check.get("enabled") and not missing_check.get("is_complete", True):
        missing_types = missing_check.get("missing_types") or []
        # Penalize confidence when expected governing clause families are missing.
        base -= min(0.22, 0.08 * len(missing_types))

    if not llm_success:
        base -= 0.08

    score = _clamp01(base)
    if score >= 0.72:
        label = "high"
    elif score >= 0.48:
        label = "medium"
    else:
        label = "low"

    return score, label


def _apply_confidence_policy(answer_text, question, confidence_label, query_result):
    text = str(answer_text or "").strip()
    if not text:
        return text

    if confidence_label == "high":
        return text

    if confidence_label == "medium":
        return "Based on the retrieved contract context:\n" + text

    # low confidence
    missing_check = query_result.get("missing_clause_check") or {}
    missing_types = missing_check.get("missing_types") or []
    if missing_types:
        missing_hint = ", ".join(missing_types[:3])
        return (
            "Low-confidence answer: key governing context may be missing "
            f"(likely missing clause types: {missing_hint}).\n"
            + text
            + f"\nClarification needed: for '{question}', share or confirm those clause sections."
        )
    return (
        "Low-confidence answer: retrieved evidence is weak or incomplete.\n"
        + text
        + f"\nClarification needed: please narrow the question for '{question}' with specific clause focus."
    )


def _has_strong_evidence(query_result, clauses, confidence_score):
    score_ok = float(confidence_score or 0.0) >= 0.76
    enough_clauses = len(clauses or []) >= 3
    missing_check = query_result.get("missing_clause_check") or {}
    complete = (not missing_check.get("enabled")) or bool(missing_check.get("is_complete", False))
    return bool(score_ok and enough_clauses and complete)


def _guard_unsafe_claims(answer_text, allow_unsafe):
    text = str(answer_text or "").strip()
    if not text:
        return text
    if allow_unsafe:
        return text

    # Ban hard negative/legal-conclusive claims unless evidence is strong.
    replacements = [
        (r"\bnot mentioned\b", "not clearly identified in the retrieved context"),
        (r"\bcannot be confirmed\b", "cannot be reliably confirmed from the retrieved context"),
        (r"\bonly principal is secured\b", "the retrieved context mainly indicates principal security"),
        (r"\bno pledge exists\b", "no explicit pledge was found in the retrieved clauses"),
        (r"\bdoes not exist\b", "was not found in the retrieved clauses"),
        (r"\bthere is no\b", "the retrieved context does not clearly show"),
        (r"\bnot present\b", "not clearly present in the retrieved context"),
        (r"\babsent\b", "not clearly evidenced in the retrieved context"),
    ]
    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    disclaimer = (
        "Evidence caution: conclusions are limited to retrieved clauses; "
        "verify against full contract sections before relying on negative assertions."
    )
    if disclaimer.lower() not in out.lower():
        out = out + "\n" + disclaimer
    return out


def _llm_answer(question, clause_type, clauses, chat_history=None):
    client, model_name, provider = _build_client()
    if not client or not model_name:
        return {"text": None, "called": False, "success": False, "model": None, "provider": None, "error": "LLM config missing"}

    context_rows = []
    for clause in clauses[:8]:
        clause_number = clause.get("clause_number") or clause.get("clause_idx") or "Unknown"
        context_rows.append(
            {
                "contract_name": clause.get("contract_name"),
                "clause_number": clause_number,
                "clause_text": clause.get("clause_text", ""),
            }
        )

    if not context_rows:
        return {"text": None, "called": False, "success": False, "model": model_name, "provider": provider, "error": "No context rows"}

    system_prompt = (
        "You are a legal contract assistant. "
        "Use only the provided context clauses and recent chat. "
        "Return a concise, synthesized final answer, not raw clause dumps. "
        "If the context is insufficient, state exactly what is missing. "
        "Never invent clause content. "
        "Do not make hard negative/conclusive claims unless explicitly proven by retrieved clauses."
    )

    user_prompt = {
        "question": question,
        "detected_clause_type": clause_type,
        "recent_chat_history": _build_chat_context(chat_history),
        "context_clauses": context_rows,
        "instructions": (
            "Output format:\n"
            "1) Final answer (2-6 sentences)\n"
            "2) Key supporting clauses as short bullet points with clause numbers\n"
            "3) If ambiguous, add one short 'Uncertainty' line.\n"
            "Do not paste long clause text verbatim."
        ),
    }

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            timeout=45,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
            ],
        )
        if response.choices and response.choices[0].message:
            text = (response.choices[0].message.content or "").strip() or None
            return {"text": text, "called": True, "success": bool(text), "model": model_name, "provider": provider, "error": None}
    except Exception as exc:
        return {"text": None, "called": True, "success": False, "model": model_name, "provider": provider, "error": str(exc)}

    return {"text": None, "called": True, "success": False, "model": model_name, "provider": provider, "error": "Empty response"}


def generate_answer(query_result, chat_history=None):
    question = query_result.get("question", "")
    clause_type = query_result.get("detected_type")
    clauses = query_result.get("clauses", [])

    citations = [_format_citation(clause) for clause in clauses]

    if not clauses:
        conf_score, conf_label = _compute_answer_confidence(query_result, clauses, llm_success=False)
        answer_text = _apply_confidence_policy(_fallback_answer(question, clause_type, clauses), question, conf_label, query_result)
        answer_text = _guard_unsafe_claims(answer_text, allow_unsafe=False)
        return {
            "answer": answer_text,
            "citations": citations,
            "llm_called": False,
            "llm_success": False,
            "llm_model_used": None,
            "llm_provider": None,
            "llm_error": None,
            "answer_confidence_score": conf_score,
            "answer_confidence_label": conf_label,
        }

    llm_result = _llm_answer(question, clause_type, clauses, chat_history=chat_history)
    llm_text = llm_result.get("text")
    if llm_text:
        conf_score, conf_label = _compute_answer_confidence(
            query_result,
            clauses,
            llm_success=bool(llm_result.get("success")),
        )
        answer_text = _apply_confidence_policy(llm_text, question, conf_label, query_result)
        answer_text = _guard_unsafe_claims(
            answer_text,
            allow_unsafe=_has_strong_evidence(query_result, clauses, conf_score),
        )

        return {
            "answer": answer_text,
            "citations": citations,
            "llm_called": llm_result.get("called"),
            "llm_success": llm_result.get("success"),
            "llm_model_used": llm_result.get("model"),
            "llm_provider": llm_result.get("provider"),
            "llm_error": llm_result.get("error"),
            "answer_confidence_score": conf_score,
            "answer_confidence_label": conf_label,
        }

    conf_score, conf_label = _compute_answer_confidence(
        query_result,
        clauses,
        llm_success=bool(llm_result.get("success")),
    )
    answer_text = _apply_confidence_policy(
        _fallback_answer(question, clause_type, clauses),
        question,
        conf_label,
        query_result,
    )
    answer_text = _guard_unsafe_claims(answer_text, allow_unsafe=False)
    return {
        "answer": answer_text,
        "citations": citations,
        "llm_called": llm_result.get("called"),
        "llm_success": llm_result.get("success"),
        "llm_model_used": llm_result.get("model"),
        "llm_provider": llm_result.get("provider"),
        "llm_error": llm_result.get("error"),
        "answer_confidence_score": conf_score,
        "answer_confidence_label": conf_label,
    }
