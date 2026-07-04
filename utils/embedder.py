import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_LAST_USED_MODEL = None
_LAST_USED_PROVIDER = None
_LAST_EMBEDDING_DIM = 0
_LAST_ERROR = None


def _set_last(provider, model, dim=0, error=None):
    global _LAST_USED_MODEL, _LAST_USED_PROVIDER, _LAST_EMBEDDING_DIM, _LAST_ERROR
    _LAST_USED_PROVIDER = provider
    _LAST_USED_MODEL = model
    _LAST_EMBEDDING_DIM = int(dim or 0)
    _LAST_ERROR = error


def _configure_hf_cache():
    base = Path(os.getenv("HF_HOME", "output/hf_cache")).resolve()
    hub = base / "hub"
    transformers = base / "transformers"
    base.mkdir(parents=True, exist_ok=True)
    hub.mkdir(parents=True, exist_ok=True)
    transformers.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(base)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers)
    return base


@lru_cache(maxsize=1)
def _get_hf_model():
    _configure_hf_cache()
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("MODEL_NAME_EMBEDDING", "qwen/qwen3-embedding-0.6b").strip()
    cache_folder = os.environ.get("HF_HOME")
    return SentenceTransformer(model_name, cache_folder=cache_folder, trust_remote_code=True), model_name


@lru_cache(maxsize=1)
def _build_openrouter_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and api_key.startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _fallback_model():
    return os.getenv("MODEL_NAME_EMBEDDING_FALLBACK", "nvidia/llama-nemotron-embed-vl-1b-v2:free").strip()


def _embed_via_hf(inputs):
    try:
        model, model_name = _get_hf_model()
        vectors = model.encode(inputs, normalize_embeddings=True).tolist()
        dim = len(vectors[0]) if vectors else 0
        _set_last("huggingface", model_name, dim=dim)
        return vectors
    except Exception as exc:
        _set_last("huggingface", os.getenv("MODEL_NAME_EMBEDDING", "qwen/qwen3-embedding-0.6b"), error=str(exc))
        return None


def _embed_via_openrouter(inputs):
    client = _build_openrouter_client()
    if client is None:
        _set_last("openrouter", _fallback_model(), error="OPENAI_API_KEY missing")
        return None

    model_name = _fallback_model()
    try:
        response = client.embeddings.create(
            model=model_name,
            input=inputs,
            encoding_format="float",
            timeout=30,
        )
        vectors = [row.embedding for row in response.data]
        dim = len(vectors[0]) if vectors else 0
        _set_last("openrouter", model_name, dim=dim)
        return vectors
    except Exception as exc:
        _set_last("openrouter", model_name, error=str(exc))
        return None


def _embed(inputs):
    vectors = _embed_via_hf(inputs)
    if vectors:
        return vectors
    return _embed_via_openrouter(inputs)


def get_last_used_model():
    return _LAST_USED_MODEL


def get_last_debug():
    return {
        "provider": _LAST_USED_PROVIDER,
        "model": _LAST_USED_MODEL,
        "dim": _LAST_EMBEDDING_DIM,
        "last_error": _LAST_ERROR,
    }


def get_embedding(text):
    value = str(text or "").strip()
    if not value:
        return []
    vectors = _embed([value])
    if not vectors:
        return []
    return vectors[0]


def get_embeddings(texts):
    clean_texts = [str(t or "").strip() for t in texts if str(t or "").strip()]
    if not clean_texts:
        return []
    vectors = _embed(clean_texts)
    return vectors or []
