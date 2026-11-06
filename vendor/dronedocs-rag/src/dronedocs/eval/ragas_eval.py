"""Optional Ragas cross-check on a sample of answered golden items.

Ragas is the reference RAG-eval framework, but it is LLM-call heavy (several calls per metric
per sample) and Groq's free tier is 8k tokens/minute, so it runs on a small sample and the
built-in single-call judge (metrics.py) remains the CI gate. Reported side by side so the
two judges can be compared.

Compatibility note: ragas 0.4.3 imports ``langchain_community.chat_models.vertexai``, which was
removed in langchain-community 0.4. We register an inert stub module before importing ragas;
nothing in our path uses Vertex AI.
"""

from __future__ import annotations

import os
import sys
import types

from ..chunking import Chunk


def _install_vertex_stub() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)

        class ChatVertexAI:  # pragma: no cover - placeholder only
            pass

        mod.ChatVertexAI = ChatVertexAI
        sys.modules[name] = mod


def run_ragas(cfg: dict, items: list[dict], rows: list[dict], chunks: dict[str, Chunk], sample: int = 8) -> dict:
    _install_vertex_stub()
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import _Faithfulness, _LLMContextRecall

    by_id = {it["id"]: it for it in items}
    picked = [r for r in rows if r["answerable"] and not r["refused"]][:sample]
    if not picked:
        return {"status": "skipped", "reason": "no answered items"}
    data = [
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": [chunks[c].text for c in r["context_ids"]],
            "reference": by_id[r["id"]]["reference"],
        }
        for r in picked
    ]
    llm_cfg = cfg["llm"]
    base_url = os.environ.get(llm_cfg.get("base_url_env", "LLM_BASE_URL")) or llm_cfg["default_base_url"]
    api_key = os.environ.get(llm_cfg.get("api_key_env", "LLM_API_KEY")) or os.environ.get("GROQ_API_KEY")
    ragas_model = cfg["eval"].get("ragas_model") or cfg["judge"]["model"]
    chat = ChatOpenAI(model=ragas_model, base_url=base_url, api_key=api_key, temperature=0, max_retries=6, timeout=90)
    llm = LangchainLLMWrapper(chat)
    result = evaluate(
        EvaluationDataset.from_list(data),
        metrics=[_Faithfulness(llm=llm), _LLMContextRecall(llm=llm)],
        run_config=RunConfig(max_workers=1, max_retries=6, max_wait=60, timeout=180),
        show_progress=False,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    out = {"status": "ok", "n": len(picked), "ragas_model": ragas_model}
    for col in ("faithfulness", "context_recall"):
        if col in df:
            vals = [float(v) for v in df[col].tolist() if v == v]  # drop NaN
            out[col] = round(sum(vals) / len(vals), 4) if vals else None
            out[f"{col}_n_scored"] = len(vals)
    builtin = [r.get("faithfulness") for r in picked if r.get("faithfulness") is not None]
    out["builtin_faithfulness_same_items"] = round(sum(builtin) / len(builtin), 4) if builtin else None
    out["ids"] = [r["id"] for r in picked]
    return out
