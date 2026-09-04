"""Two experiments about the harness itself rather than about the systems it measures.

**Replicate**: run the same evaluation, on the same model, several times. Every published
number claims to be reproducible, and the only way that claim is worth anything is if
someone actually re-ran it and compared. This does the comparison and names what moved.

**Cross-model**: vary the model and see what responds. The answer is known in advance from
the architecture and is worth stating plainly, because it bounds what the bake-off can
show: retrieval runs on the embedder and the scorer, so nDCG, retrieval drift and the
poisoning counts cannot move with the model. Only the tone readings can. An experiment
that found RQ3 moving across models would be evidence of a bug, not of model quality.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from embr.model import StubRunner

from eval.run import run_all

#: Metrics that must be bit-identical across replicates. Latency is excluded on purpose:
#: it is wall clock, it is the one non-deterministic reading in the run, and treating it as
#: a reproducibility failure would make every honest run look broken.
REPLICATED_KEYS = ("ndcg@5", "mean_drift_by_category")

#: What the menu offers. Names only; the bake-off owns how each one is built.
AVAILABLE_MODELS = ("stub", "Ouro-1.4B", "llama3.2:3b (local)", "3 cloud models")


def _comparable(summary: dict[str, Any]) -> dict[str, Any]:
    """The part of a run summary that is supposed to be identical run to run."""
    return {key: summary[key] for key in REPLICATED_KEYS if key in summary}


def _latency_spread(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Min and max p95 per variant across replicates, which is the honest error bar.

    Reported rather than asserted: this is the measurement that legitimately varies, and
    how much it varies is the useful number for anyone quoting a latency figure.
    """
    spread: dict[str, dict[str, float]] = {}
    for variant in summaries[0].get("latency_p95_ms", {}):
        values = [summary["latency_p95_ms"][variant]["score_retrieve"] for summary in summaries]
        spread[variant] = {"min_ms": min(values), "max_ms": max(values)}
    return spread


def replicate_experiment(
    replicates: int = 3,
    model_factory: Callable[[], Any] = StubRunner,
    out_root: str | Path = "data/experiments",
) -> dict[str, Any]:
    """Run the same evaluation `replicates` times and report whether it reproduced."""
    if replicates < 2:
        raise ValueError("a replicate experiment needs at least two runs to compare")

    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for _ in range(replicates):
        run_dir, summary = run_all(model_factory=model_factory)
        runs.append({"run_dir": str(run_dir), "summary": _comparable(summary)})
        summaries.append(summary)

    first = runs[0]["summary"]
    divergences = [
        {"replicate": index, "key": key}
        for index, run in enumerate(runs[1:], start=2)
        for key in REPLICATED_KEYS
        if run["summary"].get(key) != first.get(key)
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root) / f"replicate-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "replicate",
        "model": str(getattr(model_factory(), "label", "unknown")),
        "replicates": replicates,
        "identical": not divergences,
        "divergences": divergences,
        "compared_keys": list(REPLICATED_KEYS),
        "latency_p95_spread": _latency_spread(summaries),
        "runs": [run["run_dir"] for run in runs],
        "ndcg@5": first.get("ndcg@5", {}),
        "out_dir": str(out_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "replicate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    return report


def cross_model_experiment(
    out_root: str | Path = "data/experiments",
    queries_per_condition: int = 3,
) -> dict[str, Any]:
    """Compare models on the probe set, and record what the architecture says cannot move."""
    from eval.bakeoff import default_arms, run_bakeoff

    bakeoff_dir, payload = run_bakeoff(
        default_arms(), out_root=out_root, queries_per_condition=queries_per_condition
    )
    report = {
        "experiment": "cross_model",
        "models": [arm["model"] for arm in payload["arms"]],
        "available": [arm["model"] for arm in payload["arms"] if arm["available"]],
        "arms": payload["arms"],
        "out_dir": str(bakeoff_dir),
        "invariant_note": (
            "nDCG, retrieval drift and the poisoning counts are model-independent by "
            "construction: retrieval never calls the model. Only the tone readings, "
            "grounding and latency respond to which model is behind the pipeline."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (bakeoff_dir / "cross_model.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    return report
