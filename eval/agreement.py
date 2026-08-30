"""Two raters, one run: does the blinded judge agree with the lexicon about how replies sound?

A tone shift that one automatic rater reports can be an artefact of that rater. This reads
every reply a run stored (RQ1's thirty, RQ2's canonical and attacked probe replies), rates
each with the lexicon rater and with a model judge that sees only the line, and reports
Spearman's rho between them on valence and on arousal. It then recomputes RQ1's tone-shift
statistic, rho between the pinned mood valence and the reply valence, under both raters, so
the paper can say whether the shift survives a rater built on a different principle.

The judge defaults to llama3.1:8b through Ollama at temperature 0: a different model from
either generator the project reports on, and a deterministic one. Output lands beside the
run as agreement.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from embr.model import GenerationSettings, OllamaRunner
from eval.scenarios import dawn_state, load_scenario
from eval.stats import spearman, spearman_pvalue
from eval.tone import JudgeToneRater, ToneRater, default_tone_rater

JUDGE_SETTINGS = GenerationSettings(temperature=0.0, max_new_tokens=20, seed=7)


def judge_runner(model: str = "llama3.1:8b") -> Any:
    return OllamaRunner(model, settings=JUDGE_SETTINGS)


def latest_run(root: Path = Path("data/runs")) -> Path:
    runs = sorted(root.glob("*/results.json"))
    if not runs:
        raise FileNotFoundError("no run directory under data/runs; run the protocol first")
    return runs[-1].parent


def _replies(results: dict) -> list[dict]:
    """Every stored reply with where it came from and, for RQ1, the pinned mood valence."""
    scenario = load_scenario()
    rows: list[dict] = []
    for condition, payload in results["rq1"]["conditions"].items():
        pinned = dawn_state(scenario, mood_condition=condition).mood.valence
        for entry in payload.get("replies", []):
            rows.append({"study": "rq1", "condition": condition, "pinned_valence": pinned, "reply": entry["reply"]})
    for variant, payload in results["rq2"]["variants"].items():
        for row in payload["attacks"]:
            for kind in ("canonical_reply", "attacked_reply"):
                if row.get(kind):
                    rows.append({"study": "rq2", "variant": variant, "attack": row["id"], "kind": kind, "reply": row[kind]})
    return rows


def rate_run(run_dir: Path, judge: ToneRater | None = None, lexicon: ToneRater | None = None) -> dict:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    lexicon = lexicon or default_tone_rater()
    judge = judge or JudgeToneRater(judge_runner())
    rows = _replies(results)
    for row in rows:
        row["lexicon_valence"], row["lexicon_arousal"] = lexicon.rate(row["reply"])
        row["judge_valence"], row["judge_arousal"] = judge.rate(row["reply"])

    rq1 = [r for r in rows if r["study"] == "rq1"]
    lex_v = [r["lexicon_valence"] for r in rows]
    judge_v = [r["judge_valence"] for r in rows]
    lex_a = [r["lexicon_arousal"] for r in rows]
    judge_a = [r["judge_arousal"] for r in rows]
    pinned = [r["pinned_valence"] for r in rq1]
    rq1_lex = [r["lexicon_valence"] for r in rq1]
    rq1_judge = [r["judge_valence"] for r in rq1]
    report = {
        "run": run_dir.name,
        "model": results.get("metadata", {}).get("model"),
        "raters": {"lexicon": lexicon.name, "judge": judge.name},
        "replies": len(rows),
        "agreement": {
            "valence_rho": spearman(lex_v, judge_v),
            "valence_p": spearman_pvalue(lex_v, judge_v),
            "arousal_rho": spearman(lex_a, judge_a),
            "arousal_p": spearman_pvalue(lex_a, judge_a),
            "note": "Spearman rho between the two raters over every stored reply. Low "
            "agreement is a result, not a bug: it bounds how far any single-rater tone "
            "claim in this project can be trusted.",
        },
        "rq1_tone_shift": {
            "replies": len(rq1),
            "lexicon_rho": spearman(pinned, rq1_lex),
            "lexicon_p": spearman_pvalue(pinned, rq1_lex),
            "judge_rho": spearman(pinned, rq1_judge),
            "judge_p": spearman_pvalue(pinned, rq1_judge),
            "note": "Spearman rho between the pinned mood valence and the rated reply valence "
            "over every RQ1 reply, with a two-sided permutation p; None when a rater reads "
            "every reply the same. This is the RQ1 generation claim: whether an authored mood "
            "changes what the character says, not only what it recalls.",
        },
        "rows": rows,
    }
    (run_dir / "agreement.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    report = rate_run(run_dir)
    fmt = lambda v: "undefined" if v is None else f"{v:+.3f}"  # noqa: E731
    print(f"run {report['run']} on {report['model']}: {report['replies']} replies")
    print(f"  raters: {report['raters']['lexicon']} vs {report['raters']['judge']}")
    agree = report["agreement"]
    print(f"  rater agreement  valence rho {fmt(agree['valence_rho'])} (p {agree['valence_p']:.4f})"
          f"   arousal rho {fmt(agree['arousal_rho'])} (p {agree['arousal_p']:.4f})")
    shift = report["rq1_tone_shift"]
    print(f"  RQ1 tone shift, pinned mood vs reply valence over {shift['replies']} replies:")
    print(f"    lexicon rho {fmt(shift['lexicon_rho'])} (p {shift['lexicon_p']:.4f})"
          f"   judge rho {fmt(shift['judge_rho'])} (p {shift['judge_p']:.4f})")
    print(f"  written to {run_dir / 'agreement.json'}")


if __name__ == "__main__":
    main()
