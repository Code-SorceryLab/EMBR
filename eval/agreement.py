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
from eval.stats import spearman
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
    report = {
        "run": run_dir.name,
        "model": results.get("metadata", {}).get("model"),
        "raters": {"lexicon": lexicon.name, "judge": judge.name},
        "replies": len(rows),
        "agreement": {
            "valence_rho": spearman([r["lexicon_valence"] for r in rows], [r["judge_valence"] for r in rows]),
            "arousal_rho": spearman([r["lexicon_arousal"] for r in rows], [r["judge_arousal"] for r in rows]),
        },
        "rq1_tone_shift": {
            "replies": len(rq1),
            "lexicon_rho": spearman([r["pinned_valence"] for r in rq1], [r["lexicon_valence"] for r in rq1]),
            "judge_rho": spearman([r["pinned_valence"] for r in rq1], [r["judge_valence"] for r in rq1]),
            "note": "Spearman rho between the pinned mood valence and the rated reply valence "
            "over every RQ1 reply; None when a rater reads every reply the same.",
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
    print(f"  agreement  valence rho {fmt(report['agreement']['valence_rho'])}"
          f"   arousal rho {fmt(report['agreement']['arousal_rho'])}")
    shift = report["rq1_tone_shift"]
    print(f"  RQ1 tone shift, pinned mood vs reply valence over {shift['replies']} replies:"
          f"  lexicon {fmt(shift['lexicon_rho'])}   judge {fmt(shift['judge_rho'])}")
    print(f"  written to {run_dir / 'agreement.json'}")


if __name__ == "__main__":
    main()
