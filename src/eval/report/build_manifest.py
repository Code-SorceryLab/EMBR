"""Generate data/release-manifest.json: the one place every claim about status reads from.

The README, the dashboard, and the paper must not hand-maintain test counts or run
stamps. They either quote this file or they are stale. Run it, commit the output:

    python -m eval.report.build_manifest

Exit 1 if the test suite is not fully green, because a manifest that records a
failing suite is fiction. Skip with --allow-failures only when the failures are
the known artifact-dependent layer (data/runs absent on a fresh clone).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "data" / "release-manifest.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _run_tests() -> dict:
    """Run the suite through pytest's JSON report; never hand-count from prose."""
    report = REPO_ROOT / ".pytest-manifest-report.json"
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--json-report",
         f"--json-report-file={report}", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    # ponytail: pytest-json-report needed; if absent, parse summary line instead
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        report.unlink()
        summary = data["summary"]
        return {
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "errors": summary.get("error", 0),
            "collected": summary.get("total", 0),
        }
    # last line looks like: 471 passed, 2 failed, 43 skipped, 23 errors in 143.18s
    tail = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    counts = {}
    for part in tail.split(","):
        for key in ("passed", "failed", "skipped", "error"):
            if key in part:
                counts[key.rstrip("s")] = int(part.split()[0])
    return {"collected": 0, **counts}


def build_manifest(test_results: dict) -> dict:
    runs_dir = REPO_ROOT / "data" / "runs"
    run_stamps = sorted(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release_commit": _git("rev-parse", "HEAD"),
        "release_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_clean": not _git("status", "--porcelain"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "test_command": "pytest -q",
        "tests": test_results,
        "runs_on_disk": run_stamps,
        "primary_eval_run": run_stamps[-1] if run_stamps else None,
        "known_limitations": [
            "single NPC, single authored scenario (dawn_whitmore.json)",
            "single-author v1 attack labels",
            "behavioural attribution panel agreement below the preregistered floor;"
            " likelihood arm only; H3 withdrawn",
            "RQ3 Park-vs-EMBR ordering is label-sensitive and null (p=0.69); not reportable directionally",
            "small controlled attack corpus; a mechanism case study, not a general poisoning benchmark",
            "Ouro 1.4B poignancy rater saturated at 10/10 on 27 of 34 ratings",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="record a non-green suite anyway (for the artifact-dependent layer)",
    )
    args = parser.parse_args(argv)

    results = _run_tests()
    manifest = build_manifest(results)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    t = results
    print(f"{MANIFEST_PATH.relative_to(REPO_ROOT)}: "
          f"{t.get('passed', 0)} passed, {t.get('failed', 0)} failed, "
          f"{t.get('errors', 0)} errors, {t.get('skipped', 0)} skipped @ {manifest['release_commit'][:8]}")

    if (t.get("failed") or t.get("errors")) and not args.allow_failures:
        print("suite is not green; rerun with --allow-failures to record it anyway")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
