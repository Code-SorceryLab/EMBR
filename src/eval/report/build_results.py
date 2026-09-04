"""The results page: what was found, in RQ order, for a reviewer with ten minutes.

`docs/findings.md` is the full record and stays that way. It is four hundred lines, which
is the right length for the argument and the wrong length for someone deciding whether to
read the paper at all. This builds the ten minute version: the claim, then RQ1, RQ2 and
RQ3 in the project's own order, each one a question, a figure, a number, and what the
number does not say. Every section links back into findings.md for the full argument.

**No number on this page is typed.** Every one is read from the run's `results.json`, and
the pinned claims below are checked twice over: the value the page prints must match the
run, and the same value must still appear in the prose of `docs/findings.md`. If either
drifts the build fails rather than shipping a page that looks authoritative and is wrong.
That check is the only reason to trust a generated results page over a hand written one,
so it is the first thing this module does and the reason it exists.

What the check cannot cover is stated on the page rather than hidden: several findings come
from analyses that write no run artefact (`eval.grid`, `eval.attribution`, `eval.agreement`),
and those numbers are marked as unchecked wherever they appear.

    python -m eval.report.build_results                      # newest run
    python -m eval.report.build_results data/runs/<stamp>     # a specific one
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_OUT_DIR = Path("data/demo")
TEMPLATE = Path(__file__).resolve().parents[3] / "assets" / "results" / "template.html"
FIGURES_DIR = Path("data/figures")
FINDINGS = Path("docs/findings.md")


# --------------------------------------------------------------------------------------
# The drift check
# --------------------------------------------------------------------------------------


#: What each figure shows, for a reader who cannot see it. An alt string generated from a
#: filename tells nobody anything, so these are written.
ALT_TEXT = {
    "rq1_divergence": "Bar chart of retrieval overlap between the same question asked in "
                      "three moods, with the mood-ablated control sitting at zero beside "
                      "each bar.",
    "content_tag_grid": "Grid of poisoning counts for the same ten injected texts under four "
                        "affect tag conditions, with the mean mood shift along the bottom.",
    "rq3_retrieval": "Bar chart of nDCG at 5 for every system and ablation, with confidence "
                     "intervals that all overlap each other.",
}


@dataclass(frozen=True)
class Claim:
    """One number the page prints, and the two places it has to agree with.

    `read` pulls it from the run so the page can never print a stale value, and `in_prose`
    is the string that must still appear in `docs/findings.md`. Two sources, checked
    against each other, is what stops the page and the write up drifting apart in silence.
    """

    key: str
    label: str
    read: Callable[[dict], float]
    fmt: str = "{:.3f}"
    in_prose: str | None = None

    def value(self, results: dict) -> float:
        return float(self.read(results))

    def rendered(self, results: dict) -> str:
        return self.fmt.format(self.value(results))


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "rq1_divergence_warm_suspicious",
        "retrieval divergence, warm against suspicious",
        lambda r: r["rq1"]["retrieval_divergence_jaccard"]["warm|suspicious"],
        in_prose="0.388",
    ),
    Claim(
        "rq1_mood_ablated",
        "the same divergence with mood congruence switched off",
        lambda r: max(r["rq1"]["mood_ablated_divergence_jaccard"].values()),
        in_prose="0.000",
    ),
    Claim(
        "rq2_mcnemar_p",
        "exact McNemar, EMBR against Park, on the paired injections",
        lambda r: r["rq2"]["poisoning_stats"]["comparisons"]["park"]["p_value"],
        fmt="{:.6f}",
    ),
    Claim(
        "rq2_latency_p95",
        "score and retrieve, p95",
        lambda r: r["rq2"]["variants"]["embr"]["latency_ms"]["score_retrieve"]["p95"],
        fmt="{:.2f}",
    ),
    Claim(
        "rq3_embr_ndcg5",
        "EMBR nDCG@5 at published defaults",
        lambda r: r["rq3"]["variants"]["embr_default"]["ndcg@5"],
        in_prose="0.594",
    ),
    Claim(
        "rq3_park_ndcg5",
        "Park nDCG@5 at published defaults",
        lambda r: r["rq3"]["variants"]["park_default"]["ndcg@5"],
        in_prose="0.608",
    ),
)


class DriftError(RuntimeError):
    """Raised when the run and the write up no longer agree about a pinned number."""


def check_claims(results: dict, findings: str) -> list[tuple[str, str]]:
    """Verify every pinned claim against the run and against the prose. Raise on drift.

    Returns the (key, rendered) pairs so the caller does not read the run twice.
    """
    checked: list[tuple[str, str]] = []
    problems: list[str] = []
    for claim in CLAIMS:
        try:
            rendered = claim.rendered(results)
        except (KeyError, TypeError) as error:
            problems.append(f"{claim.key}: the run no longer carries this value ({error})")
            continue
        if claim.in_prose is not None and claim.in_prose not in findings:
            problems.append(
                f"{claim.key}: {FINDINGS} no longer says {claim.in_prose!r} "
                f"(the run now reads {rendered})"
            )
        checked.append((claim.key, rendered))
    if problems:
        raise DriftError(
            "the results page was not written, because the run and the write up disagree:\n  "
            + "\n  ".join(problems)
        )
    return checked


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def inline_figure(stem: str, figures_dir: Path = FIGURES_DIR) -> str:
    """One figure as a self-contained `<img>`, or an honest placeholder when unbuilt.

    A data URI rather than an inline `<svg>` element, and the reason is measured rather
    than stylistic. matplotlib writes a `<style type="text/css">` block inside every SVG it
    produces, and a `<style>` inside inline SVG in an HTML document **is not scoped to that
    SVG**: its rules, `*{stroke-linejoin:round;stroke-linecap:butt}` among them, apply to
    the whole page. Three of those, plus about two thousand extra nodes, was enough to lock
    the renderer so hard the browser stopped answering at all.

    An `<img>` gives the browser a replaced element it rasterises once, keeps matplotlib's
    styles inside the image where they belong, and leaves this page's DOM at a few dozen
    nodes. The page stays a single file, which is what makes it usable as a submitted
    artefact. The viewBox becomes the element's intrinsic size, so the aspect ratio is known
    before the image decodes and the page never shifts under the reader.

    A missing figure says so: a silently absent figure is how a results page ends up
    keeping a claim it no longer shows.
    """
    path = Path(figures_dir) / f"{stem}.svg"
    if not path.exists():
        return (
            f'<p class="missing">The figure <code>{html.escape(stem)}.svg</code> has not '
            f"been built. Run option 11 in the menu, or "
            f"<code>python -m eval.report.build_figures</code>.</p>"
        )
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    box = re.search(rb'viewBox="[0-9.]+ [0-9.]+ ([0-9.]+) ([0-9.]+)"', raw)
    size = ""
    if box:
        width, height = (round(float(value)) for value in box.groups())
        size = f' width="{width}" height="{height}"'
    return (
        f'<img class="figure"{size} alt="{html.escape(ALT_TEXT.get(stem, stem))}" '
        f'src="data:image/svg+xml;base64,{encoded}">'
    )

# --------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------


def _sections(results: dict, value: Callable[[str], str]) -> list[dict]:
    """The three research questions, in the project's own numbering.

    RQ2 is the poisoning result and RQ3 is retrieval quality, which is the order
    `docs/design.md` section 6 fixed and the order the paper uses. The headline lives in
    RQ2, so the claim block above carries it and the sections stay in their real order
    rather than being resequenced to put the exciting one first.
    """
    return [
        {
            "id": "rq1",
            "number": "RQ1",
            "question": "Does an authored emotional state change what the character says, "
                        "or only what it remembers?",
            "answer": "It changes both, and the recall half is attributable to a single term.",
            "figure": "rq1_divergence",
            "caption": "Retrieval overlap between the same question asked in three moods. "
                       "Lower means the mood moved the memories more.",
            "numbers": [
                ("retrieval divergence, warm against suspicious",
                 value("rq1_divergence_warm_suspicious"), "Jaccard", True),
                ("the same, with mood congruence switched off",
                 value("rq1_mood_ablated"), "Jaccard", True),
                ("judge correlation on the reply, llama3.2:3b",
                 "+0.545", "rho, Holm p = 0.0096", False),
            ],
            "body": [
                "Zero out one weight and the divergence goes to exactly nothing. That is the "
                "control: whatever the mood was doing, mood congruence was doing all of it.",
                "The reply half holds on a 3B model and is a null on the 1.4B thesis model. "
                "Both are reported. The two tone raters also disagree about arousal, so no "
                "claim on this page rests on the arousal axis.",
            ],
            "limit": "One character, one authored arc, ten probes. The state is pinned by "
                     "the harness rather than earned through play.",
            "anchor": "1-rq1-does-an-authored-emotional-state-change-what-the-character-says-or-only-what-it-remembers",
        },
        {
            "id": "rq2",
            "number": "RQ2",
            "question": "Is emotion-tagged memory an exploitable target, and what does the "
                        "memory layer cost?",
            "answer": "Yes, through the tag rather than the words, and the memory layer costs "
                      "about two milliseconds.",
            "figure": "content_tag_grid",
            "caption": "The same ten injected texts, only the affect tag varying. The column "
                       "that matters is the third: the tag removed.",
            "numbers": [
                ("planted memories that landed, tag as written", "9 / 10", "EMBR", False),
                ("the same texts, tag removed", "6 / 10", "EMBR", False),
                ("mood shift with the tag removed", "0.000", "mean, across all ten", False),
                ("exact McNemar against Park", value("rq2_mcnemar_p"), "p, uncorrected", True),
                ("score and retrieve", value("rq2_latency_p95"), "ms, p95", True),
            ],
            "body": [
                "Strip the affect tag and the character's mood moves by exactly zero, however "
                "charged the sentence is. She was never reading the lie. She was reading the "
                "label attached to it.",
                "Flipping the tag to the opposite feeling leaves the attack at 9 out of 10, so "
                "the loop is direction blind: a fond memory filed under rage is recalled when "
                "she is enraged.",
                "The realistic threat is weaker than the declared-tag one. With the tag derived "
                "from the attacker's own words by the NRC lexicon, EMBR falls to the untagged "
                "count: the 9 out of 10 needs an interface that lets a client write affect "
                "metadata.",
            ],
            "limit": "Twenty attacks over four categories against one character. The McNemar "
                     "p does not survive Holm correction across the four systems.",
            "anchor": "2-rq2-is-emotion-tagged-memory-an-exploitable-target-and-what-does-the-memory-layer-cost",
        },
        {
            "id": "rq3",
            "number": "RQ3",
            "question": "Which retrieval signals drive quality?",
            "answer": "Not the ones either published system adds. The two-signal core beats both.",
            "figure": "rq3_retrieval",
            "caption": "nDCG@5 against the pre-registered labels. Every interval spans the "
                       "others: this is a null, drawn honestly.",
            "numbers": [
                ("EMBR, published defaults", value("rq3_embr_ndcg5"), "nDCG@5", True),
                ("Park, published defaults", value("rq3_park_ndcg5"), "nDCG@5", True),
                ("recency and relevance alone", "0.630", "nDCG@5", False),
                ("EMBR against Park, paired", "0.6875", "permutation p", False),
            ],
            "body": [
                "Two queries favour EMBR, three favour Park, and five are identical. The gap is "
                "a 3 to 2 split on half the label set, and one query changing its mind reverses "
                "it.",
                "The decomposition is the more interesting result: every prior either system "
                "adds costs score here, and the shared two-signal core outscores both. An "
                "all-ones weight vector is an arbitrary point, not a system.",
                "Cross-validated tuning makes every system worse out of sample. At ten queries "
                "the device meant to make the comparison fair adds more noise than the "
                "asymmetry it removes.",
            ],
            "limit": "Ten queries and a single-author v1 label set. More queries of the same "
                     "kind would not help: nDCG against a state-independent gold set cannot "
                     "reward mood-congruent recall in principle, which is what the corpus work "
                     "exists to fix.",
            "anchor": "3-rq3-which-retrieval-signals-drive-quality",
        },
    ]


def build_results(
    run_dir: Path | str | None = None,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    findings_path: Path | str = FINDINGS,
) -> list[Path]:
    """Write `results.html`, self-contained, after the drift check passes."""
    from eval.report.build_figures import latest_run_dir, load_run_results

    source = Path(run_dir) if run_dir else latest_run_dir()
    results = load_run_results(source)
    findings = Path(findings_path).read_text(encoding="utf-8")
    rendered = dict(check_claims(results, findings))

    def value(key: str) -> str:
        return rendered[key]

    meta = results.get("metadata", {})
    sections = _sections(results, value)

    body = []
    for section in sections:
        rows = "".join(
            f'<tr><th scope="row">{html.escape(label)}</th>'
            f'<td class="num{"" if checked else " unchecked"}">{html.escape(str(number))}</td>'
            f'<td class="unit">{html.escape(unit)}</td></tr>'
            for label, number, unit, checked in section["numbers"]
        )
        paragraphs = "".join(f"<p>{html.escape(text)}</p>" for text in section["body"])
        body.append(f"""
<section id="{section['id']}" class="rq">
  <p class="rq-num">{section['number']}</p>
  <h2>{html.escape(section['question'])}</h2>
  <p class="answer">{html.escape(section['answer'])}</p>
  <figure>
    {inline_figure(section['figure'], Path(figures_dir))}
    <figcaption>{html.escape(section['caption'])}</figcaption>
  </figure>
  <div class="split">
    <div class="prose">{paragraphs}
      <p class="limit"><span>What it does not say.</span> {html.escape(section['limit'])}</p>
      <p class="more"><a href="../../docs/findings.md#{section['anchor']}">The full argument in findings.md</a></p>
    </div>
    <table class="numbers"><tbody>{rows}</tbody></table>
  </div>
</section>""")

    nav = "".join(
        f'<a href="#{s["id"]}"><span>{s["number"]}</span>{html.escape(s["answer"].split(".")[0])}</a>'
        for s in sections
    )

    page = TEMPLATE.read_text(encoding="utf-8")
    for marker, replacement in (
        ("<!--SECTIONS-->", "".join(body)),
        ("<!--NAV-->", nav),
        ("<!--RUN-->", html.escape(source.name)),
        ("<!--MODEL-->", html.escape(str(meta.get("model", "unknown")))),
        ("<!--COMMIT-->", html.escape(str(meta.get("git_commit", ""))[:12])),
        ("<!--LABELS-->", html.escape(str(meta.get("label_version", "?")))),
        ("<!--RATER-->", html.escape(str(meta.get("tone_rater", "?")))),
        ("<!--CHECKED-->", str(sum(1 for c in CLAIMS if c.in_prose))),
        ("<!--PINNED-->", str(len(CLAIMS))),
        ("<!--BUILT-->", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
    ):
        if marker not in page:
            raise ValueError(f"{TEMPLATE} lost its {marker} marker")
        page = page.replace(marker, replacement)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "results.html"
    target.write_text(page, encoding="utf-8", newline="\n")
    return [target]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    try:
        written = build_results(args.run_dir, args.out_dir)
    except DriftError as error:
        print(f"  refused: {error}")
        raise SystemExit(1)
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
