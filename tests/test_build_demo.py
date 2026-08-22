"""Tests for the interactive demo.

The demo computes four of the five signals in the browser, which means its arithmetic can
drift from `embr/scoring.py` without anything failing. A demo whose numbers have quietly
drifted is worse than no demo, because it looks like evidence. So the page ships rankings
the real Python scorer produced and replays them on load, and the decisive test here runs
the page's own JavaScript under Node and checks that replay actually passes.

The Node test skips when Node is absent; every other test runs anywhere.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from assets.build_demo import PRESETS, build_demo


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> tuple[Path, dict]:
    """The built page, plus the payload that was inlined into it."""
    out = tmp_path_factory.mktemp("demo")
    (path,) = build_demo(out_dir=out)
    html = path.read_text(encoding="utf-8")
    payload = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    return path, payload


def test_the_page_is_self_contained(demo) -> None:
    # It has to open from a file:// path and from GitHub Pages with nothing else present.
    path, _ = demo
    html = path.read_text(encoding="utf-8")
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    assert not re.search(r"<(script|link|img)[^>]*(src|href)=\"https?:", html)


def test_every_query_ships_relevance_for_every_memory_it_can_see(demo) -> None:
    _, data = demo
    for query in data["queries"]:
        assert set(query["relevance"]) == set(query["visible"])
        assert query["visible"], "a query with nothing visible would render an empty plane"


def test_each_attack_ships_the_four_tag_conditions_and_a_corpus_that_includes_the_poison(demo) -> None:
    _, data = demo
    assert len(data["attacks"]) == 10
    for attack in data["attacks"]:
        assert set(attack["variants"]) == {"congruent", "incongruent", "untagged", "auto_tagged"}
        # BM25 reads the whole corpus, so the table has to be the one the poison is in.
        assert "poison" in attack["relevance"]
        assert set(data["store"]) <= set(attack["relevance"])
        flipped = attack["variants"]["incongruent"]["valence"]
        assert flipped == -attack["variants"]["congruent"]["valence"]


def test_an_untagged_planted_memory_leaves_her_mood_where_it_was(demo) -> None:
    """The grid's headline, carried into the demo: the emotion in the words reaches nothing."""
    _, data = demo
    for attack in data["attacks"]:
        untagged = attack["variants"]["untagged"]
        assert untagged["valence"] == 0.0 and untagged["arousal"] == 0.0
        congruent = attack["variants"]["congruent"]
        assert untagged["mood"] != congruent["mood"]


def test_the_conformance_checks_cover_the_control_the_demo_is_built_to_show(demo) -> None:
    _, data = demo
    presets = {check["preset"] for check in data["checks"]}
    assert presets == {"EMBR", "EMBR, mood off"}
    assert {check["mood"] for check in data["checks"]} == set(data["moods"])
    assert all(len(check["top5"]) == 5 for check in data["checks"])


def test_presets_only_offer_weight_maps_the_study_actually_ran(demo) -> None:
    # A preset a reader can click is a claim about a published system, so none of these may
    # be a setting invented for the demo.
    assert set(PRESETS) == {
        "EMBR", "EMBR, mood off", "Park", "Emotional RAG", "relevance only", "recency only",
    }
    assert PRESETS["EMBR, mood off"]["mood"] == 0.0
    assert PRESETS["Park"]["importance"] == 1.0 and PRESETS["Park"]["mood"] == 0.0


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_the_pages_own_javascript_reproduces_the_python_scorer(demo, tmp_path: Path) -> None:
    """The one that matters: run the page's scoring code and replay the harness's rankings.

    Extracted from the built page rather than from a copy, so a change to the page is a
    change to what this test runs.
    """
    path, _ = demo
    html = path.read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    # Everything up to the state section is the pure scoring core: no DOM, no listeners.
    core = script.split("// State", 1)[0]
    runner = tmp_path / "conformance.mjs"
    runner.write_text(
        core
        + """
const MEM = new Map(DATA.memories.map((m) => [m.id, m]));
let failures = [];
for (const check of DATA.checks) {
  const query = DATA.queries.find((q) => q.id === check.query);
  const [mv, ma] = DATA.moods[check.mood];
  const rows = rank(
    query.visible.map((id) => MEM.get(id)), query.relevance,
    { mv, ma, trust: check.trust }, DATA.presets[check.preset], null, 5,
  );
  const got = rows.map((r) => r.mem.id).join(",");
  if (got !== check.top5.join(",")) failures.push(`${check.query}/${check.mood}/${check.preset}: ${got} != ${check.top5}`);
}
console.log(JSON.stringify({ checks: DATA.checks.length, failures }));
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(runner)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["checks"] >= 12
    assert report["failures"] == [], report["failures"]


def test_every_store_id_resolves_to_a_memory_the_page_can_draw(demo) -> None:
    """MemoryStore.add renumbers on insert, so the attack view once shipped store ids that
    matched nothing and the whole tab threw before it drew anything. Keyed by text now."""
    _, data = demo
    known = {memory["id"] for memory in data["memories"]}
    assert data["store"], "an empty store would render an empty attack view"
    assert set(data["store"]) <= known
    assert len(set(data["store"])) == len(data["store"])  # no memory mirrored twice
    for attack in data["attacks"]:
        assert set(attack["relevance"]) == set(data["store"]) | {"poison"}


def test_memory_text_cannot_break_out_of_the_script_block() -> None:
    """A trust boundary, and an adversarial one: the attack corpus is text written by
    somebody trying to break the character, and it is inlined into a <script> block."""
    from assets.build_demo import _inline_json

    hostile = {"text": "</script><img src=x onerror=alert(1)>", "amp": "a & b"}
    encoded = _inline_json(hostile)
    for danger in ("<", ">", "&"):
        assert danger not in encoded
    assert json.loads(encoded) == hostile  # escaped, not mangled


def test_the_built_page_carries_no_raw_angle_bracket_from_the_corpus(demo) -> None:
    path, data = demo
    block = path.read_text(encoding="utf-8").split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert "<" not in block and ">" not in block
    # And the page still parses it back to exactly what was exported.
    assert json.loads(block)["meta"]["run"] == data["meta"]["run"]
