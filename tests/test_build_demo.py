"""Tests for the interactive demo.

The demo computes four of the five signals in the browser, which means its arithmetic can
drift from `embr/scoring.py` without anything failing. A demo whose numbers have quietly
drifted is worse than no demo, because it looks like evidence. So the page ships rankings
the real Python scorer produced and replays them on load, and the decisive test here runs
the page's own JavaScript under Node and checks that replay actually passes.

The Node test skips when Node is absent; every other test runs anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from assets.build_demo import PRESETS, build_demo


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> list[Path]:
    """Both pages, built once from the same payload."""
    return build_demo(out_dir=tmp_path_factory.mktemp("demo"))


@pytest.fixture(scope="module")
def demo(built) -> tuple[Path, dict]:
    """The flat diagram, plus the payload that was inlined into it."""
    path = built[0]
    html = path.read_text(encoding="utf-8")
    payload = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    return path, payload


@pytest.fixture(scope="module")
def brain3d(built) -> Path:
    """The 3D page. Skips rather than fails when three.js has not been vendored."""
    if len(built) < 2:
        pytest.skip("the 3D template is absent")
    return built[1]


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


def test_every_guided_step_declares_the_weights_it_is_describing() -> None:
    """A tour step that only patches what its caption mentions inherits the step before it.

    The back button and "skip to the end" both jump, so a step that leaves the weights to
    whoever ran last can arrive with every signal at zero and then narrate an empty prompt
    as though the attack had been repelled. Each step declares its whole state instead, and
    the weights are the field that broke, so the weights are the field pinned here.
    """
    template = (Path("assets") / "demo" / "template.html").read_text(encoding="utf-8")
    body = template.split("const TOUR = [", 1)[1].split("\n];", 1)[0]
    steps = re.findall(r"set: \{(.*?)\},\n", body, re.S)
    assert len(steps) >= 8, "the tour lost its steps, or its shape changed"
    for index, step in enumerate(steps):
        assert "weights" in step, f"tour step {index + 1} does not say what the weights are"


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


# --------------------------------------------------------------------------------------- 3D
# The 3D page re-implements the same five signals a second time, which is a second place for
# the arithmetic to drift away from `embr/scoring.py` while still looking like evidence.

THREE_SHA256 = "8a5f7249903b54d30f79f708699d2fed2d6a1d0741a4cd41377d1f01bb5a2271"


def _last_script(html: str) -> str:
    """The page's own code, which is the last script block; the first one is the library."""
    return html.rsplit("<script>", 1)[1].split("</script>", 1)[0]


def test_the_vendored_renderer_is_the_bytes_we_recorded() -> None:
    """Vendored third-party code that nobody can check is just code of unknown origin."""
    from assets.build_demo import VENDORED_THREE

    if not VENDORED_THREE.exists():
        pytest.skip("three.js has not been vendored")
    digest = hashlib.sha256(VENDORED_THREE.read_bytes()).hexdigest()
    assert digest == THREE_SHA256, "assets/vendor/three.min.js is not the recorded r149 build"


def test_the_3d_page_reaches_no_network(brain3d: Path) -> None:
    html = brain3d.read_text(encoding="utf-8")
    assert not re.search(r"<(script|link|img)[^>]*(src|href)=\"https?:", html)
    # three.js carries a fetch-based loader we never call, so the check has to be on our code.
    assert "fetch(" not in _last_script(html)
    assert "XMLHttpRequest" not in _last_script(html)


def _code_only(text: str) -> str:
    """Whitespace and line comments stripped, so the comparison is of code and not of prose.

    Neither function here contains a string holding `//`, which is the case that would make
    this too crude.
    """
    lines = (line.split("//", 1)[0] for line in text.splitlines())
    return "".join("".join(line.split()) for line in lines)


def test_the_3d_page_scores_exactly_like_the_flat_one(brain3d: Path, demo) -> None:
    """Two copies of the same arithmetic is one copy too many to leave unchecked."""
    flat, _ = demo
    for name in ("function cosine", "function signals", "function rank"):
        a = _last_script(flat.read_text(encoding="utf-8")).split(name, 1)[1].split("\n}", 1)[0]
        b = _last_script(brain3d.read_text(encoding="utf-8")).split(name, 1)[1].split("\n}", 1)[0]
        assert _code_only(a) == _code_only(b), f"{name} has drifted between the two demo pages"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_the_3d_pages_own_javascript_reproduces_the_python_scorer(brain3d: Path, tmp_path: Path) -> None:
    core = _last_script(brain3d.read_text(encoding="utf-8")).split("// State", 1)[0]
    runner = tmp_path / "conformance3d.mjs"
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
  if (got !== check.top5.join(",")) failures.push(`${check.query}/${check.mood}: ${got}`);
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
    assert report["checks"] >= 12 and report["failures"] == [], report["failures"]


def test_the_planted_memory_usually_scores_no_relevance_at_all(demo) -> None:
    """The claim findings.md 2.4 makes, pinned to the data it was read off.

    Both halves matter and the second is the one that keeps the first honest: the injection
    scores a true zero against the probe in most attacks, and so does about half the corpus,
    so the zero is not what makes the injection special. What makes it special is that the
    tie relevance leaves behind is settled by the four signals the attacker can reach.
    """
    _, data = demo
    poison = [attack["relevance"]["poison"] for attack in data["attacks"]]
    assert sum(1 for value in poison if value == 0.0) >= 7, poison

    store = set(data["store"])
    genuine = [value for attack in data["attacks"]
               for key, value in attack["relevance"].items() if key in store]
    share = sum(1 for value in genuine if value == 0.0) / len(genuine)
    assert 0.4 <= share <= 0.7, f"a zero is {share:.0%} of genuine memories, not about half"
