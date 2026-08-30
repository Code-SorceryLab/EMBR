"""The behavioural consistency check: does what she *does* track what happened to her?

A tone score says whether a reply sounds warm or cold. A player does not experience a tone
score. A player asks for the discounted room again and is told yes or no, and whether that
answer tracks the betrayal they committed is what believability actually is. This is the
task-like proxy for that: one request, asked at three points in the pre-registered arc, and
the answer should move with the arc.

Binary, scriptable, no raters. The prompt asks for yes or no before anything else and the
first word is parsed. The stub cannot answer it, and says so in the output rather than
scoring a phantom no.

Two readings per point, because they answer different questions:

  * **retrieval**: is a memory from the relevant beat in the top 5 when the request is made?
    Model-free and exact. This is whether the memory layer *offered* the model the fact.
  * **behaviour**: did the model's yes or no track the arc? This is whether it *used* it.

A retrieval yes with a behaviour no is the same shape as RQ1's null on Ouro: the loop closes
to retrieval and not to generation. Reporting them side by side is the point.

    python -m eval.consistency                  # stub: retrieval only, behaviour unparseable
    python -m eval.consistency --model ouro
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embr import Conversation, DeterministicEmbedder, MemoryStore, StubRunner, embr_scorer
from embr.model import OuroRunner
from embr.prompt import PromptBuilder

from eval.run import _eval_clock, _model_label, _provenance, load_eval_scenario
from eval.scenarios import Scenario, dawn_state, label_sha256
from eval.tuning import _session_index_of

#: The one request. It is what the founding lie bought, so it is the thing a betrayal should
#: cost. Phrased to share words with the room memories, so relevance has something to bite.
REQUEST = "Could I have the discounted room again tonight, same rate as before?"


@dataclass(frozen=True)
class ArcPoint:
    """Where in the arc the request is made, and what a consistent keeper answers."""

    name: str
    after_session: int
    expected: str  # "yes" or "no"
    beat_memory_ids: frozenset[int]  # memories that should be on the table at this point
    why: str


#: Session indices follow the label file: 0 the arrival and the lie, 1 the warm middle,
#: 2 the first doubts, 3 the confrontation, 4 the confession and repayment.
ARC: tuple[ArcPoint, ...] = (
    ArcPoint(
        name="before_exposure",
        after_session=1,
        expected="yes",
        beat_memory_ids=frozenset({1, 2}),
        why="She believes the errand and has already extended the rate once.",
    ),
    ArcPoint(
        name="after_betrayal",
        after_session=3,
        expected="no",
        beat_memory_ids=frozenset({15, 16, 17}),
        why="The story fell apart and she said the discount was never about the coin.",
    ),
    ArcPoint(
        name="after_amends",
        after_session=4,
        expected="no",
        beat_memory_ids=frozenset({20, 21, 22}),
        why="Forgiven, but she said she would not be quick to vouch again. The rate was "
        "vouching. A yes here would be forgetting, not forgiving.",
    ),
)

_YES = re.compile(r"^\W*(yes|aye|of course|certainly|gladly)\b", re.IGNORECASE)
_NO = re.compile(r"^\W*(no|nay|not|never|i cannot|i can't|i won't|i will not)\b", re.IGNORECASE)


class YesNoPromptBuilder(PromptBuilder):
    """The standard prompt, with the answer format pinned so the first word is parseable."""

    def build(self, state, memories, player_input, *, include_mood: bool = True) -> str:
        base = super().build(state, memories, player_input, include_mood=include_mood)
        return base + " Begin your reply with the single word Yes or No, then say why."


def parse_answer(reply: str) -> str | None:
    """'yes', 'no', or None when the model did not start with either."""
    if _YES.match(reply):
        return "yes"
    if _NO.match(reply):
        return "no"
    return None


@dataclass(frozen=True)
class PointResult:
    point: str
    expected: str
    answer: str | None
    behaviour_consistent: bool | None  # None when unparseable
    beat_retrieved: bool
    retrieved_ids: tuple[int, ...]
    reply: str


def _conversation_at(
    scenario: Scenario, after_session: int, model: Any, builder: PromptBuilder
) -> tuple[Conversation, dict[int, int]]:
    """Dawn holding only the sessions up to `after_session`, in the eval's pinned frame.

    Returns the store's id map back to scenario indices, since the store renumbers on add.
    """
    embedder = DeterministicEmbedder()
    session_of = _session_index_of(scenario)
    store = MemoryStore(embedder=embedder)
    id_map: dict[int, int] = {}
    for memory in scenario.memories:
        if session_of[memory.timestamp] <= after_session:
            stored = store.add(replace(memory))
            id_map[stored.id] = memory.id
    conversation = Conversation(
        state=dawn_state(scenario),
        store=store,
        scorer=embr_scorer(embedder=embedder, now=_eval_clock),
        prompt_builder=builder,
        model=model,
        top_k=5,
    )
    return conversation, id_map


def run_consistency(
    model_factory: Callable[[], Any] = StubRunner, scenario: Scenario | None = None
) -> list[PointResult]:
    scenario = scenario or load_eval_scenario()
    builder = YesNoPromptBuilder()
    results: list[PointResult] = []
    for point in ARC:
        conversation, id_map = _conversation_at(
            scenario, point.after_session, model_factory(), builder
        )
        turn = conversation.take_turn(REQUEST)
        retrieved = tuple(id_map[m.id] for m in turn.retrieved)
        answer = parse_answer(turn.reply)
        results.append(
            PointResult(
                point=point.name,
                expected=point.expected,
                answer=answer,
                behaviour_consistent=None if answer is None else answer == point.expected,
                beat_retrieved=bool(point.beat_memory_ids & set(retrieved)),
                retrieved_ids=retrieved,
                reply=turn.reply,
            )
        )
    return results


def summarise(results: list[PointResult]) -> dict[str, Any]:
    parsed = [r for r in results if r.behaviour_consistent is not None]
    return {
        "points": len(results),
        "beat_retrieved": sum(r.beat_retrieved for r in results),
        "behaviour_parsed": len(parsed),
        "behaviour_consistent": sum(r.behaviour_consistent for r in parsed),
        # The arc is one thing. Consistent at every point, or not consistent.
        "arc_consistent": bool(parsed) and len(parsed) == len(results)
        and all(r.behaviour_consistent for r in parsed),
    }


def write_run(
    results: list[PointResult],
    out_root: str | Path = "data/experiments",
    model_factory: Callable[[], Any] = StubRunner,
) -> Path:
    scenario = load_eval_scenario()
    payload = {
        "consistency": {
            "request": REQUEST,
            "summary": summarise(results),
            "points": [
                {
                    "point": r.point,
                    "expected": r.expected,
                    "answer": r.answer,
                    "behaviour_consistent": r.behaviour_consistent,
                    "beat_retrieved": r.beat_retrieved,
                    "retrieved_ids": list(r.retrieved_ids),
                    "reply": r.reply,
                }
                for r in results
            ],
        },
        "metadata": {
            **_provenance(),
            "label_set": scenario.name,
            "label_sha256": label_sha256(),
            "model": _model_label(model_factory),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    out = Path(out_root) / "consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="stub", help="stub or ouro")
    parser.add_argument("--out", default="data/experiments")
    args = parser.parse_args()
    factory = {"stub": StubRunner, "ouro": OuroRunner}[args.model]

    results = run_consistency(factory)
    out = write_run(results, args.out, factory)
    summary = summarise(results)
    print(f"wrote {out}")
    print(f"  {'point':18s} {'expect':>6s} {'answer':>8s} {'beat in top5':>13s}")
    for r in results:
        answer = r.answer or "unparseable"
        print(f"  {r.point:18s} {r.expected:>6s} {answer:>8s} {str(r.beat_retrieved):>13s}")
    print(
        f"  retrieval offered the beat at {summary['beat_retrieved']}/{summary['points']} points; "
        f"behaviour consistent at {summary['behaviour_consistent']}/{summary['behaviour_parsed']} "
        f"parsed; arc consistent: {summary['arc_consistent']}"
    )


if __name__ == "__main__":
    main()
