"""Per-stage latency instrumentation for the RQ2 overhead numbers.

The paper reports how many milliseconds EMBR adds to a game turn, split by stage: the
memory write, the score-and-retrieve pass, and the model call. The pipeline itself must
not grow timing code (the core stays measurement-free), so `instrument` wraps the three
stage entry points of an existing `Conversation` from the outside and records durations
as they happen. Percentiles use the nearest-rank definition, the plain one a reader can
recompute by hand from the raw lists.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from embr import Conversation, Memory


@dataclass
class StageTimings:
    """Raw per-call durations in milliseconds, one list per pipeline stage."""

    write: list[float] = field(default_factory=list)  # store.add
    score_retrieve: list[float] = field(default_factory=list)  # scorer.top_k
    model: list[float] = field(default_factory=list)  # model.generate


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile of `values`; an empty list reports 0.0.

    A stage that never ran (say, no writes happened) should read as zero overhead in the
    report rather than crash it, hence the 0.0. Rank is ceil(p * n / 100) on an ascending
    sorted copy; multiplying before dividing keeps mathematically integral ranks exact in
    floats, and the two-sided clamp lands p=0 on the smallest value and p>100 on the largest.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(p * len(ordered) / 100)))
    return ordered[rank - 1]


def _timed(stage: Callable, bucket: list[float]) -> Callable:
    """Wrap one stage callable so every call appends its duration (ms) to `bucket`."""

    @functools.wraps(stage)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        result = stage(*args, **kwargs)
        bucket.append((time.perf_counter() - started) * 1000.0)
        return result

    return wrapper


def instrument(conversation: Conversation) -> StageTimings:
    """Attach timing wrappers to a conversation's three stages; returns the live timings.

    The wrappers shadow the bound methods on the individual store/scorer/model instances,
    so only this one conversation is observed and `embr/` itself stays untouched.
    """
    timings = StageTimings()
    conversation.store.add = _timed(conversation.store.add, timings.write)
    conversation.scorer.top_k = _timed(conversation.scorer.top_k, timings.score_retrieve)
    conversation.model.generate = _timed(conversation.model.generate, timings.model)
    return timings


# A small rotation of neutral lines, so the benchmark exercises varied queries without
# injecting any affect that would move the character's state between turns.
_NEUTRAL_INPUTS = (
    "any news around the tavern tonight?",
    "how fares the road to the capital?",
    "what is good on the menu today?",
    "quiet evening so far, is it not?",
)


def benchmark(
    build_conversation: Callable[[], Conversation], turns: int = 100
) -> dict[str, dict[str, float]]:
    """Drive `turns` turns through a fresh instrumented conversation; report p50/p95 (ms)
    plus the sample count per stage, so a reader can verify the wrappers actually fired.

    Every third turn also writes a small event, so the write stage gets samples while the
    store still grows at a game-like trickle rather than one write per line of dialogue.
    """
    conversation = build_conversation()
    timings = instrument(conversation)
    for turn_index in range(turns):
        event = None
        if turn_index % 3 == 2:  # every third turn carries a memory-worthy event
            event = Memory(
                text=f"the player made small talk at the bar (turn {turn_index + 1})",
                valence=0.05,
                arousal=0.1,
            )
        conversation.take_turn(_NEUTRAL_INPUTS[turn_index % len(_NEUTRAL_INPUTS)], event=event)
    return {
        stage: {"p50": percentile(values, 50), "p95": percentile(values, 95), "count": len(values)}
        for stage, values in vars(timings).items()
    }
