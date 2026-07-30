"""Tests for the per-stage latency instrumentation.

The timing wrappers must observe the pipeline from the outside (no edits to embr/), and
the percentile helper must follow the nearest-rank definition exactly, since the paper
reports p50/p95 from it.
"""

from __future__ import annotations

from embr import build_demo_conversation

from eval.latency import StageTimings, benchmark, instrument, percentile


def test_percentile_nearest_rank_hand_case() -> None:
    # Nearest rank on [10, 20, 30, 40]: p50 -> ceil(0.5 * 4) = rank 2 -> 20,
    # p95 -> ceil(0.95 * 4) = rank 4 -> 40. Order of the input must not matter.
    values = [40.0, 10.0, 30.0, 20.0]
    assert percentile(values, 50) == 20.0
    assert percentile(values, 95) == 40.0
    assert percentile([5.0], 50) == 5.0


def test_percentile_of_an_empty_list_is_zero() -> None:
    # A stage that never ran reports 0.0 rather than crashing the report.
    assert percentile([], 50) == 0.0


def test_percentile_rank_is_float_exact_and_clamped_on_both_sides() -> None:
    # (7/100)*100 is 7.000000000000001 in floats, which ceil would round to rank 8;
    # multiplying before dividing keeps mathematically integral ranks exact.
    values = [float(index) for index in range(1, 101)]
    assert percentile(values, 7) == 7.0
    # p above 100 must land on the largest value instead of running off the list.
    assert percentile([1.0, 2.0], 150) == 2.0


def test_instrument_times_all_three_stages_of_a_turn() -> None:
    from embr import EventType, Memory

    conversation = build_demo_conversation()
    timings = instrument(conversation)
    assert isinstance(timings, StageTimings)

    conversation.take_turn(
        "any news?",
        event=Memory(text="the player asked for news", valence=0.1, arousal=0.1,
                     event_type=EventType.NORMAL),
    )
    # One turn with an event: exactly one write, one retrieval, one model call.
    assert len(timings.write) == 1
    assert len(timings.score_retrieve) == 1
    assert len(timings.model) == 1
    assert all(value >= 0.0 for value in timings.write + timings.score_retrieve + timings.model)


def test_instrument_leaves_a_turn_without_an_event_unwritten() -> None:
    conversation = build_demo_conversation()
    timings = instrument(conversation)
    conversation.take_turn("just chatting")
    assert timings.write == []  # no event, no write stage
    assert len(timings.model) == 1


def test_benchmark_reports_p50_p95_and_sample_counts_per_stage() -> None:
    report = benchmark(build_demo_conversation, turns=6)
    assert set(report) == {"write", "score_retrieve", "model"}
    for stage in report.values():
        assert set(stage) == {"p50", "p95", "count"}
        assert stage["p50"] >= 0.0
        assert stage["p95"] >= stage["p50"]
    # Every third turn writes an event, so 6 turns produce exactly 2 writes but 6
    # retrievals and 6 model calls; the counts prove the wrappers actually fired.
    assert report["write"]["count"] == 2
    assert report["score_retrieve"]["count"] == 6
    assert report["model"]["count"] == 6
    # Two real writes happened, so their p95 is a real (positive) duration.
    assert report["write"]["p95"] > 0.0
