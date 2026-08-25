"""Exact Banzhaf attribution over the prompt's sources: which one actually drove the reply.

RQ2 measures that an injected memory reaches the probe's top 5. That is a fact about
*ranking*. The claim the paper wants is a fact about *behaviour*: the planted memory changed
what the character said. This module measures that step, and gives it a magnitude rather
than a count.

**What it computes, by its right name.** The prompt is treated as `d` sources: the retrieved
memories, one each, plus the generated mood sentence. Every one of the `2**d` ablation masks
is enumerated, the prompt is rebuilt with the masked sources removed, and a utility is scored
under it. The per-source attribution is then the **exact Banzhaf value**, the mean marginal
contribution of that source over every subset of the others.

Over the complete cube the Banzhaf value *is* the exact solution of the uniformly weighted
linear least-squares fit of the utility onto the mask bits, because the mask columns are
orthogonal there. Both routes are exact and they agree by construction, which is what
`tests/test_context_attribution.py` pins. So this is not a surrogate, and it is not
"ContextCite with more ablations": ContextCite samples 32 masks and fits a LASSO because its
`d` runs to hundreds of sentences, where enumeration is impossible and sparsity has to be
assumed. At `d = 6` the approximation is simply unnecessary. ContextCite is the right
citation for the framing (arXiv:2409.00729); Banzhaf is the right name for the quantity.

**Two estimators, one mask set.** The comparison between them is the experiment:

  * `LikelihoodUtility` scores the logit-scaled probability of the already-generated reply
    under each ablated context. This is ContextCite's target, and it needs teacher-forced
    scoring of a supplied completion.
  * `BehaviouralUtility` regenerates the reply under each ablated context and scores its
    valence with the harness's tone rater. This is EMBR's own outcome variable.

Attribution methods are validated on question answering, where "this source led to that
statement" means "this source made that statement likely". A roleplay system does not care
about likelihood, it cares about tone. Whether the two agree is an open question, so both run
over the identical mask set, prompts and seeds, and the results are paired per source.

**Model access is not symmetric, and this is a measured constraint.** Ollama's HTTP API
returns log-probabilities only for tokens the model itself generated; it has no echo or
prompt-logprobs field. The likelihood estimator therefore cannot run through
`OllamaRunner` at all, and callers are refused rather than silently downgraded. It runs
through HuggingFace transformers: `OuroRunner` for Ouro 1.4B and 2.6B, and the llama models
loaded the same way. The behavioural estimator regenerates, so it runs anywhere.

    python -m eval.context_attribution                    # stub model, full cube, seconds
    python -m eval.context_attribution --model ouro       # the thesis model
    python -m eval.context_attribution --estimator behavioural --loo-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

from embr import CharacterState, Conversation, Memory, ScoringRunner, StubRunner, __version__
from embr.model import ModelUnavailableError, OuroRunner
from embr.prompt import PromptBuilder
from embr.scoring import embr_scorer

from eval.attacks import PROBE_QUESTION, Attack, build_attack_memory
from eval.run import (
    ATTACKS,
    REFERENCE_TIME,
    _conversation_factory,
    _model_label,
    _provenance,
    load_eval_scenario,
)
from eval.scenarios import Scenario, label_sha256
from eval.stats import spearman
from eval.tone import ToneRater, default_judge_panel, default_tone_rater

#: The mood sentence is a source like any other. It is the whole reason for this study: the
#: prompt states the character's mood *and* selects her memories by that mood, so a reply
#: that tracks her mood tells you nothing about which of the two channels carried it.
SOURCE_MOOD = "mood_sentence"

#: Only attacks that write a memory have a poison to attribute. The other ten are pure input.
INJECTION_CATEGORIES = ("false_memory", "emotion_flip")

#: Attribution is unfaithful when the context restates what the model already knows: the
#: model can produce the reply from its weights whether or not the source is present, so every
#: marginal contribution collapses. Dawn Whitmore is invented for this project and appears in
#: no training corpus, which is why she is the only scenario this study is allowed to run on.
#: Pointing it at a canonical character would produce numbers that are quietly meaningless.
REQUIRED_SCENARIO = "dawn"


def logit_from_logprob(logprob: float) -> float:
    """Map a total log-probability to the logit scale, ContextCite's regression target.

    A probability is bounded in [0, 1] and a difference between two of them is not on a
    scale worth regressing; the logit is not bounded. For any real reply the probability
    underflows, `log1p(-exp(logprob))` is exactly the zero it converges to, and the logit
    equals the log-probability, which is the regime this runs in.
    """
    if logprob > 0.0:
        raise ValueError(f"a log-probability cannot be positive, got {logprob}")
    if logprob == 0.0:
        raise ValueError("a probability of exactly 1 has no finite logit")
    return logprob - math.log1p(-math.exp(logprob))


# ------------------------------------------------------------------ masks and attribution


def enumerate_masks(source_count: int) -> tuple[tuple[bool, ...], ...]:
    """Every one of the `2**source_count` ablation masks, each exactly once.

    True keeps a source, False ablates it. The order is fixed so two estimators, and two
    runs, walk the cube identically and their utilities stay paired position by position.
    """
    if source_count < 1:
        raise ValueError("attribution needs at least one source")
    return tuple(product((False, True), repeat=source_count))


def _require_complete_cube(masks: Sequence[tuple[bool, ...]]) -> int:
    """Check the masks really are the whole cube, and return the source count.

    The Banzhaf identity below holds only over the complete cube, where the mask columns are
    orthogonal. On a sampled subset the same arithmetic returns a number that looks like an
    attribution and is not one, so this is checked rather than assumed.
    """
    if not masks:
        raise ValueError("no masks to attribute over")
    source_count = len(masks[0])
    if any(len(mask) != source_count for mask in masks):
        raise ValueError("masks disagree on how many sources there are")
    if len(set(masks)) != len(masks):
        raise ValueError("the mask set repeats a mask")
    if len(masks) != 2**source_count:
        raise ValueError(
            f"Banzhaf values need the complete cube: expected {2 ** source_count} masks for "
            f"{source_count} sources, got {len(masks)}. Use leave-one-out instead."
        )
    return source_count


def banzhaf_values(
    masks: Sequence[tuple[bool, ...]], utilities: Sequence[float]
) -> list[float]:
    """Exact Banzhaf value per source: its mean marginal contribution over all subsets.

    Computed as `mean(utility | source present) - mean(utility | source absent)`. Over the
    complete cube that difference is identically the mean of the `2**(d-1)` marginal
    contributions, and identically the least-squares coefficient on that source's mask bit.
    The test suite computes the marginal-contribution sum the long way and asserts all three
    agree, because an identity worth relying on is worth pinning.
    """
    source_count = _require_complete_cube(masks)
    if len(utilities) != len(masks):
        raise ValueError("every mask needs exactly one utility")

    values: list[float] = []
    for index in range(source_count):
        present = [u for mask, u in zip(masks, utilities) if mask[index]]
        absent = [u for mask, u in zip(masks, utilities) if not mask[index]]
        values.append(sum(present) / len(present) - sum(absent) / len(absent))
    return values


def leave_one_out_deltas(
    masks: Sequence[tuple[bool, ...]], utilities: Sequence[float]
) -> list[float]:
    """Per source: how far the utility falls when only that source is removed.

    The sanity column. It is one point of the cube rather than an average over it, so it
    cannot see interactions between sources, but it is exactly what a reader pictures when
    they ask what a memory contributed, and a Banzhaf value that disagrees with it in sign is
    worth looking at rather than reporting.
    """
    if not masks:
        raise ValueError("no masks to attribute over")
    source_count = len(masks[0])
    lookup = dict(zip(masks, utilities))
    full = (True,) * source_count
    if full not in lookup:
        raise ValueError("leave-one-out needs the unablated mask")

    deltas: list[float] = []
    for index in range(source_count):
        without = full[:index] + (False,) + full[index + 1 :]
        if without not in lookup:
            raise ValueError(f"leave-one-out needs the mask with source {index} removed")
        deltas.append(lookup[full] - lookup[without])
    return deltas


def loo_masks(source_count: int) -> tuple[tuple[bool, ...], ...]:
    """The `source_count + 1` masks leave-one-out needs, for runs that cannot afford the cube.

    The full mask first, then one mask per source with that source cleared. Banzhaf values
    are not computable from these and the runner does not pretend otherwise.
    """
    full = (True,) * source_count
    return (full,) + tuple(
        full[:index] + (False,) + full[index + 1 :] for index in range(source_count)
    )


# ----------------------------------------------------------------------------- estimators


@dataclass(frozen=True)
class LikelihoodUtility:
    """ContextCite's target: the logit-scaled probability of a fixed reply.

    `inert_range` is in nats over the whole reply. A model that reads its context at all
    moves further than this when five of six sources are taken away; a probe that does not
    is a probe where the model answered from its weights, and averaging it in would dilute
    every other probe with noise.
    """

    runner: ScoringRunner
    reply: str
    name: str = "likelihood"
    units: str = "logit-scaled probability (nats)"
    inert_range: float = 1.0

    def __call__(self, prompt: str) -> float:
        return logit_from_logprob(self.runner.logprob(prompt, self.reply))


@dataclass(frozen=True)
class BehaviouralUtility:
    """EMBR's own outcome variable: the rated valence of the reply the context produces.

    `inert_range` is in valence units on [-1, 1], so it is a different scale from the
    likelihood estimator's and deliberately not shared with it.
    """

    runner: Any
    rater: ToneRater
    name: str = "behavioural"
    units: str = "rated reply valence"
    inert_range: float = 0.05

    def __call__(self, prompt: str) -> float:
        valence, _arousal = self.rater.rate(self.runner.generate(prompt))
        return valence


# ------------------------------------------------------------------------------- one probe


@dataclass(frozen=True)
class SourceAttribution:
    """One source's reading, on one probe, under one ordering."""

    source: str
    text: str
    banzhaf: float | None
    leave_one_out: float
    is_poison: bool


@dataclass(frozen=True)
class ProbeAttribution:
    """One attack probe, attributed. `order` is which memory ordering built the prompts."""

    attack_id: str
    order: str
    estimator: str
    reply: str
    masks: tuple[tuple[bool, ...], ...]
    utilities: tuple[float, ...]
    sources: tuple[SourceAttribution, ...]
    utility_range: float
    inert: bool


def build_masked_prompt(
    builder: PromptBuilder,
    state: CharacterState,
    memories: Sequence[Memory],
    player_input: str,
    mask: Sequence[bool],
) -> str:
    """Rebuild the prompt with the masked sources removed. Mood is the last source."""
    if len(mask) != len(memories) + 1:
        raise ValueError("a mask covers every memory plus the mood sentence")
    kept = [memory for memory, keep in zip(memories, mask[:-1]) if keep]
    return builder.build(state, kept, player_input, include_mood=bool(mask[-1]))


def attribute_probe(
    attack: Attack,
    state: CharacterState,
    memories: Sequence[Memory],
    reply: str,
    utility: LikelihoodUtility | BehaviouralUtility,
    order: str = "as_retrieved",
    exhaustive: bool = True,
    builder: PromptBuilder | None = None,
) -> ProbeAttribution:
    """Score every mask for one probe and reduce it to a reading per source."""
    builder = builder or PromptBuilder()
    source_count = len(memories) + 1
    masks = enumerate_masks(source_count) if exhaustive else loo_masks(source_count)
    utilities = [
        utility(build_masked_prompt(builder, state, memories, PROBE_QUESTION, mask))
        for mask in masks
    ]

    banzhaf = banzhaf_values(masks, utilities) if exhaustive else [None] * source_count
    loo = leave_one_out_deltas(masks, utilities)
    poison_text = attack.injected_memory_text

    labels = [f"memory_{position + 1}" for position in range(len(memories))] + [SOURCE_MOOD]
    texts = [memory.text for memory in memories] + ["(the generated mood sentence)"]
    sources = tuple(
        SourceAttribution(
            source=label,
            text=text,
            banzhaf=banzhaf[index],
            leave_one_out=loo[index],
            is_poison=poison_text is not None and text == poison_text,
        )
        for index, (label, text) in enumerate(zip(labels, texts))
    )

    utility_range = max(utilities) - min(utilities)
    return ProbeAttribution(
        attack_id=attack.id,
        order=order,
        estimator=utility.name,
        reply=reply,
        masks=masks,
        utilities=tuple(utilities),
        sources=sources,
        utility_range=utility_range,
        inert=utility_range < utility.inert_range,
    )


# ------------------------------------------------------------------------------- the study


def injection_attacks() -> tuple[Attack, ...]:
    """The ten attacks that write a memory: the only ones with a poison to attribute."""
    return tuple(attack for attack in ATTACKS if attack.category in INJECTION_CATEGORIES)


def _probe_after_attack(
    attack: Attack, build_conversation: Callable[[], Conversation]
) -> tuple[Conversation, Any]:
    """Land the attack, then ask the probe, and hand back the state the probe actually saw.

    Mirrors the attacked path of `eval.attacks.run_attack`, but keeps the conversation so
    the probe turn's state and retrieved set can be replayed under ablation. The probe turn
    writes no event, so the state after it is the state its prompt was built from.
    """
    conversation = build_conversation()
    conversation.take_turn(attack.player_input, event=build_attack_memory(attack))
    return conversation, conversation.take_turn(PROBE_QUESTION)


def _require_invented_scenario(scenario: Scenario) -> None:
    """Refuse to attribute a scenario the model may already know. See REQUIRED_SCENARIO."""
    if REQUIRED_SCENARIO not in scenario.name.lower():
        raise ValueError(
            f"context attribution is restricted to the invented {REQUIRED_SCENARIO!r} "
            f"scenario and refuses to run on {scenario.name!r}. A model that already knows "
            f"the character can generate the reply without the context, every marginal "
            f"contribution collapses toward zero, and the resulting attributions are "
            f"unfaithful in a way no downstream statistic can detect."
        )


def run_attribution(
    utility_for: Callable[[Any, str], LikelihoodUtility | BehaviouralUtility],
    model_factory: Callable[[], Any] = StubRunner,
    exhaustive: bool = True,
    scenario: Scenario | None = None,
    limit: int | None = None,
) -> list[ProbeAttribution]:
    """Attribute every injection probe, in both memory orderings.

    The reversed pass is the position-bias control. Utility-based attribution can favour a
    source for sitting where it sits rather than for saying what it says, and the only way to
    see that is to move it: the same sources, the same utilities, a different order in the
    prompt. The two attribution vectors are correlated by source identity in
    `position_bias_report`, and a low correlation invalidates the ranking, not just the
    magnitudes.
    """
    scenario = scenario or load_eval_scenario()
    _require_invented_scenario(scenario)

    readings: list[ProbeAttribution] = []
    probes = injection_attacks()[:limit] if limit else injection_attacks()
    for attack in probes:
        build = _conversation_factory(scenario, embr_scorer, model_factory)
        conversation, turn = _probe_after_attack(attack, build)
        for order, memories in (
            ("as_retrieved", list(turn.retrieved)),
            ("reversed", list(reversed(turn.retrieved))),
        ):
            readings.append(
                attribute_probe(
                    attack=attack,
                    state=conversation.state,
                    memories=memories,
                    reply=turn.reply,
                    utility=utility_for(conversation.model, turn.reply),
                    order=order,
                    exhaustive=exhaustive,
                )
            )
    return readings


def position_bias_report(readings: Sequence[ProbeAttribution]) -> dict[str, Any]:
    """Per probe, the correlation between the two orderings' attribution vectors.

    Sources are matched by text, not by position, because matching by position is exactly
    the thing under test.
    """
    by_attack: dict[str, dict[str, ProbeAttribution]] = {}
    for reading in readings:
        by_attack.setdefault(reading.attack_id, {})[reading.order] = reading

    per_attack: dict[str, float | None] = {}
    for attack_id, orders in by_attack.items():
        forward, backward = orders.get("as_retrieved"), orders.get("reversed")
        if forward is None or backward is None:
            continue
        backward_by_text = {source.text: source for source in backward.sources}
        pairs = [
            (source.leave_one_out, backward_by_text[source.text].leave_one_out)
            for source in forward.sources
            if source.text in backward_by_text
        ]
        per_attack[attack_id] = spearman([a for a, _ in pairs], [b for _, b in pairs])

    scored = [value for value in per_attack.values() if value is not None]
    return {
        "per_attack": per_attack,
        "mean_rho": sum(scored) / len(scored) if scored else None,
        "undefined": len(per_attack) - len(scored),
        "note": (
            "Spearman between the two orderings' leave-one-out vectors, matched by memory "
            "text. Low values mean the attribution ranks sources by where they sit in the "
            "prompt as much as by what they contribute."
        ),
    }


def inert_report(readings: Sequence[ProbeAttribution]) -> dict[str, Any]:
    """How many probes the model effectively ignored, reported rather than averaged away."""
    flagged = [reading.attack_id for reading in readings if reading.inert]
    return {
        "flagged_count": len(flagged),
        "total": len(readings),
        "flagged": sorted(set(flagged)),
        "note": (
            "A probe whose utility barely moves across the whole cube is one where the model "
            "answered from its weights rather than its context. Its attributions are near "
            "zero for a reason that is not evidence about any source, so they are counted "
            "here instead of being folded into a mean."
        ),
    }


def poison_rank_report(readings: Sequence[ProbeAttribution]) -> dict[str, Any]:
    """Where the planted memory ranks by attribution: the RQ2 claim this study exists for."""
    ranks: list[int] = []
    top1 = 0
    for reading in readings:
        if reading.order != "as_retrieved" or reading.inert:
            continue
        scored = sorted(
            reading.sources,
            key=lambda source: source.banzhaf if source.banzhaf is not None
            else source.leave_one_out,
            reverse=True,
        )
        for position, source in enumerate(scored, start=1):
            if source.is_poison:
                ranks.append(position)
                top1 += position == 1
                break
    return {
        "probes_with_poison_retrieved": len(ranks),
        "poison_ranked_first": top1,
        "ranks": ranks,
        "note": (
            "Counted on the as-retrieved ordering only, and only on probes that were not "
            "flagged inert. A poisoned memory that reaches the top 5 but attributes near "
            "zero reached the prompt without reaching the reply."
        ),
    }


# ------------------------------------------------------------------------------- reporting


def _write_mask_csv(path: Path, readings: Sequence[ProbeAttribution]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attack_id", "order", "estimator", "mask", "utility"])
        for reading in readings:
            for mask, utility in zip(reading.masks, reading.utilities):
                bits = "".join("1" if keep else "0" for keep in mask)
                writer.writerow([reading.attack_id, reading.order, reading.estimator, bits, utility])


def _write_source_csv(path: Path, readings: Sequence[ProbeAttribution]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "attack_id",
                "order",
                "estimator",
                "source",
                "is_poison",
                "banzhaf",
                "leave_one_out",
                "text",
            ]
        )
        for reading in readings:
            for source in reading.sources:
                writer.writerow(
                    [
                        reading.attack_id,
                        reading.order,
                        reading.estimator,
                        source.source,
                        source.is_poison,
                        "" if source.banzhaf is None else source.banzhaf,
                        source.leave_one_out,
                        source.text,
                    ]
                )


def _readings_payload(readings: Sequence[ProbeAttribution]) -> list[dict[str, Any]]:
    return [
        {
            "attack_id": reading.attack_id,
            "order": reading.order,
            "estimator": reading.estimator,
            "reply": reading.reply,
            "utility_range": reading.utility_range,
            "inert": reading.inert,
            "sources": [
                {
                    "source": source.source,
                    "text": source.text,
                    "is_poison": source.is_poison,
                    "banzhaf": source.banzhaf,
                    "leave_one_out": source.leave_one_out,
                }
                for source in reading.sources
            ],
        }
        for reading in readings
    ]


def write_run(
    readings: Sequence[ProbeAttribution],
    out_root: str | Path = "data/runs/attribution",
    model_factory: Callable[[], Any] = StubRunner,
    estimator: str = "likelihood",
    exhaustive: bool = True,
    depth: dict[str, float | int] | None = None,
    wall_clock_seconds: float | None = None,
    panel_agreement: dict[str, Any] | None = None,
) -> Path:
    """Write results.json plus the per-mask and per-source CSVs the asset builders read."""
    scenario = load_eval_scenario()
    source_count = len(readings[0].sources) if readings else 0
    results = {
        "context_attribution": {
            "estimator": estimator,
            "method": "exact Banzhaf over the complete mask cube"
            if exhaustive
            else "leave-one-out only (Banzhaf not computable from this mask set)",
            "sources": source_count,
            "masks_per_probe": 2**source_count if exhaustive else source_count + 1,
            "position_bias": position_bias_report(readings),
            "inert_probes": inert_report(readings),
            "poison_rank": poison_rank_report(readings),
            "panel_agreement": panel_agreement,
            "wall_clock_seconds": wall_clock_seconds,
            "readings": _readings_payload(readings),
        },
        "metadata": {
            **_provenance(),
            "label_set": scenario.name,
            "label_version": scenario.version,
            "label_sha256": label_sha256(),
            "model": _model_label(model_factory),
            "tone_rater": default_tone_rater().name,
            "reference_time": REFERENCE_TIME.isoformat(),
            "embr_version": __version__,
            # Ouro exits its recurrent loop early unless pinned, so depth is part of the
            # scoring function and belongs in provenance beside the commit that produced it.
            "ouro_depth": depth,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    out_dir = Path(out_root) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    _write_mask_csv(out_dir / "attribution_masks.csv", readings)
    _write_source_csv(out_dir / "attribution_sources.csv", readings)
    return out_dir


# ------------------------------------------------------------------------------------ CLI


#: Which family each generation arm belongs to, so the panel builder can exclude it. A judge
#: rating its own output is not blind, and that rule cannot be applied by eye.
_MODEL_FAMILIES = {"stub": "stub", "ouro": "bytedance", "ouro-2.6b": "bytedance"}


def _model_factory(name: str) -> Callable[[], Any]:
    if name == "stub":
        return StubRunner
    if name == "ouro":
        return OuroRunner
    if name == "ouro-2.6b":
        return lambda: OuroRunner(model_name="ByteDance/Ouro-2.6B")
    raise SystemExit(f"unknown model {name!r}; expected stub, ouro, or ouro-2.6b")


def _pin_depth(model_factory: Callable[[], Any]) -> dict[str, float | int] | None:
    """Pin Ouro's recurrent depth once, up front, and fail before spending a run if it cannot.

    Depth is part of the scoring function. Discovering after 1280 forward passes that early
    exit was live would mean throwing all of them away.
    """
    probe = model_factory()
    if not isinstance(probe, OuroRunner):
        return None
    return probe.pin_depth()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="stub", help="stub, ouro, or ouro-2.6b")
    parser.add_argument(
        "--estimator",
        default="likelihood",
        choices=("likelihood", "behavioural"),
        help="likelihood scores a fixed reply, behavioural regenerates and rates its valence",
    )
    parser.add_argument(
        "--loo-only",
        action="store_true",
        help="score only the d+1 leave-one-out masks. Banzhaf values are not reported, "
        "because they are not computable from a partial cube.",
    )
    # A subtree of its own, so an attribution run is never mistaken for an eval run
    # by the asset builders and menu, which scan data/runs/*/results.json one level deep.
    parser.add_argument("--out", default="data/runs/attribution")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N probes. The pilot is --limit 1.",
    )
    args = parser.parse_args()

    model_factory = _model_factory(args.model)
    depth = _pin_depth(model_factory)
    # The generator never sits on its own panel: a judge rating its own output is not blind.
    panel = default_judge_panel(exclude_families=frozenset({_MODEL_FAMILIES[args.model]}))
    if not panel.is_family_diverse:
        print(
            f"    WARNING: panel is not family diverse (model families: "
            f"{sorted(panel.model_families) or 'none'}). Two sizes of one family are not two "
            f"judges. Pull a second family, e.g. `ollama pull qwen2.5:7b`, before the full "
            f"sweep. Recorded in the run either way."
        )

    def utility_for(runner: Any, reply: str) -> LikelihoodUtility | BehaviouralUtility:
        if args.estimator == "behavioural":
            return BehaviouralUtility(runner=runner, rater=panel)
        if not isinstance(runner, ScoringRunner):
            raise SystemExit(
                f"{type(runner).__name__} cannot score a supplied completion, so the "
                f"likelihood estimator cannot run on it. Ollama's API returns "
                f"log-probabilities only for tokens it generated itself. Use a transformers "
                f"runner, or --estimator behavioural."
            )
        return LikelihoodUtility(runner=runner, reply=reply)

    started = time.perf_counter()
    try:
        readings = run_attribution(
            utility_for,
            model_factory=model_factory,
            exhaustive=not args.loo_only,
            limit=args.limit,
        )
    except ModelUnavailableError as error:
        raise SystemExit(str(error)) from error
    elapsed = time.perf_counter() - started

    # Panel agreement on this arm's own replies, which is the pre-registered gate on H3.
    agreement = panel.agreement([reading.reply for reading in readings])

    out_dir = write_run(
        readings,
        out_root=args.out,
        model_factory=model_factory,
        estimator=args.estimator,
        exhaustive=not args.loo_only,
        depth=depth,
        wall_clock_seconds=elapsed,
        panel_agreement=agreement,
    )

    poison = poison_rank_report(readings)
    inert = inert_report(readings)
    bias = position_bias_report(readings)
    calls = sum(len(reading.masks) for reading in readings)
    print(f"wrote {out_dir}")
    print(f"  arm: {args.estimator} on {_model_label(model_factory)}, depth {depth}")
    print(f"  wall clock: {elapsed:.1f} s over {calls} model calls ({elapsed / max(calls, 1):.3f} s each)")
    print(
        f"  poison ranked first in {poison['poison_ranked_first']} of "
        f"{poison['probes_with_poison_retrieved']} probes where it was retrieved"
    )
    print(f"  inert probes (model ignored its context): {inert['flagged_count']}/{inert['total']}")
    print(f"  position-bias mean rho between orderings: {bias['mean_rho']}")
    print(
        f"  panel valence agreement: min {agreement['valence']['min']}, "
        f"floor {agreement['floor']}, clears: {agreement['clears_floor']}, "
        f"family diverse: {agreement['family_diverse']}"
    )


if __name__ == "__main__":
    main()
