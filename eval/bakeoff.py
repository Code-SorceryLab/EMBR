"""Model bake-off: hold everything constant except the model, then measure what moves.

The comparison the thesis needs is looped against conventional: Ouro repeats the same
internal computation several times per token instead of stacking layers, so the question is
what that buys and what it costs. Cloud models are the quality ceiling, not competitors.

Every arm sees the same prompts, the same memories, the same retrieval and the same
sampling settings, so the model is the only thing that varies. Four readings per arm:

  * **latency**, percentiles over the per turn wall clock, because the thesis claims a
    roughly 600 ms budget and that claim is either met or it is not,
  * **memory grounding**, whether the reply actually used a memory it was handed, which is
    the whole point of retrieval and the thing a fluent model can fake by ignoring it,
  * **mood responsiveness**, the spread in rated valence across the pinned mood conditions,
    since a model that answers identically in every mood makes the affect signal inert,
  * **persona breaks**, replies that step outside the character.

A full `eval.run` per model is not affordable here: it makes hundreds of generations, and
at Ouro's measured throughput that is hours per arm. This runs a fixed probe set instead,
which is what makes an arm comparable rather than merely cheap.

Transcripts are saved for every arm, because these metrics are proxies and a human reading
ten replies will see things no rater catches.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from embr import Conversation, DeterministicEmbedder, MemoryStore, embr_scorer

from eval.run import _eval_clock, load_eval_scenario
from eval.scenarios import Scenario, dawn_state
from eval.tone import default_tone_rater

#: Phrases that mean the model stopped being the character. Deliberately short and literal:
#: a clever detector would need its own validation, and these are the failures that matter.
PERSONA_BREAK_MARKERS = (
    "as an ai",
    "as a language model",
    "i'm an ai",
    "i am an ai",
    "language model",
    "i cannot fulfill",
    "i can't fulfill",
    "openai",
    "anthropic",
    "assistant",
    "system prompt",
)

#: Words too common to prove a reply used a memory rather than merely sharing English.
_STOPWORDS = frozenset(
    "the a an and or but if then than that this these those of to in on at by for with "
    "from as is are was were be been being do does did have has had i you he she it we "
    "they me him her them my your his its our their not no yes so very just about into "
    "over under out up down what when where who whom which how why all any both each".split()
)

#: How many characters of a reply to keep in the transcript. Enough to judge voice.
TRANSCRIPT_CHARS = 400


@dataclass(frozen=True)
class Arm:
    """One model under test, and how to build it.

    A factory rather than an instance so a model that cannot be constructed here fails as
    a recorded unavailable arm instead of an import error that takes the whole run down.
    """

    name: str
    build: Callable[[], Any] = field(repr=False)
    kind: str = "conventional"  # "looped" for Ouro, the arm the thesis is actually about


@dataclass
class TurnRecord:
    """One probe turn against one model."""

    condition: str
    query: str
    reply: str
    latency_ms: float
    valence: float
    arousal: float
    grounded: bool
    persona_break: bool


def _content_words(text: str) -> set[str]:
    """Lowercased words worth matching on, so grounding is not satisfied by 'the'."""
    words = "".join(character if character.isalnum() else " " for character in text.lower())
    return {word for word in words.split() if len(word) > 3 and word not in _STOPWORDS}


def is_grounded(reply: str, memory_texts: list[str], minimum_overlap: int = 2) -> bool:
    """Whether the reply visibly used one of the memories it was given.

    Overlap of content words against any single memory, not against the pooled set: sharing
    one word with each of five memories is not evidence of using any of them. Two words is
    a low bar on purpose, because this is a screen for models that ignore the memory block
    entirely, not a semantic entailment check.
    """
    reply_words = _content_words(reply)
    return any(
        len(reply_words & _content_words(memory)) >= minimum_overlap for memory in memory_texts
    )


def has_persona_break(reply: str) -> bool:
    """Whether the reply stepped out of character in a way a player would notice."""
    lowered = reply.lower()
    return any(marker in lowered for marker in PERSONA_BREAK_MARKERS)


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, matching how eval.latency reports its own numbers."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def _probe_turns(scenario: Scenario, queries_per_condition: int) -> list[tuple[str, str]]:
    """The fixed probe set: the same queries under each pinned mood, in a stable order.

    Crossing queries with moods is what makes mood responsiveness measurable at all: the
    same question asked in three moods is the only way to see whether the model responds
    to state rather than just to the question.
    """
    queries = [query.query for query in scenario.queries[:queries_per_condition]]
    return [
        (condition, query) for condition in scenario.mood_conditions for query in queries
    ]


def run_arm(
    arm: Arm, scenario: Scenario, queries_per_condition: int = 3
) -> dict[str, Any]:
    """Run one model over the fixed probe set and summarise it.

    An unavailable model is a recorded outcome, not an exception: a bake-off that dies
    because one cloud endpoint is down loses the arms that did work.
    """
    rater = default_tone_rater()
    memory_texts = [memory.text for memory in scenario.memories]
    records: list[TurnRecord] = []

    try:
        model = arm.build()
    except Exception as error:
        return {"model": arm.name, "kind": arm.kind, "available": False, "error": str(error)}

    for condition, query in _probe_turns(scenario, queries_per_condition):
        store = MemoryStore(embedder=DeterministicEmbedder())
        for memory in scenario.memories:
            store.add(memory)
        conversation = Conversation(
            state=dawn_state(scenario, mood_condition=condition),
            store=store,
            scorer=embr_scorer(embedder=DeterministicEmbedder(), now=_eval_clock),
            model=model,
            top_k=5,
        )
        started = time.perf_counter()
        try:
            reply = conversation.take_turn(query).reply
        except Exception as error:  # one bad turn costs this arm, never the whole bake-off
            return {
                "model": arm.name,
                "kind": arm.kind,
                "available": False,
                "error": f"{type(error).__name__}: {error}",
                "completed_turns": len(records),
            }
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        valence, arousal = rater.rate(reply)
        records.append(
            TurnRecord(
                condition=condition,
                query=query,
                reply=reply[:TRANSCRIPT_CHARS],
                latency_ms=elapsed_ms,
                valence=valence,
                arousal=arousal,
                grounded=is_grounded(reply, memory_texts),
                persona_break=has_persona_break(reply),
            )
        )

    latencies = [record.latency_ms for record in records]
    by_condition = {
        condition: statistics.fmean(
            [record.valence for record in records if record.condition == condition]
        )
        for condition in {record.condition for record in records}
    }
    # The spread across moods, not the mean: a model can be warm everywhere and still be
    # completely unresponsive to the state the architecture is feeding it.
    mood_spread = (max(by_condition.values()) - min(by_condition.values())) if by_condition else 0.0

    return {
        "model": arm.name,
        "kind": arm.kind,
        "available": True,
        "turns": len(records),
        # Recorded per arm because the cloud arms do not share the local token budget, and
        # an asymmetry that is not in the artifact is an asymmetry nobody can check.
        "max_new_tokens": getattr(getattr(model, "settings", None), "max_new_tokens", None),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "mean": statistics.fmean(latencies) if latencies else 0.0,
        },
        "grounded_rate": sum(record.grounded for record in records) / len(records),
        "persona_break_rate": sum(record.persona_break for record in records) / len(records),
        "mood_valence_spread": mood_spread,
        "mean_valence_by_condition": by_condition,
        "transcript": [asdict(record) for record in records],
    }


#: The three hosted models the bake-off uses as a quality ceiling: three different families
#: and a wide size spread, so "bigger" and "different lineage" are separable.
#:
#: Chosen for answering directly. Heavier reasoning models (gpt-oss:20b, qwen3.5:397b) spend
#: the entire token budget on a hidden thinking channel against EMBR's prompt and return an
#: empty reply, at 120 tokens and still at 700. They are excluded because an arm that never
#: speaks is not a measurement, not because they are worse models.
CLOUD_MODELS = ("gemma4:31b", "gpt-oss:120b", "mistral-large-3:675b")

OLLAMA_CLOUD_HOST = "https://ollama.com"

#: Cloud arms need a bigger budget than the local arms because they think before speaking.
#: See the note in `default_arms`: this is a recorded asymmetry, not an oversight.
CLOUD_MAX_TOKENS = 700


def default_arms(include_cloud: bool = True) -> list[Arm]:
    """The standard bake-off line-up, skipping cloud arms when no key is configured.

    Ouro is the arm the thesis is about. The local conventional model is the honest
    control: roughly twice the parameters, same machine, no network. The cloud models are
    a ceiling, and their latencies include network time so they are not comparable to the
    local arms as speed measurements.
    """
    from embr.model import (
        DEFAULT_GENERATION_SETTINGS,
        OllamaRunner,
        OuroRunner,
        StubRunner,
        read_ollama_api_key,
    )

    arms = [
        Arm("stub", StubRunner, kind="stub"),
        Arm("Ouro-1.4B", OuroRunner, kind="looped"),
        Arm(
            "llama3.2:3b (local)",
            lambda: OllamaRunner(model="llama3.2:3b"),
            kind="conventional",
        ),
    ]
    api_key = read_ollama_api_key()
    if include_cloud and api_key:
        # Every hosted model here is a reasoning model: it spends the token budget on a
        # hidden thinking channel and only then speaks. At the shared 120 token budget all
        # three return an empty reply, so they need a larger one to say anything at all.
        # This deliberately breaks "hold sampling equal", which is why cloud arms are a
        # quality ceiling and not a latency comparison. Their wall clock also includes
        # network time, so it was never comparable to the local arms regardless.
        cloud_settings = replace(DEFAULT_GENERATION_SETTINGS, max_new_tokens=CLOUD_MAX_TOKENS)
        arms += [
            Arm(
                f"{name} (cloud)",
                lambda name=name: OllamaRunner(
                    model=name,
                    host=OLLAMA_CLOUD_HOST,
                    api_key=api_key,
                    settings=cloud_settings,
                ),
                kind="cloud",
            )
            for name in CLOUD_MODELS
        ]
    return arms


def run_bakeoff(
    arms: list[Arm],
    out_root: str | Path = "data/bakeoff",
    queries_per_condition: int = 3,
) -> tuple[Path, dict[str, Any]]:
    """Run every arm over the same probe set and write one comparable result directory."""
    scenario = load_eval_scenario()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [run_arm(arm, scenario, queries_per_condition) for arm in arms]
    payload = {
        "arms": results,
        "metadata": {
            "probe_turns_per_arm": len(_probe_turns(scenario, queries_per_condition)),
            "queries_per_condition": queries_per_condition,
            "conditions": list(scenario.mood_conditions),
            "label_set": scenario.name,
            "label_version": scenario.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Every arm saw identical prompts, memories, retrieval and sampling. "
                "Latency is wall clock on one machine and includes network time for "
                "cloud arms, so local and cloud latencies are not like for like."
            ),
        },
    }
    (out_dir / "bakeoff.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )
    return out_dir, payload


def main() -> None:
    """Run the default line-up and print the comparison table."""
    out_dir, payload = run_bakeoff(default_arms())
    print(f"Bake-off written to {out_dir}\n")
    header = f"{'model':<24}{'kind':<14}{'p50 ms':>10}{'p95 ms':>10}{'grounded':>10}{'mood':>8}"
    print(header)
    for arm in payload["arms"]:
        if not arm["available"]:
            print(f"{arm['model']:<24}{arm['kind']:<14}{'unavailable':>38}")
            continue
        print(
            f"{arm['model']:<24}{arm['kind']:<14}"
            f"{arm['latency_ms']['p50']:>10.0f}{arm['latency_ms']['p95']:>10.0f}"
            f"{arm['grounded_rate']:>10.0%}{arm['mood_valence_spread']:>8.3f}"
        )


if __name__ == "__main__":
    main()
