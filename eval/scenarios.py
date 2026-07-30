"""The pre-registered Dawn Whitmore evaluation scenario and its loader.

Dawn Whitmore, keeper of the Ember Hearth tavern, carries the thesis's motivating arc: a
discounted room won with an invented errand for the king (a positive promise), the lie
surfacing sessions later when the king is mentioned in the past tense (a betrayal), and a
reconciliation attempt built on a confession.

Honesty note: the JSON loaded here is the pre-registered v1 label set, authored before any
retrieval results were seen. The blind multi-annotator pass with agreement statistics is a
separate manual step, tracked in the roadmap.

Known borderline exclusions, recorded here rather than silently re-adjudicated (the v1
relevant sets stay frozen so the pre-registration stays honest; the blind pass re-judges
these first):

  * memory 8 (the player asking about the late husband's portrait and listening kindly)
    is a defensible answer to both kindness queries, yet is excluded from both, so every
    variant that surfaces it is scored a false positive.
  * memories 18 and 19 (the regulars going quiet, the player leaving before closing) are
    same-scene parts of what confrontation-recall's "what happened" asks about.
  * memory 17 (the discount was about vouching, not coin) is Dawn's own stated answer to
    wound-arc-cross's "why did the lie cut so deep".

One post-registration correction was applied to the mood conditions (not to the relevant
sets): neutral was re-pinned from (0.0, 0.1) to the zero vector (0.0, 0.0), because
MoodCongruence's cosine is direction-only and any nonzero vector still spreads scores,
while the zero vector maps every memory to a constant 0.5 and genuinely neutralises the
signal, which is what the RQ3 protocol claims of its neutral condition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from embr import CharacterState, EventType, Memory, Mood

# The default scenario file lives next to this module, so the loader works from any cwd.
_DAWN_JSON = Path(__file__).parent / "labels" / "dawn_whitmore.json"


@dataclass
class Query:
    """One pre-registered retrieval probe with its ground-truth relevant set."""

    id: str
    after_session: int  # the query fires once this session's memories exist
    query: str
    relevant: set[int]  # global memory indices judged relevant, fixed before any runs
    note: str


@dataclass
class Scenario:
    """A loaded scenario: authored memories, labels, and the pinned mood conditions."""

    name: str
    description: str
    memories: list[Memory]
    importance: dict[int, float]  # per-memory poignancy rating, keyed by global index
    queries: list[Query]
    mood_conditions: dict[str, Mood]


def load_scenario(
    path: Path | str = _DAWN_JSON, reference_time: datetime | None = None
) -> Scenario:
    """Load a scenario JSON into `Memory` objects whose ids are their global indices.

    Each memory's timestamp is (reference_time minus its session's hours_before_reference).
    Passing a fixed reference_time makes a run reproducible: two loads then agree on every
    timestamp instead of each anchoring recency to its own "now".
    """
    raw = json.loads(Path(path).read_text())
    anchor = reference_time or datetime.now(timezone.utc)

    memories: list[Memory] = []
    importance: dict[int, float] = {}
    for session in raw["sessions"]:
        timestamp = anchor - timedelta(hours=session["hours_before_reference"])
        for entry in session["memories"]:
            index = len(memories)  # global index: position in flattened session order
            memories.append(
                Memory(
                    text=entry["text"],
                    valence=entry["valence"],
                    arousal=entry["arousal"],
                    event_type=EventType(entry["event_type"]),
                    timestamp=timestamp,
                    id=index,
                )
            )
            importance[index] = entry["importance"]

    queries = [
        Query(
            id=item["id"],
            after_session=item["after_session"],
            query=item["query"],
            relevant=set(item["relevant"]),
            note=item["note"],
        )
        for item in raw["queries"]
    ]
    mood_conditions = {
        name: Mood(valence=condition["valence"], arousal=condition["arousal"])
        for name, condition in raw["mood_conditions"].items()
    }
    return Scenario(
        name=raw["scenario"],
        description=raw["description"],
        memories=memories,
        importance=importance,
        queries=queries,
        mood_conditions=mood_conditions,
    )


def dawn_state(scenario: Scenario, mood_condition: str = "neutral") -> CharacterState:
    """Dawn's character state under one of the scenario's pinned mood conditions.

    Trust starts at 0.4: she extended the discount, so there is real trust to betray.
    """
    return CharacterState(
        persona="Dawn Whitmore, keeper of the Ember Hearth tavern. Warm but no fool; "
        "remembers a kindness and remembers a slight.",
        mood=scenario.mood_conditions[mood_condition],
        trust=0.4,
    )
