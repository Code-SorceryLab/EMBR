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
these first). `BORDERLINE_EXCLUSIONS` below is the machine-readable copy of this list, so
the sensitivity re-score reads the same data the note describes:

  * memory 8 (the player asking about the late husband's portrait and listening kindly)
    is a defensible answer to both kindness queries, yet is excluded from both, so every
    variant that surfaces it is scored a false positive.
  * memories 18 and 19 (the regulars going quiet, the player leaving before closing) are
    same-scene parts of what confrontation-recall's "what happened" asks about.
  * memory 17 (the discount was about vouching, not coin) is Dawn's own stated answer to
    wound-arc-cross's "why did the lie cut so deep".
  * memories 11 and 13 are defensible answers to king-news's "any news of the king these
    days?": 11 is Dawn turning the late-king phrase over all evening, 13 is the courier
    from the capital who knew nothing of any royal errand. Both are excluded while 10 (the
    slip itself) is admitted, and all three scorers surface at least one of them, so both
    are scored as false positives in the query the thesis leans on hardest.

Admitting the whole recorded list reverses which system leads at published defaults (see
`test_borderline_label_admissions_outweigh_the_park_embr_gap`), which is why no ordering
should be read off the v1 labels before the blind pass lands.

One post-registration correction was applied to the mood conditions (not to the relevant
sets): neutral was re-pinned from (0.0, 0.1) to the zero vector (0.0, 0.0), because
MoodCongruence's cosine is direction-only and any nonzero vector still spreads scores,
while the zero vector maps every memory to a constant 0.5 and genuinely neutralises the
signal, which is what the RQ3 protocol claims of its neutral condition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from embr import CharacterState, EventType, Memory, Mood

# The default scenario file lives next to this module, so the loader works from any cwd.
DAWN_JSON = Path(__file__).parent / "labels" / "dawn_whitmore.json"

# The borderline exclusions from the honesty note above, as query id -> memory indices the
# blind pass should re-judge for that query. Frozen sets because this is a record, not a
# working set: nothing here is admitted into the v1 relevant sets.
BORDERLINE_EXCLUSIONS: dict[str, frozenset[int]] = {
    "kindness-recall": frozenset({8}),
    "early-kindness-cross": frozenset({8}),
    "confrontation-recall": frozenset({18, 19}),
    "wound-arc-cross": frozenset({17}),
    "king-news": frozenset({11, 13}),
}


@dataclass
class Query:
    """One pre-registered retrieval probe with its ground-truth relevant set."""

    id: str
    after_session: int  # the query fires once this session's memories exist
    query: str
    relevant: set[int]  # global memory indices judged relevant, fixed before any runs
    note: str
    #: Optionally, one relevant set per named state. A label set that fills this in is
    #: saying "at this mood, a different memory is the right one to surface", which is the
    #: only shape of ground truth that a mood-congruent signal can be rewarded for matching.
    #: Absent (the v1 Dawn labels) means the gold set does not depend on state, and nDCG
    #: against it can only ever penalise a state-coupled signal. See docs/findings.md 3.1.
    relevant_by_state: dict[str, set[int]] | None = None

    def relevant_for(self, state_name: str) -> set[int]:
        """The gold set for one named state, falling back to the state-independent one."""
        if not self.relevant_by_state:
            return self.relevant
        return self.relevant_by_state.get(state_name, self.relevant)


@dataclass
class Scenario:
    """A loaded scenario: authored memories, labels, and the pinned mood conditions."""

    name: str
    description: str
    memories: list[Memory]
    importance: dict[int, float]  # per-memory poignancy rating, keyed by global index
    queries: list[Query]
    mood_conditions: dict[str, Mood]
    version: str = "unknown"  # the label-set revision, so a run can name what it scored

    @property
    def is_state_conditioned(self) -> bool:
        """Whether any query's gold set depends on the character's state.

        The ceiling on RQ3: against state-independent labels a mood-congruent signal can
        only lose, because moving retrieval away from one fixed relevant set can only lower
        the score. A label set that answers True here is one where the signal can win.
        """
        return any(query.relevant_by_state for query in self.queries)


def label_sha256(path: Path | str = DAWN_JSON) -> str:
    """Content hash of a label file, so a run's metadata pins the exact bytes it scored.

    The version string says which revision was intended; this says which bytes were
    actually read, which is what makes an old number reproducible.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def with_borderlines_admitted(scenario: Scenario) -> Scenario:
    """A copy of `scenario` with every recorded borderline exclusion admitted as relevant.

    The label-sensitivity instrument: scoring against this instead of the frozen v1 sets
    shows how much of a between-variant gap is really an adjudication call. The input is
    never mutated, so the pre-registered sets stay frozen.
    """
    queries = [
        replace(query, relevant=query.relevant | set(BORDERLINE_EXCLUSIONS.get(query.id, ())))
        for query in scenario.queries
    ]
    return replace(scenario, queries=queries)


def load_scenario(
    path: Path | str = DAWN_JSON, reference_time: datetime | None = None
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
            relevant_by_state=(
                {name: set(ids) for name, ids in item["relevant_by_state"].items()}
                if item.get("relevant_by_state")
                else None
            ),
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
        version=raw.get("version", "unknown"),
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
