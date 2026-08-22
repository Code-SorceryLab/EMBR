# The state-conditioned corpus

The one thing this project needs and does not have. Everything else in RQ3 is bounded by it,
and no change to the harness lifts that bound: the harness side is finished and tested, and
what is missing is labelled data of a shape nobody in this repository is allowed to write.

## 1. Why a bigger label set is not the fix

The obvious reading of RQ3's null result is "ten queries is too few, write a hundred". It is
the wrong fix. Adding queries of the same *kind* raises the power and changes nothing about
what the metric can express, because the limit is structural:

> nDCG against a state-independent gold set cannot reward mood-congruent recall, in
> principle. A signal that moves retrieval as the character's state moves can only ever move
> it away from the one fixed relevant set, and every such move costs score.

That is not a hypothesis. Scored per state against state-*independent* labels
(`run_rq3_state_conditioned` on the v1 Dawn set), a state-coupled scorer pays for the
coupling and a state-blind one pays nothing at all:

| variant | scored at neutral only | scored per state | what the coupling costs |
|---|---|---|---|
| Park (no state channel) | 0.608 | 0.608 | **0.000** |
| EMBR | 0.594 | 0.586 | -0.007 |
| Emotional RAG | 0.552 | 0.515 | -0.036 |

The system that ignores the character's mood is the only one that loses nothing. Under labels
of this shape, the correct strategy is to have no emotional memory.

## 2. What the corpus has to contain

One relevant set **per state**, authored by someone who was not evaluating a retriever:

```jsonc
{
  "id": "king-news",
  "after_session": 3,
  "query": "any news of the king these days?",
  "relevant": [1, 14],                        // the state-independent fallback, still required
  "relevant_by_state": {                      // the part that lifts the ceiling
    "warm":       [1, 5, 3],
    "neutral":    [1, 14, 13],
    "suspicious": [13, 10, 11]
  },
  "note": "Why each set differs, in the writer's terms rather than the retriever's."
}
```

`eval/scenarios.py` reads this today, `Query.relevant_for(state)` resolves it,
`Scenario.is_state_conditioned` reports whether a label set has it, and
`eval.metrics.state_conditioned_ndcg` scores against it. A label file without
`relevant_by_state` keeps working exactly as before. The harness is not the blocker.

## 3. Where the labels may come from, and where they may not

**They may not come from us.** Not from the maintainers, not from a model, and above all not
from anyone who has seen EMBR's results. A gold set written by the party that wants a
particular ordering is not evidence, and this project has already had to retract one headline
for a subtler version of the same problem (`handoff.md` 6.1b). The same rule kills three
tempting shortcuts:

- **Deriving labels from authored metadata by a rule.** "Betrayals are relevant when trust is
  low" is circular: it is EMBR's event-type gate written as a gold standard.
- **Asking a model.** The poignancy arm already showed what happens when the attacker, or the
  experimenter, can talk to the rater.
- **Reusing the walkthrough's `watch_for` claims.** They are authored, and they predate these
  results, but there are five of them and they exist to make the demo's point.

**They should come from a shipped game whose writers gated lines on relationship state before
anyone thought about retrieval.** That is the whole argument for Stardew Valley: heart-level
dialogue is a writer saying "at this relationship depth, *this* is the line", which is exactly
`relevant_by_state` in another notation, produced for another purpose, by someone with no
stake in the outcome.

| Game data | Maps to | Why it is ground truth |
|---|---|---|
| Heart-gated lines in `Content/Characters/Dialogue/` | the state axis | the writer chose which line fires at which depth |
| Conversation topics, expiring after four days | episodic memory and recency | an authored decay curve |
| Gift tastes per item per character | affect valence | per-character affective labels across hundreds of items |

Roughly thirty villagers gives hundreds to thousands of state-to-line pairs against the
current ten, and the content is external: EMBR never saw it, and its author never saw EMBR.

## 4. How to build it, when the game is available

Stardew is **not installed on the development machine**, which is why this document exists
instead of an extractor. Writing one against a file format nobody here can open would be
guessing, and a confident wrong parser is worse than no parser.

The order of work, for whoever has the game:

1. **Verify the format first, against real files.** Content ships as `.xnb` and many installs
   also carry an unpacked `Content (unpacked)` folder. Open one dialogue file and one gift
   taste file and write down what is actually in them before writing any code.
2. **Extract, never commit.** The dialogue is ConcernedApe's copyrighted content. The
   extractor reads the user's own installation and writes a label file locally; the label
   file stays out of git, and only the derived metrics are ever published. Ship the extractor,
   not the extraction.
3. **Simulate a playthrough offline.** Feed the extracted events through EMBR so it builds a
   real store, then ask whether it retrieves what the game would have said at that heart
   level. Deterministic, reproducible, large N, no C# and no running game.
4. **Pre-register before scoring.** Fix the queries and the per-state sets, hash the file, and
   only then run a retriever against it. `label_sha256` exists for this.

**Two limits to write down now, not after the fact.** Stardew has no arousal dimension, so the
corpus tests the valence axis only, which happens to be the axis the attribution experiment
found does the indexing. And you cannot betray a villager, so the event-type gate has no
equivalent and Dawn stays as the controlled betrayal arc.

## 5. The prediction, worth pre-registering because it can fail

Under state-conditioned labels, EMBR's mood congruence should stop being a liability and
start being worth something: the cost in the table above should turn into a gain, and it
should scale with how much the writer's choice actually depends on relationship state.

**If it does not, that is the more interesting result**, and it should be reported as loudly.
It would mean the mood-congruent recall RQ1 measures is real but does not correspond to what
a human writer thinks belongs to a relationship state, which is a finding about the
psychology being modelled rather than about the code.
