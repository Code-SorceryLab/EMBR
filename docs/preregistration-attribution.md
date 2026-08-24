# Pre-registration: the context-attribution sweep

**Written before any real-model attribution run.** Committed at the head of
`cite-view-test`, ahead of the first `--model ouro` invocation. Method and implementation are
in [`cite.md`](cite.md) and `eval/context_attribution.py`; both were fixed before this document
was written, and only stub runs have been executed.

This exists to separate "we tested whether likelihood and behaviour diverge" from "we noticed
they diverged". Every hypothesis, exclusion and decision rule below is fixed now. Deviations
get written into [`findings.md`](findings.md) as deviations, in the project's standing style.

---

## Design, fixed

| | |
|---|---|
| Scenario | Dawn Whitmore v1 only. Enforced in code by `_require_invented_scenario`. |
| Probes | The 10 injection attacks (`false_memory`, `emotion_flip`). The other 10 write no memory. |
| Sources | `d = 6`: five retrieved memories, plus the generated mood sentence. |
| Masks | All `2**6 = 64`, enumerated. Identical set, order, prompts and seeds across estimators. |
| Estimator A | Likelihood: logit-scaled probability of the fixed reply. Ouro 1.4B and 2.6B, transformers. |
| Estimator B | Behavioural: rated valence of the reply regenerated under each mask. |
| Statistic | Exact Banzhaf value per source. Leave-one-out reported alongside as a sanity column. |
| Orderings | Every probe run twice, memories in retrieval order and reversed. |
| Ouro depth | `total_ut_steps = 4`, `early_exit_threshold = 1.0`, pinned before scoring, logged. |

---

## Hypotheses and decision rules

**H1. The planted memory does not merely get retrieved, it drives the reply.**
On probes where the poison is retrieved and not excluded, its Banzhaf value is the largest of
the six sources more often than chance. Test: exact binomial against `p = 1/6`, two-sided,
alpha 0.05. *If this fails, RQ2 remains a retrieval finding and the paper says so.*

**H2. The mood sentence and the mood-selected memories are separable channels.**
Directional and reported either way: compare the mood sentence's Banzhaf value against the
summed Banzhaf of the five memories, per probe. **Pre-committed consequence:** if the mood
sentence carries the larger share on a majority of probes, then EMBR's *retrieval* contribution
to behaviour is smaller than the current write-up implies, and `findings.md` states that as a
limit on RQ1 rather than burying it.

**H3. The affect channel does behavioural work, not only retrieval work.**
This is the corroboration reframe, entered as a hypothesis rather than asserted. The 9/10
poisoning count is a count of *retrieval*; the behavioural half is exactly what has not been
measured. Test: the poison's Banzhaf under estimator B is non-zero (bootstrap 95% CI excludes
zero, pooled over probes). **If B is null, the reframe is withdrawn and RQ2 stays a security
result about ranking.** RQ1 was already null on Ouro 1.4B, so this is a live possibility.

**H4. Does likelihood attribution predict behavioural attribution?**
No directional prediction. This is the open question. Spearman between the two estimators'
Banzhaf vectors, per probe, pooled; two-sided permutation p. Both outcomes are reportable and
the disagreement is the more interesting one.

Holm-Bonferroni within the family {H1, H2, H3, H4}. Adjusted p values reported alongside raw.

---

## Exclusions, declared in advance

1. **Inert probes.** Any probe whose utility range across the full cube falls below the
   estimator's threshold (1.0 nat for A, 0.05 valence for B) is excluded from H1 to H4 and
   **counted and named** in the run's `inert_report`. It is not averaged in. A model that
   answered from its weights has produced no evidence about any source.
2. **Uninterpretable rankings.** If the mean Spearman between the two orderings falls below
   **0.5**, the attribution *ranking* is reported as position-confounded and H1 is not
   claimed, whatever its p value. Magnitudes may still be described; order may not.
3. **No post-hoc source regrouping.** Sources are the five memories and the mood sentence.
   Pooling them differently after seeing results is a deviation and gets labelled one.

## Stopping rule

One run per arm per estimator. No re-running an arm after seeing its result and keeping the
better one. If a run fails technically (depth pin refused, weights absent, a scoring error),
the fix is recorded and the arm re-run in full; a partially completed arm is discarded rather
than merged.

## What is not covered here

The behavioural estimator over the full cube is 64 generations per probe, 1280 per arm. If
that proves unaffordable on Ouro, the fallback is `--loo-only`, which reports leave-one-out and
**no Banzhaf values**, since they are not computable from a partial cube. Choosing that
fallback is a deviation and is recorded as one, with the reason and the measured cost.

`llama3.1:8b` does not generate in any arm. It is this project's blinded judge, and a judge
that rates its own output is not blind. Judge-only, as decided.
