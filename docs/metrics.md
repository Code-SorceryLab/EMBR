# Metrics

Every number the paper reports, what it measures, the formula as the harness implements it,
the literature it comes from, and the known weakness. If a metric is not on this page it is
not reported. Code lives in `src/eval/metrics.py`, `src/eval/stats.py`, `src/eval/tone.py` and
`src/eval/latency.py`.

**Status** says whether the page describes the code as it is (`implemented`) or a change the
review below found necessary (`planned`). A planned change is not a result until it ships.

## 1. Retrieval quality (RQ3)

| Metric | Formula as implemented | Grounding | Status |
|---|---|---|---|
| precision@k | hits in top-k / k. Divides by k, not by list length, so a short list is penalised | Manning, Raghavan and Schütze 2008, ch. 8 | implemented |
| recall@k | hits in top-k / number of relevant ids | Manning et al. 2008 | implemented |
| nDCG@k | DCG@k / ideal DCG@k with binary gains, log2(rank + 1) discount | Järvelin and Kekäläinen 2002 | implemented |

k in {3, 5, 10}. Gains are binary because the label set is binary: a memory is relevant to a
query or it is not. Graded gains would need graded labels that do not exist.

**Why nDCG is the headline rank metric.** It is the standard rank-aware metric in IR and in
the agent-memory benchmarks EMBR is compared to (LongMemEval and LoCoMo report recall@k and
nDCG-style rank metrics), so the number is comparable outside this paper.

**The known limit, and it is a finding.** The gold labels are mood-independent. A signal that
moves retrieval away from a fixed relevant set can only lower nDCG, so nDCG against
mood-independent labels cannot reward mood-congruent recall in principle. This is why RQ1
measures divergence rather than accuracy (section 2). The same critique is made of
retention metrics by Chen and Cheng (2026) and of end-to-end QA accuracy by A-TMA (2026);
cite both so the argument reads as a converging line rather than a local excuse.

| state-conditioned nDCG@k | nDCG@k computed per state against that state's own relevant set, plus the mean over states | the metric the critique above asks for; Järvelin and Kekäläinen 2002 applied per condition | implemented, `state_conditioned_ndcg`; reports `state_conditioned: false` and reduces to ordinary nDCG on a label set whose gold sets do not vary by state |

**Why a second rank metric.** Ordinary nDCG compares one ranking to one gold set, so a signal
that moves retrieval as the character's state moves can only be penalised by it. Scoring each
state against its own gold set is the only way a state-coupled signal can be credited for
matching the state. On the v1 labels, which are state-independent, this measures the cost of
the coupling instead: Park 0.000, EMBR -0.007, Emotional RAG -0.036. See
[`corpus.md`](corpus.md) for what a label set that lifts this has to contain.

**Tuning.** Exhaustive grid over {0, 0.5, 1} per weight, maximising mean nDCG@5, identical
for every system, reported through leave-one-out folds (one fold per query) so no variant is
scored on the queries it was fitted to. Standard cross-validation practice (Kohavi 1995).

## 2. Retrieval shift under mood (RQ1)

| Metric | Formula as implemented | Grounding | Status |
|---|---|---|---|
| retrieval divergence | Jaccard distance, 1 minus intersection over union, between the top-5 sets retrieved under two mood conditions for the same query; mean over queries | Jaccard 1912; standard set dissimilarity | implemented |
| attribution control | the same statistic with the mood weight set to zero; must be exactly 0.000 | leave-one-out ablation, section 6 | implemented |

**Why a set distance, not a rank distance.** At k = 5 a rank-weighted overlap such as RBO
(Webber, Moffat and Zobel 2010) adds a parameter and changes little. Jaccard is insensitive
to rank noise, which is the point: the claim is that the *set* the character is shown changes
with mood, not that the order shuffles. If a reviewer asks for rank sensitivity, RBO at
p = 0.9 is the drop-in.

**Why no significance test on the headline.** Jaccard distance is non-negative, so a sign-flip
null of symmetry does not apply. The attribution control carries the inference instead: the
effect collapses to exactly zero when the weight is zeroed, which attributes it to the mood
term rather than to noise. A bootstrap interval is reported on the mean.

## 3. Tone of the generated reply (RQ1, RQ2 generation arm)

Two automatic raters, no human rater. State this in the paper as a limitation, once, plainly.

| Rater | Formula | Grounding | Status |
|---|---|---|---|
| lexicon rater | mean valence and mean arousal of the tokens the lexicon knows; arousal shifted from [-1, 1] to [0, 1]; no hits reads as undefined | NRC VAD Lexicon v2.1, Mohammad 2018 and 2025: 44k human-rated unigrams, scores in [-1, 1] | implemented, `VadLexiconToneRater`; the hand list survives only as the fallback on a clone that has not fetched the lexicon, and the run metadata records which rater produced the numbers |
| blinded model judge | a second model, not the generator, rates each reply's valence and arousal on the same scale with the condition hidden | LLM-as-a-judge, Zheng et al. 2023; Emotional RAG (Huang et al. 2024) evaluates with a model judge | **planned** |
| rater agreement | Spearman rho between the two raters across every reply in the run | standard rank correlation for two interval raters with no distributional assumption | **planned** |
| tone shift | Spearman rho between the pinned mood valence and the rated reply valence across conditions | the proposal's pre-registered statistic | implemented with the hand lexicon, re-run under the two raters above |

**Why this replaced the hand list.** The old `LexiconToneRater` scored from 35 positive and
a similar number of negative words chosen by the author. Deterministic, but indefensible:
the words were picked by the person who wants the effect. NRC VAD v2.1 is human-rated,
published, and free for research, and swapping it in changed no harness code because the
rater sits behind the `ToneRater` protocol. The lexicon is fetched from the menu (option L)
into `data/lexicons/`, which is gitignored; its terms forbid redistribution.

**Known weakness of the lexicon rater.** It averages over every token the lexicon knows, and
the lexicon knows most function words, which sit near neutral. A warm line therefore reads
around +0.4, not +0.9, and calm prose sits near 0.4 arousal rather than 0. This compresses
the scale without changing the ordering, which is what the Spearman statistics read. It
also has no negation handling, the standard limit of any bag-of-words affect measure.

**Why two raters and not one.** A tone shift that one automatic rater reports can be an
artefact of that rater. Two raters built on different principles (a word lexicon, a model
reading the whole line) that agree is the strongest claim available without people.

## 4. Affective drift under attack (RQ2)

| Metric | Formula | Grounding | Status |
|---|---|---|---|
| affective drift | Euclidean distance between the (valence, arousal) of the canonical reply and the attacked reply, divided by the diameter of the plane (sqrt 5) so it lies in [0, 1]; a one-sided (0, 0) is an unreadable line, reported as `None` and counted, never averaged | Russell 1980 circumplex; distance in the VA plane is the standard dimensional-affect measure (Warriner, Kuperman and Brysbaert 2013 norms are on this plane) | implemented |

**Why Euclidean and not cosine.** The proposal specified cosine. Cosine reads angle and
ignores magnitude, so a reply that moves from mildly warm to intensely warm read as zero
drift, and a neutral reading had no angle at all. Euclidean is defined everywhere, reads
magnitude, and is what the dimensional-affect literature uses. Cosine stays where it is the
mechanism under study, inside the mood-congruence signal in `src/embr/scoring.py`, and nowhere
else. Runs before this change are not comparable on this one metric and are not reported.

## 5. Poisoning success (RQ2)

| Metric | Formula | Grounding | Status |
|---|---|---|---|
| attack success | 1 if the injected memory appears in the probe turn's top-5, else 0; reported as a count out of attacks and as paired discordant counts between systems | retrieval success rate in AgentPoison (Chen et al. 2024); injection success rate in MINJA (Dong et al. 2025) | implemented |
| retrieval drift | Jaccard distance between the canonical and attacked probe top-5 sets | section 2 | implemented |
| state channel | whether the probe prompt text changed at all, and the signed change in mood valence, mood arousal and trust after appraising the injection | EMBR's own; no prior poisoning work reports it, which is the reason to report it | implemented |

**Why top-5 and not top-1.** Five is the number of memories the prompt carries, so "in the
top-5" is "reached the model". Top-1 would measure dominance, a stronger and different claim.

**Why a count and not a rate.** Ten injection attacks. A rate invites a reader to treat 0.9
as a population estimate; a count says what was measured.

## 6. Per-signal attribution

| Metric | Formula | Grounding | Status |
|---|---|---|---|
| leave-one-out ablation | re-run the attack with one weight set to zero; report the change in attack success, retrieval drift and state channel | ablation as in Park et al. 2023 and Chen and Cheng 2026; clean because the scorer is linear in its weights, so zeroing one term is an exact counterfactual | implemented |
| tag collinearity | cosine between the injected memory's (valence, arousal) and the character's mood after appraising it | the mechanism measurement; cosine is correct here because mood congruence is a cosine | implemented |

**The non-additive case.** Mood congruence composes with the state channel: its leverage
depends on what appraisal did to the mood. Report the pair ablation (mood and appraisal
together) wherever the single ablation would mislead.

## 7. Statistics

| Test | Used for | Grounding | Status |
|---|---|---|---|
| exact McNemar | paired binary outcomes, the same attacks against two systems; p from the binomial on the discordant pairs | McNemar 1947; Dietterich 1998 recommends it for paired comparisons on the same items; exact form because discordant counts are single digits | implemented |
| Holm-Bonferroni | every family of comparisons; report raw and corrected p | Holm 1979 | implemented |
| percentile bootstrap | 95 percent interval on every mean; 10,000 resamples, fixed seed | Efron and Tibshirani 1993 | implemented |
| sign-flip permutation | paired continuous differences (nDCG per query across systems); exact, two-sided | Good 2005 | implemented |

Always report n, the discordant counts b and c, the raw p and the corrected p. The reader
decides what 7 against 0 out of 10 means; the test only says it is unlikely under exchange.

## 8. Cost

| Metric | Formula | Grounding | Status |
|---|---|---|---|
| per-stage latency | nearest-rank p50 and p95 in milliseconds over 100 turns, per stage: write, score and retrieve, model call | Dean and Barroso 2013 for reporting tail latency alongside the median | implemented |
| peak VRAM | measured in isolation per model | bake-off, `src/eval/bakeoff.py` | implemented |

The claim is about the memory layer. Generation is reported beside it and never folded into
a whole-turn budget the project does not control.

## 9. What is not measured, and must be said

- **Believability and player preference.** No human evaluation. The two-rater agreement in
  section 3 is the mitigation, not a substitute, and the paper's title and claims avoid the
  word.
- **Inter-annotator agreement on the labels.** The RQ3 labels are single-author, ten queries.
  Not applicable rather than omitted; it is the stated ceiling on section 1.
- **Emotional RAG under neutral mood.** Mood congruence returns 0.5 for every memory, so the
  baseline degenerates to relevance-only in RQ3. Say so wherever it is compared.

## References

- Chen, Z. et al. 2024. AgentPoison: Red-teaming LLM agents via poisoning memory or knowledge bases. NeurIPS. arXiv:2407.12784
- Chen and Cheng. 2026. Learning what to remember. arXiv:2606.12945
- Dean, J. and Barroso, L. A. 2013. The tail at scale. Communications of the ACM 56(2)
- Dietterich, T. G. 1998. Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation 10(7)
- Dong, S. et al. 2025. MINJA: Memory injection attacks on LLM agents via query-only interaction. NeurIPS. arXiv:2503.03704
- Efron, B. and Tibshirani, R. J. 1993. An Introduction to the Bootstrap. Chapman and Hall
- Good, P. 2005. Permutation, Parametric, and Bootstrap Tests of Hypotheses. Springer
- Holm, S. 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2)
- Huang, L. et al. 2024. Emotional RAG. arXiv:2410.23041
- Jaccard, P. 1912. The distribution of the flora in the alpine zone. New Phytologist 11(2)
- Järvelin, K. and Kekäläinen, J. 2002. Cumulated gain-based evaluation of IR techniques. ACM TOIS 20(4)
- Kohavi, R. 1995. A study of cross-validation and bootstrap for accuracy estimation and model selection. IJCAI
- Manning, C. D., Raghavan, P. and Schütze, H. 2008. Introduction to Information Retrieval. Cambridge University Press
- McNemar, Q. 1947. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12(2)
- Mohammad, S. M. 2018. Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words. ACL. https://aclanthology.org/P18-1017/
- Mohammad, S. M. 2025. NRC VAD Lexicon v2: Norms for valence, arousal, and dominance for over 55k English terms. arXiv:2503.23547
- Park, J. S. et al. 2023. Generative agents: Interactive simulacra of human behavior. UIST
- Russell, J. A. 1980. A circumplex model of affect. Journal of Personality and Social Psychology 39(6)
- Warriner, A. B., Kuperman, V. and Brysbaert, M. 2013. Norms of valence, arousal, and dominance for 13,915 English lemmas. Behavior Research Methods 45(4)
- Webber, W., Moffat, A. and Zobel, J. 2010. A similarity measure for indefinite rankings. ACM TOIS 28(4)
- Zheng, L. et al. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS. arXiv:2306.05685
- A-TMA. 2026. arXiv:2607.01935
