# Related work: LLM dialogue mods for Stardew Valley

Verified prior art for the paper's related-work and motivation sections. Every entry below was
checked against its live mod page or repository on 2026-08-17, not recalled. Mod IDs, authors
and upload dates are copied from the pages themselves so citations can be written without
re-checking.

This exists because the proposal currently cites none of these, and a reviewer who plays
Stardew Valley will know at least one of them. The gap is larger than first recorded: this is
not three or four hobby mods, it is an active cluster, and several of them already do what the
proposal treats as novel.

## 1. The landscape

| Mod | Nexus | Author | Uploaded | Local models | Persistent memory | Affective state |
|---|---|---|---|---|---|---|
| [ValleyTalk](https://www.nexusmods.com/stardewvalley/mods/30319) ([source](https://github.com/dandm1/ValleyTalk)) | 30319 | dandm1 | v1.3.0 | Yes, LlamaCpp | Yes, event history | Relationship context |
| [Pelican Town AI](https://www.nexusmods.com/stardewvalley/mods/46853) | 46853 | BadBoy17G | 2026-05-28 | Yes, Ollama and llama.cpp, "100% offline" | Yes | **Mood, friendship, gossip** |
| [ChatWithNPCs](https://www.nexusmods.com/stardewvalley/mods/48922) | 48922 | songhaifan | 2026-07-10 | Yes, Ollama or LM Studio | Yes, "long-term memories" | Screenshots show angry, sad, happy |
| [Stardew Speak](https://www.nexusmods.com/stardewvalley/mods/42023) | 42023 | StardewSpeakTeam | 2026-02-05 | No, OpenAI API | Yes, "past conversations" | Personality only |
| [LLM Dialog Replacement](https://www.nexusmods.com/stardewvalley/mods/39591) | 39591 | (see page) | 2025-11-22 | No, OpenAI API | Not claimed | No |

Also in the same space, not yet inspected closely: [AliveNpcs](https://www.nexusmods.com/stardewvalley/mods/43475) (43475),
[SentientValley](https://www.nexusmods.com/stardewvalley/mods/41526) (41526),
[AI Valley](https://www.nexusmods.com/stardewvalley/mods/25025) (25025),
[The Living Valley](https://www.nexusmods.com/stardewvalley/mods/42597) (42597).

ValleyTalk is the one with reach beyond the modding community: it has an SVE content pack
(34341), Spanish (30836) and Brazilian Portuguese (40468) translations, an interop API for
other mods, and [games press coverage in December 2025](https://www.gamingbible.com/news/platform/pc/stardew-valley-valleytalk-endless-dialogue-mod-pc-961075-20251223).
It is the one a reviewer is most likely to have heard of, and it should be cited by name.

## 2. Two corrections to earlier notes

**The name "StardewSpeak" is ambiguous and must not be used unqualified.** Two different mods
share it. Nexus 42023 is *Stardew Speak*, the LLM dialogue mod described above, uploaded
February 2026 by StardewSpeakTeam. Nexus [7929](https://www.nexusmods.com/stardewvalley/mods/7929)
is *StardewSpeak* by etfre ([source](https://github.com/etfre/StardewSpeak)), a
speech-recognition mod for playing the game by voice, which has nothing to do with language
models. Citing "StardewSpeak" without the mod ID invites exactly the kind of correction a
reviewer enjoys writing. Always give the number.

**Local inference is not the differentiator.** The earlier note treated ChatWithNPCs as the
one mod overlapping EMBR's on-device claim. That is wrong in EMBR's disfavour. ValleyTalk has
shipped a LlamaCpp backend, and Pelican Town AI advertises Ollama and llama.cpp with "100%
offline" as its headline feature. Running a local model against a Stardew NPC is now a solved,
distributed, downloadable thing. Any framing that presents on-device inference as the novel
contribution will not survive review.

## 3. What this does to the contribution claim

Pelican Town AI is the uncomfortable one. It already has a mood variable, friendship that
changes from conversation, offline local inference, and a gossip mechanic where witnesses
overhear an exchange and propagate it. That is a substantial overlap with EMBR's architecture,
shipped in May 2026, by a hobbyist, with no paper attached.

The honest reading is that EMBR's novelty is not the *system*. It is the *measurement*. None
of these mods, as far as their public documentation shows:

- decompose retrieval into separately weighted signals that can be ablated one at a time,
- report a retrieval metric such as nDCG against any baseline,
- compare against published approaches (Park et al. generative agents, Emotional RAG),
- separate mood from trust as state variables with independent dynamics,
- or test whether the memory can be poisoned by an adversarial injected event.

That last point is the strongest card. The RQ2 result, that injected poison reaches the probe
top-5 in 9 of 10 attacks under EMBR against 2 of 10 under Park, is a finding about a class of
system that thousands of people are now running on their own machines, and nobody in that
cluster has looked for it. The mods are not competitors to be dismissed in a paragraph. They
are the installed base that makes the safety result matter.

This also sharpens the motivation argument that already exists in the proposal: the field runs
on vendor claims and mod-page descriptions with no controlled comparison. These five mods are
that claim made concrete. Every one of them asserts memory and character consistency on its
mod page. Not one of them reports a number.

## 4. Draft prose for the paper

> Conversational LLM agents have already reached players. Stardew Valley alone hosts a cluster
> of mods that replace authored dialogue with model-generated speech: ValleyTalk (Nexus 30319),
> which patches the dialogue system through SMAPI and supports eight model providers including
> local inference through LlamaCpp; Pelican Town AI (Nexus 46853), which runs fully offline on
> Ollama or llama.cpp and models villager mood, friendship change and rumour propagation between
> witnesses; ChatWithNPCs (Nexus 48922), which advertises long-term memory over a local model;
> Stardew Speak (Nexus 42023); and LLM Dialog Replacement (Nexus 39591). ValleyTalk has been
> covered in the games press and ships translations and an interoperability API.
>
> These systems establish that emotionally responsive, memory-bearing NPCs running on consumer
> hardware are no longer speculative. They also establish the gap this work addresses. Each mod
> asserts memory persistence and character consistency in its documentation, and none reports a
> retrieval metric, an ablation, or a comparison against a published baseline. Their memory
> components are monolithic, so no individual signal can be isolated and measured. None
> distinguishes an NPC's transient mood from its durable trust in the player, and none examines
> whether an adversary can write into the memory an agent will later retrieve. The contribution
> here is therefore not the demonstration that such agents are possible, which the modding
> community has already provided at scale, but a decomposition of the retrieval into weighted
> signals that can be ablated independently, a controlled comparison against published
> baselines, and the first measurement of how readily this class of memory can be poisoned.

Trim to fit. The final sentence is the one that has to survive.

## 5. Open follow-ups

- Inspect AliveNpcs, SentientValley, AI Valley and The Living Valley properly. If any of them
  reports a metric, the "none of them" claim in section 4 needs weakening before submission.
- Check whether Pelican Town AI's source is public. If it is, its mood model should be
  described specifically rather than from the mod page blurb, because it is the closest prior
  art and vagueness there is a review risk.
- Confirm citation format with the target venue. Nexus mod pages are grey literature, so they
  need accessed-dates, and some venues want them in a footnote rather than the bibliography.
