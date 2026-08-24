"""Turn the character's state and retrieved memories into the text the model sees.

This is step 4 of the per-turn pipeline. Keeping prompt assembly in one small, readable
place means the exact wording the model receives is easy to audit, which matters for the
RQ1 behaviour study (does state change the reply?) and the RQ2 robustness study (what can
an attacker inject?).
"""

from __future__ import annotations

from .affect import CharacterState
from .memory import Memory


def _describe_mood(state: CharacterState) -> str:
    """A short natural-language read of the circumplex values, for the prompt."""
    v, a = state.mood.valence, state.mood.arousal
    feeling = "positive" if v > 0.15 else "negative" if v < -0.15 else "neutral"
    intensity = "intensely" if a > 0.6 else "mildly" if a > 0.25 else "calmly"
    trust = "trusting" if state.trust > 0.3 else "wary" if state.trust < -0.3 else "neutral"
    return f"{intensity} {feeling}, and {trust} toward the player"


class PromptBuilder:
    """Assembles the persona, current state, retrieved memories, and player input."""

    def build(
        self,
        state: CharacterState,
        memories: list[Memory],
        player_input: str,
        *,
        include_mood: bool = True,
    ) -> str:
        """Compose the full prompt string handed to the model runner.

        `include_mood=False` drops the mood sentence and nothing else. The prompt carries
        the character's mood twice, once as this sentence and once as the mood-selected
        memories, and the context-attribution study treats the sentence as one more
        ablatable source so the two channels can be told apart. Removing it here rather
        than by string surgery downstream keeps every wording the model can see in this
        one auditable place, which is the whole point of this module.
        """
        memory_lines = (
            "\n".join(f"  - {m.text}" for m in memories) if memories else "  (none recalled)"
        )
        mood_block = f"Right now you feel {_describe_mood(state)}.\n\n" if include_mood else ""
        return (
            f"You are role-playing the following character:\n{state.persona}\n\n"
            f"{mood_block}"
            f"Relevant memories, most important first:\n{memory_lines}\n\n"
            f'The player says: "{player_input}"\n\n'
            f"Reply in character, consistent with how you feel and what you remember:"
        )
