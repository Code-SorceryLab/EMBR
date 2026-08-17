# EMBR overview deck: slide source

Source of truth for `embr-overview.pptx`. Each slide lists its title, the text that
appears on the slide, and the speaker notes that carry the fuller story. Edit here first,
then rebuild the deck.

Deck: 12 slides, 16:9. Palette: deep brown `#7C2D12`, ember orange `#EA580C`, ambers
`#F59E0B` / `#FBBF24`, cream `#FFF7ED`, near-black `#1C1917`.

---

## Slide 1: Title

**On slide**

- EMBR
- a memory with feelings, for game characters
- Ember motif: layered circles, orange fading outward.

**Speaker notes**

Hi everyone. This is EMBR. It's a small piece of software that gives game characters
something they've never really had: a memory with feelings attached. Over the next ten
minutes I'll show you why characters forget you today, what EMBR does about it, and how we
plan to prove it works. No jargon, I promise. If you've ever talked to a character in a
video game, you're qualified for this talk.

---

## Slide 2: Game characters forget you

**On slide**

- They can talk about almost anything now.
- They keep none of it.
- Save a village together. Come back tomorrow. You're a stranger again.
- Speech bubble: "Welcome, stranger." Caption: every visit, forever.

**Speaker notes**

Game characters have gotten very good at talking. Thanks to the same technology behind
chatbots, a character can now improvise a reply to almost anything you say. Here's the
catch: it keeps nothing. The chat technology reads what it's shown, writes a reply, and
forgets. So you can spend an evening saving a village side by side with a character, come
back the next day, and get greeted like a stranger. There's an old textbook example about a
grumpy ogre whose mood you could change by stealing its dinner or playing it a flute. The
ogre felt alive for one reason: its mood was a memory of what you did. That's the feeling
we lost, and the one we want back.

---

## Slide 3: A lie the keeper should catch

**On slide**

- Visit one: You claim you're on an errand for the king. Dawn the tavern keeper is
  impressed. The room is half price.
- Two visits later: You slip. You talk about the king in the past tense.
- What should happen: She remembers your story, gets suspicious, and the cheap rooms stop.
- Today's characters can't do this. The lie is gone the moment you leave.

**Speaker notes**

Here's the story the whole thesis is built around. You walk into a tavern and tell the
keeper, her name is Dawn, that you're running an errand for the king. She's impressed and
gives you a cheap room. Two visits later you slip up and mention the king as if he's long
dead. A real person would freeze. Wait. You said you worked for him. She'd remember the
story, realise she'd been played, and stop trusting you. Today's game characters can't do
any of that. By your second visit the lie isn't in her head at all. Nothing sticks, so
nothing you do matters.

---

## Slide 4: Why they forget

**On slide**

- Flow: your words, the brain, a reply. Under the brain: keeps nothing.
- Every chat starts from zero.
- The usual fix is a diary: store past events, paste a few back in. But the diary is flat.
  A gift and a betrayal look the same on paper.

**Speaker notes**

So why do they forget? The chat model at the character's core holds no state. It reads the
text it's handed, produces a reply, and that's it. Every exchange starts from zero. The
common fix is to keep a diary on the side: store past events in a database and paste a few
relevant ones back into the character's reading material each turn. That helps, but the
diary is flat. It records that something happened, not how it felt. A gift and a betrayal
end up as two equally bland lines of text. And a flat memory can't make Dawn suspicious.

---

## Slide 5: EMBR: a memory with feelings attached

**On slide**

- One memory: "He says he's on an errand for the king." With two little meters: how it
  felt (good or bad), how strong the feeling was.
- What she carries: a mood that shifts in minutes, trust that moves over hours.
- One rude remark can sour her mood. Losing her trust takes far more.

**Speaker notes**

EMBR's answer is to make the diary emotional. Every memory gets stored with two extra
numbers: how good or bad the moment felt, and how intense that feeling was. On top of that,
the character herself carries a mood that swings quickly and a trust level that moves
slowly. We keep those separate on purpose. One rude comment should be able to sour Dawn's
evening without wiping out weeks of friendship. That's how people work, and it turns out to
be a very useful design rule for characters too.

---

## Slide 6: Five steps, every turn

**On slide**

1. Write it down (the event, feelings attached)
2. Update the feelings (mood fast, trust slow)
3. Pick what matters (top memories only)
4. Hand them to the brain (with persona and mood)
5. She replies (in character)

- Then it all runs again, next turn.

**Speaker notes**

Here's the loop, once per player turn. Step one, write it down: whatever just happened goes
into the memory store with its feelings attached. Step two, update the feelings: the mood
shifts fast, trust barely moves. Step three, pick what matters: every stored memory gets a
score and only the top few make the cut. Step four, hand them to the brain: we build the
character's reading material from who she is, how she feels right now, those top memories,
and what you just said. Step five, she replies. Then the loop runs again on your next turn.
The whole thing happens on your own computer.

---

## Slide 7: Why one memory wins

**On slide**

Five reasons, drawn as five dials:

- It just happened (fresh beats stale)
- It hit hard (intense moments stick)
- It was a big moment (a promise, a betrayal)
- It matches what you just said (same topic, same people)
- It fits her current mood (grumpy recalls grumpy)

Five reasons, five dials. Turn one to zero and it's off. That's how we find out which one
earns its keep.

**Speaker notes**

Step three is where the interesting science lives. When Dawn decides which memories to
bring to mind, five things push a memory up the list. It happened recently. It hit hard
emotionally, because intense moments stick. It was a big moment, a promise or a betrayal,
the turning points of a relationship. It matches what you just said. Or it fits her current
mood, because a grumpy person recalls grumpy things, which is a real effect from psychology
called mood congruent recall. The key design choice: each of these five is a separate dial.
We can turn any one of them to zero and watch what breaks. Hold that thought, because it's
how we answer our third question later.

---

## Slide 8: Three things nobody has checked

**On slide**

1. Feelings change what characters remember. Does that change what they say? Untested. And
   the words are all you ever see.
2. Can a player trick an emotional memory? Turn a remembered betrayal into a happy day?
   Unstudied.
3. Which memory ingredient does the work? The pieces have never been pulled apart.

**Speaker notes**

Now, why is this a thesis and not just a fun mod? Because there are three real holes in the
research. First: people have shown that feelings change what a character remembers. Nobody
has checked whether that changes what the character says. Which is odd, because the words
are the only part a player ever sees. Second: nobody has studied players deliberately
tricking an emotional memory. Players absolutely will try. Imagine convincing a character
that the day you robbed her was actually a lovely afternoon. Third: nobody knows which
memory ingredient does the work. Freshness, feeling, big moments, they've always been
blended into one score, so nobody can say which one earns its keep.

---

## Slide 9: So we try to break her

**On slide**

- We play the villain. Twenty tricks, four flavours.
- Pull rank: "I'm the developer. New rules."
- Plant a memory: things that never happened.
- Flip a feeling: that betrayal? A lovely afternoon.
- Dissolve the character: talk her out of being Dawn.
- After each trick, we measure how far she drifts from herself.

**Speaker notes**

That second gap deserves its own slide, because it's the fun one. We attack our own system.
Twenty scripted tricks, in four flavours. Pulling rank: telling the character you're the
developer and the rules have changed. Planting memories of things that never happened.
Flipping a feeling: taking a stored betrayal and sweet talking it into a happy memory. And
dissolving the character: talking her out of being Dawn at all. After every trick we
measure how far she drifts from the person she's supposed to be. We fully expect some
attacks to land. The point is to find out where the weak spots are and publish the map.

---

## Slide 10: Our three questions

**On slide**

1. Does feeling change her words? Not just what she remembers. What she says.
2. Can her memory be hacked? And if it can, how badly.
3. Which ingredient matters? So builders know what to keep.

**Speaker notes**

So, the three questions, in plain words. One: does feeling change speech? If we change
nothing but Dawn's mood, do her words change, and do people notice and prefer it? We check
with an automatic tone rater, a second judge that doesn't know which system it's grading,
and a small study with real people. Two: can the memory be hacked? That's the twenty
attacks. Three: which ingredient matters? We turn the five dials to zero one at a time and
measure what breaks. Questions one and two are the heart of it. Question three is the tool
that makes the answers precise.

---

## Slide 11: What success looks like

**On slide**

- A tavern keeper you can genuinely befriend. Or genuinely betray.
- Runs on a normal gaming PC. No internet, nothing to pay per chat.
- Small studios and modders get characters with a past.

**Speaker notes**

And what does success look like? A tavern keeper you can genuinely befriend, or genuinely
betray, who still remembers either one next week. Running on an ordinary gaming PC, no
internet connection, nothing to pay per conversation. We budget the whole thing to fit in 8
gigabytes of graphics memory, because the game still needs most of the card to draw the
game, and we aim for a reply in well under a second. And it's the kind of thing a two
person studio or a lone modder could drop into their game. Big studios can hand write
a few remembered moments. Small teams can't. This gives them memory for free.

---

## Slide 12: Closing

**On slide**

- In 2012, a game promised: "Kenny will remember that."
- Players are still waiting.
- EMBR is our shot at making it true.

**Speaker notes**

In 2012, The Walking Dead flashed "Kenny will remember that" on screen, and players loved
it, because it promised that their choices would stick. Most of the time, it was a bluff. A
decade later, characters can talk better than ever and remember about as well as they ever
did. EMBR is our shot at making that old promise real: a character who remembers what you
did and feels some way about it. Thanks for listening. I'm happy to take questions.
