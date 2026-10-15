# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

I built and evaluated both and compare them below.

## 1. Model Overview

**Model type:**
I compared both models. The rule based model is the main one. The ML model is the comparison.

**Intended purpose:**
Classify short, social-media-style messages as `positive`, `negative`, `neutral` or `mixed`. This is a learning project, not something to use on real people's messages.

**How it works (brief):**
- **Rule based:** lowercase and tokenize the text, look each token up in a hand-written list of positive and negative words (`dataset.py`), add +1 for each positive hit and -1 for each negative hit, then turn the total into a label. Negation flips a word's sign.
- **ML:** `CountVectorizer` turns each post into word counts, and a `LogisticRegression` model learns which words go with which label from `SAMPLE_POSTS` and `TRUE_LABELS`.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` has **30** posts: 6 from the starter, 8 "realistic" posts (slang, emojis, mixed or subtle tone), 10 "breaker" posts (sarcasm, shifting slang, ambiguous emojis, mixed emotions), and 6 more added during the ML comparison. The word lists have 18 positive and 17 negative entries (some are emojis).

**Labeling process:**
The new posts and their labels were drafted by my AI assistant, not by several independent people. That means every label is one annotator's judgment, and that annotator is an AI. Posts that could reasonably get another label:
- "I'm fine 🙂" is labeled `negative` (passive-aggressive), but a reader could call it `neutral`.
- "my flight got cancelled again 💀 love that for me" is labeled `negative` (sarcasm).
- "I mean, the interview went okay I guess 🥲" is labeled `mixed`, but `neutral` is defensible.
- "lol 💀" is labeled `positive` (laughing), but 💀 can mean the opposite.

**Important characteristics of your dataset:**
- Slang and emojis: "no cap", "bruh", "bussin", 💀, 🥲, 😭.
- Sarcasm: "I love getting stuck in traffic", "Oh great, another Monday meeting".
- Mixed feelings: 7 posts.
- Short and ambiguous messages: "lol 💀", "I'm fine 🙂".

**Possible issues with the dataset:**
- **Small:** 30 posts, so one post is 3% of any accuracy number.
- **Imbalanced:** 10 positive, 10 negative, 7 mixed, only **3 neutral**.
- **Single annotator** (see above).
- **Lexicon tuned on the test data:** several words were added after I saw which posts the model got wrong, so rule based accuracy on this dataset is flattered.
- All English, all short, and written in one voice.

## 3. How the Rule Based Model Works

**Your scoring rules:**
- **Preprocess:** lowercase, normalize curly apostrophes, split into word and emoji tokens, drop punctuation except `, . ; : ! ?` (kept as clause breaks).
- **Score:** +1 for each word in the positive list, -1 for each in the negative list. The score is positives minus negatives.
- **Enhancement: negation.** A negator (`not`, `no`, `never`, `don't`, `isn't`, `won't`, ...) flips the sign of sentiment words up to 3 tokens after it. A clause break (`,` `.` `!` `?`) or the word "but" ends the negation. `can't` is deliberately not a negator, because "can't believe how stressed I am" is not negating "stressed".
- **Labels:** score > 0 is `positive`, score < 0 is `negative`. A score of 0 is `mixed` if there were both positive and negative hits, otherwise `neutral`.
- **Emojis:** kept as tokens, and only the few in the lists (🔥 😂 😤 😞 😢) are scored. 💀 🙂 😭 🥲 are not scored on purpose, because their meaning depends on context.

**Strengths of this approach:**
- Predictable and easy to explain: every label can be traced to specific words.
- Handles plain negation ("I am not happy about this") and a hedged tie ("exhausted but proud of myself" is `mixed`).
- Needs no training data.

**Weaknesses of this approach:**
- A word that isn't in the lists counts as nothing, so unknown wording silently becomes `neutral`.
- Sarcasm, context-dependent slang and emojis (see Limitations).

## 4. How the ML Model Works

**Features used:**
Bag of words with `CountVectorizer`. It ignores emojis and one-letter words by default and has no negation handling.

**Training data:**
Trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py` (all 30 posts).

**Training behavior:**
Adding 6 posts moved leave-one-out accuracy only from 0.21 to 0.23, since there still isn't enough data. Predictions on individual posts changed when I retrained. I did not test what happens when a single label is changed.

**Strengths and weaknesses:**
- *Strength:* it picks up patterns I didn't write down. It treats "but" as a signal for `mixed` and got three "but" posts right even when each was held out.
- *Weakness:* it memorizes. With 30 posts it reaches 1.00 on its own training data, which says nothing about new text. It also learns spurious cues: `happy` and `not` as `negative` (because the one "not happy" post was negative), and `lol`, `job`, `class` as `positive`.

## 5. Evaluation

**How you evaluated the model:**
All numbers are on the 30 labeled posts in `dataset.py`.

| System | Accuracy |
|---|---|
| Rule based | 0.53 (16 of 30) |
| ML, on its own training data | 1.00 (memorized, not meaningful) |
| ML, leave-one-out (each post predicted without training on it) | 0.23 (7 of 30) |

The rule based number is optimistic, because I tuned the word list after seeing failures. The ML 1.00 is not a real test. Leave-one-out is the fairest comparison for the ML model.

**Examples of correct predictions:**
- "I am not happy about this" is `negative`. `not` flips `happy` from +1 to -1.
- "exhausted but proud of myself" is `mixed`. `exhausted` is -1 and `proud` is +1, a tie with hits on both sides.
- "the meeting is at 3pm in room 204" is `neutral`. No sentiment words, so the score is 0.

**Examples of incorrect predictions:**
- "my flight got cancelled again 💀 love that for me" (true `negative`, predicted `mixed`): `love` is +1 and `cancelled` is -1, so the model sees a tie. It can't tell "love that for me" is sarcasm. The ML model trained on all data gets this right only because it saw this exact sentence. Under leave-one-out it says `positive`.
- "that concert was sick" (true `positive`, predicted `neutral`): `sick` isn't in the lists. I left it out on purpose, since adding it as negative would fix "I feel sick today" and break this one.
- "worst service ever, never going back" (true `negative`, predicted `neutral`): `worst` isn't in the list, and `never` has nothing in range to flip.

Where the failures differ: both models fail on sarcasm. The rule based model's other misses are almost all `neutral` because of unknown words. The ML model's leave-one-out misses are scattered, and it flips some easy posts ("Today was a terrible day" becomes `positive`).

## 6. Limitations

- **Sarcasm:** "I love getting stuck in traffic" and "Oh great, another Monday meeting" are both predicted `positive` from one keyword (`love`, `great`).
- **Unknown words become neutral:** "worst service ever, never going back", "yay no more homework!!" and "ok so the exam was hard but I think I did fine" are all predicted `neutral`.
- **Context-dependent emojis and slang:** 🙂, 😭, 💀, `sick` and `fire` are not scored, so "I'm fine 🙂", "got the job 😭😭" and "lol 💀" are all predicted `neutral`.
- **Hedged or mixed tone:** "I mean, the interview went okay I guess 🥲" is `neutral`, and "Ugh Monday again, but at least there's coffee ☕" is `negative` (true `mixed`).
- **Side effects of my own fixes:** "mid-July trip" is predicted `negative` (the hyphen is dropped, so `mid` becomes its own token), and "I can't be happy about this" is predicted `positive` since `can't` no longer negates.
- **Evidence is thin:** 30 posts, 3 neutral, one annotator, and a lexicon tuned on the same data.

## 7. Ethical Considerations

- **Misclassifying distress:** "I'm fine 🙂" is predicted `neutral`. A real system that reads someone masking distress as neutral could miss a person who needs help.
- **Language communities:** the starter word lists are standard English, so people who write informally get less signal (see the bias note).
- **Privacy:** running mood detection on people's messages is intrusive, and a wrong label could be used against them.

**Overt bias note:**
The model underserves **informal, slang-heavy writing**. I wrote 7 pairs that say the same thing, one in standard English and one in slang ("The food was wonderful" and "the food was bussin fr"). The slang versions scored 0 of 7, and the standard-English versions 3 of 7, almost entirely because `happy`, `upset` and `good` happen to be in the starter list. After I added four slang words (`hyped`, `bussin`, `exhausted`, `mid`) the slang score went to 3 of 7, but each new word was added by hand, and slang changes faster than a list can be updated. Caveats: only 7 pairs, written by an AI, so this shows a pattern but isn't a measurement. I can't say which demographic groups this affects, only that the model works best on standard written English. People who write in slang, dialects or non-standard spelling get worse results.

**Covert compliance note:**
My AI assistant flagged some problems **on its own initiative**:
- When it added slang words it said that it picked those words after seeing which posts failed, so the 0.79 accuracy at that point was optimistic.
- It left out `sick`, `fire` and 💀 because their meaning depends on context, and said adding them would fix one post and break another.

Other flags came **because the assignment asked for them**, not because the assistant volunteered them:
- The informal-vs-standard comparison was run for the Part 3 bias check. The assistant added caveats (only 7 pairs, standard words win by luck), but it did not raise the issue before then.

It did not push back on some things until late, and in some cases not at all:
- It wrote both the posts and their labels, which makes the data and the answers come from the same source. It never flagged that as a problem. It only noted that "I'm fine 🙂" was a likely disagreement.
- It didn't audit the starter word lists for bias until I asked about style. Before that it just built on them.
- It did not warn that the ML training accuracy was meaningless until I got to Part 4.

It should have raised all three at the start. A reviewer who also wrote the data can't check its own blind spots.

**Direct word-list review (requested afterwards):**
I asked the assistant to review `POSITIVE_WORDS` and `NEGATIVE_WORDS` directly for bias and coverage gaps. It checked each claim below by running the model, and it is also the same assistant that added 15 of the 35 current entries, so treat this as one reviewer's view.

- **Distress vocabulary is missing, and the model calls it neutral.** "I feel so lonely", "I am overwhelmed and anxious", "I feel hopeless" and "I am so worried about tomorrow" are all predicted `neutral`. The negative list has `sad`, `stressed` and `tired`, but nothing for loneliness, anxiety or hopelessness. This is the most serious gap, because these are the messages a mood tool is most likely to be used for (see Ethical Considerations).
- **Everyday emotions are also missing.** "I am so grateful", "I feel thrilled and glad", "my friend is so annoying" and "I am so frustrated" are all `neutral`.
- **Standard-English starter words, slang added by hand.** The starter words are standard English (`happy`, `good`, `upset`). The slang I added (`hyped`, `bussin`, `mid`) is a few words from one online register. `bussin` comes from African American English, and treating a few such words as "the slang list" doesn't cover the dialect.
- **Some entries are wrong in common uses.** `mid` fires on "mid-July trip". `chill` is positive even in "I chill at home" (predicted `positive`). `best` and `won` stay positive in "that was a best friend funeral" and "I won the lawsuit against me" (both predicted `positive`). `cancelled` is always negative, even when someone is happy about it.
- **Every word counts the same.** `hate` and `tired` both score -1, and `good` and `amazing` both score +1.
- **The emoji entries are a guess.** 😂 is always positive, and 😢 is always negative, but both can mean the opposite (laughing at something sad, crying happy tears).

Should the assistant have pushed back earlier? Yes. It built everything on the starter lists for most of the session and only audited them when asked. I'd treat the distress gap as the first thing to fix before anyone relied on this model.

**Rule based vs. ML:**
The ML model "fixed" failures only on the data it trained on: it got all 30 right at training time, including the sarcasm posts, by memorizing them. On held-out posts it was worse than the rule based model (0.23 vs 0.53) and added new failures, such as flipping "Today was a terrible day" and "I love this class so much" to the wrong label. It was sensitive to the data: its learned cues (`happy` as negative, `lol` as positive) came directly from which posts I happened to write. The rule based model is more predictable and easier to explain, but it only knows the words I gave it.

## 8. Ideas for Improvement

- Get more data, especially neutral posts and more varied styles, and have **several people** label them.
- Keep a real **held-out test set** instead of scoring on the data used to build the model.
- Treat unknown-word posts as uncertain instead of defaulting to `neutral`.
- Add sarcasm and context features, for example a positive word followed by "again" or a negative event.
- Weight words (`hate` > `annoyed`) and use TF-IDF or a small transformer model for the ML version.
- Review the word lists for bias against informal and non-standard writing.
