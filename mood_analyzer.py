# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the meaning of the sentiment word right after them.
NEGATORS = {
    "not", "no", "never",
    "don't", "dont", "isn't", "isnt",
    "won't", "wont", "didn't", "didnt", "wasn't", "wasnt",
}
# "can't" is left out on purpose: it is mostly an idiom ("can't believe how
# stressed I am", "can't wait") that does not negate the word that follows.

# Punctuation kept as tokens so negation cannot leak across a clause boundary.
CLAUSE_BREAKS = {",", ".", ";", ":", "!", "?"}

# A negator flips sentiment words up to this many tokens after it.
NEGATION_WINDOW = 3


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        # Phone keyboards type curly apostrophes ("don’t"); normalize them.
        cleaned = text.strip().lower().replace("’", "'")
        # Words (keeping apostrophes) or single non-word symbols such as emojis.
        raw_tokens = re.findall(r"[a-z0-9']+|[^\w\s]", cleaned)
        # Drop punctuation except clause breaks, which scope negation.
        tokens = [
            t for t in raw_tokens
            if t not in string.punctuation or t in CLAUSE_BREAKS
        ]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        positive_count, negative_count = self._count_hits(self.preprocess(text))
        return positive_count - negative_count

    def _count_hits(self, tokens: List[str]) -> Tuple[int, int]:
        """
        Count positive and negative hits, flipping a word's polarity when a
        negator appears within NEGATION_WINDOW tokens before it ("not very
        happy" counts as negative). A clause break or "but" ends the negation.
        """
        positive_count = 0
        negative_count = 0
        negation_left = 0

        for token in tokens:
            if token in NEGATORS:
                negation_left = NEGATION_WINDOW
                continue
            if token in CLAUSE_BREAKS or token == "but":
                negation_left = 0
                continue

            in_negation = negation_left > 0
            if negation_left > 0:
                negation_left -= 1

            if token in self.positive_words:
                polarity = 1
            elif token in self.negative_words:
                polarity = -1
            else:
                continue

            if in_negation:
                polarity = -polarity

            if polarity > 0:
                positive_count += 1
            else:
                negative_count += 1

        return positive_count, negative_count

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        positive_count, negative_count = self._count_hits(self.preprocess(text))
        score = positive_count - negative_count

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        # A zero score with hits on both sides is mixed, not neutral.
        if positive_count > 0 and negative_count > 0:
            return "mixed"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
