"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "best",
    "proud",
    "hopeful",
    "won",
    "🔥",
    "😂",
    "hyped",
    "bussin",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "ugh",
    "cancelled",
    "😤",
    "😞",
    "😢",
    "exhausted",
    "mid",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "Lowkey stressed about finals but kind of proud of myself for starting early 😅",
    "no cap this was the best pizza I've ever had 🍕🔥",
    "my flight got cancelled again 💀 love that for me",
    "Just got back from the store. Bought milk and eggs.",
    "omg we won!!! 😂😂 I'm literally shaking",
    "I mean, the interview went okay I guess 🥲",
    "Ugh Monday again, but at least there's coffee ☕",
    "bruh the wifi is down AGAIN 😤",
    "I love getting stuck in traffic",
    "Oh great, another Monday meeting",
    "that concert was sick",
    "your new fit is fire",
    "I feel sick today",
    "I'm fine 🙂",
    "got the job 😭😭",
    "lol 💀",
    "exhausted but proud of myself",
    "I hate how much I love this show",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",     # "Lowkey stressed about finals but kind of proud of myself for starting early 😅"
    "positive",  # "no cap this was the best pizza I've ever had 🍕🔥"
    "negative",  # "my flight got cancelled again 💀 love that for me"
    "neutral",   # "Just got back from the store. Bought milk and eggs."
    "positive",  # "omg we won!!! 😂😂 I'm literally shaking"
    "mixed",     # "I mean, the interview went okay I guess 🥲"
    "mixed",     # "Ugh Monday again, but at least there's coffee ☕"
    "negative",  # "bruh the wifi is down AGAIN 😤"
    "negative",  # "I love getting stuck in traffic"
    "negative",  # "Oh great, another Monday meeting"
    "positive",  # "that concert was sick"
    "positive",  # "your new fit is fire"
    "negative",  # "I feel sick today"
    "negative",  # "I'm fine 🙂"
    "positive",  # "got the job 😭😭"
    "positive",  # "lol 💀"
    "mixed",     # "exhausted but proud of myself"
    "mixed",     # "I hate how much I love this show"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
