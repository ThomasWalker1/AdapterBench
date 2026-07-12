"""Content prompts for the reward-tilting image setting.

The noise-adapter is prompt-agnostic (a single shared adapter, prompt-conditioned only via
the frozen generator), so training and evaluation use *disjoint* prompt sets: the eval gain
is thus measured on prompts the adapter never saw, testing that the reward tilt generalizes
across content rather than overfitting the training prompts.
"""

from __future__ import annotations

TRAIN_PROMPTS = [
    "a photo of a house", "a photo of a car", "a portrait of a woman", "a bowl of fruit",
    "a mountain landscape", "a cat sitting on a chair", "a city street at night", "a wooden table",
    "a dog running in a field", "a cup of coffee", "a sailboat on the sea", "a plate of food",
    "a forest in autumn", "a bicycle leaning on a wall", "a bird on a branch", "a vase of flowers",
    "a train at a station", "a lighthouse by the ocean", "a bakery storefront", "a child flying a kite",
    "a horse in a meadow", "a snowy village", "a market stall of vegetables", "a desk with a laptop",
]

EVAL_PROMPTS = [
    "a red apple on a table", "a castle on a hill", "a bustling harbor", "a bouquet of roses",
    "a fox in the snow", "a plate of pasta", "a violin on a chair", "a hot air balloon over fields",
]
