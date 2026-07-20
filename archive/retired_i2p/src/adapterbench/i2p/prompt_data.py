"""Content prompts for the reward-tilting image setting.

The noise-adapter is prompt-agnostic (a single shared adapter, prompt-conditioned only via
the frozen generator), so training and evaluation use *disjoint* prompt sets: the eval gain
is thus measured on prompts the adapter never saw, testing that the reward tilt generalizes
across content rather than overfitting the training prompts.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def build_prompt_conditioning_probe_prompts(
    *, train_size: int = 256, eval_size: int = 64, seed: int = 20260719
) -> tuple[list[str], list[str]]:
    """Build a deterministic, disjoint prompt set for the mechanism-free probe.

    These are deliberately ordinary text-to-image requests, not hard
    requirements or reward-conditioned prompts.  The cross product supplies a
    few hundred varied examples without introducing a new dataset dependency.
    A deterministic shuffle prevents the held-out slice from corresponding to
    one style or scene.
    """
    import random

    subjects = [
        "a tabby cat", "a golden retriever", "a red fox", "a barn owl",
        "a vintage bicycle", "a steam locomotive", "a wooden sailboat", "a yellow taxi",
        "a glass teapot", "a bowl of oranges", "a violin", "an antique camera",
        "a small cottage", "a stone lighthouse", "a greenhouse", "a mountain observatory",
        "a violinist", "a pastry chef", "an astronaut", "a child with a kite",
    ]
    settings = [
        "beside a quiet lake", "in a sunlit studio", "on a rainy city street",
        "under a starry sky", "in an autumn forest", "on a snowy hillside",
        "at a crowded market", "near the ocean at dawn",
    ]
    renderings = [
        "a detailed photograph of", "a cinematic photograph of", "a watercolor painting of",
        "an oil painting of", "a colored-pencil drawing of", "a polished 3D render of",
    ]
    prompts = [
        f"{rendering} {subject} {setting}"
        for subject in subjects
        for setting in settings
        for rendering in renderings
    ]
    needed = train_size + eval_size
    if needed > len(prompts):
        raise ValueError(f"requested {needed} prompts, but only {len(prompts)} are available")
    random.Random(seed).shuffle(prompts)
    return prompts[:train_size], prompts[train_size:needed]


@dataclass(frozen=True)
class SelectiveEraseScene:
    """One scene with two valid, mutually exclusive erase requests."""

    prompt: str
    first_object: str
    second_object: str

    def tasks(self) -> tuple[tuple[str, str], tuple[str, str]]:
        """Return ``(erase, retain)`` in both directions."""
        return (
            (self.first_object, self.second_object),
            (self.second_object, self.first_object),
        )


def build_selective_erasure_probe_scenes(
    *, train_size: int = 128, eval_size: int = 32, seed: int = 20260719
) -> tuple[list[SelectiveEraseScene], list[SelectiveEraseScene]]:
    """Paired scenes for the target-conditioned selective-erasure probe.

    Every returned scene is evaluated twice with the *identical* generator
    prompt, once per erase target.  Train and evaluation use disjoint prompt
    templates while sharing the object vocabulary, so the probe tests
    compositional prompt generalization without conflating it with zero-shot
    object-name generalization.
    """
    import itertools
    import random

    objects = [
        "cat", "dog", "horse", "elephant",
        "car", "bicycle", "airplane", "sailboat",
        "apple", "banana", "chair", "clock",
    ]
    train_templates = [
        "a studio photograph with a {left} on the left and a {right} on the right",
        "a detailed image containing both a {left} and a {right}, clearly separated",
        "a cinematic scene showing a {left} beside a {right}",
        "a clean product-style photograph of a {left} together with a {right}",
    ]
    eval_templates = [
        "a bright photograph showing a {left} next to a {right}",
        "a realistic scene containing a {left} and, nearby, a {right}",
    ]
    pairs = list(itertools.combinations(objects, 2))
    rng = random.Random(seed)
    rng.shuffle(pairs)

    def make(templates: list[str]) -> list[SelectiveEraseScene]:
        scenes = [
            SelectiveEraseScene(
                prompt=template.format(left=left, right=right),
                first_object=left,
                second_object=right,
            )
            for template in templates
            for left, right in pairs
        ]
        rng.shuffle(scenes)
        return scenes

    train_scenes = make(train_templates)
    eval_scenes = make(eval_templates)
    if train_size > len(train_scenes) or eval_size > len(eval_scenes):
        raise ValueError(
            f"requested train/eval={train_size}/{eval_size}, available="
            f"{len(train_scenes)}/{len(eval_scenes)}"
        )
    train = train_scenes[:train_size]
    train_prompts = {scene.prompt for scene in train}
    eval_ = [scene for scene in eval_scenes if scene.prompt not in train_prompts][:eval_size]
    if len(eval_) != eval_size:
        raise RuntimeError("not enough disjoint selective-erasure evaluation scenes")
    return train, eval_
