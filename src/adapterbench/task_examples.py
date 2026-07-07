"""Build small TaskExample sets for the Text-to-LoRA Gemma-2-2B benchmark tasks.

Prompt templates and 3-shot in-context examples below are copied verbatim from
upstream Text-to-LoRA's own evaluation code (``hyper_llm_modulator.vllm_eval``'s
``ARC_TEMPLATE``/``HSWAG_TEMPLATE``/``BOOLQ_TEMPLATE``/gsm8k template and
``hyper_llm_modulator.utils.eval_prompts.IN_CONTEXT_EXAMPLES``), not re-derived —
the paper's own published numbers were produced by generating an answer against
these exact prompts and extracting a leading choice letter/digit (or a loose
true/false keyword for BoolQ), not by scoring choice log-likelihoods. Reproducing
that generation+extraction protocol (see ``hf_downstream_evaluator.py``) is what
makes our numbers comparable to the paper's at all.

Condition text (used only to condition the hypernetwork's generation, never shown
to the interpreter at eval time) is sourced from a released checkpoint's own
``args.yaml`` (``eval_ds_info``). One description variant per task is used, chosen
deterministically, and recorded in each example's metadata.
"""

from __future__ import annotations

from pathlib import Path
import re

import yaml

from .contracts import TaskExample

# (dataset_id, config_name, split)
DATASET_SOURCES: dict[str, tuple[str, str | None, str]] = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "boolq": ("google/boolq", None, "validation"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
}

_GSM8K_ANSWER_RE = re.compile(r"####\s*([\-0-9,\.]+)")

_TASK_TEMPLATES = {
    "arc": (
        "Answer the question below by choosing the correct choice.\n\n"
        "{question}\n\n"
        "{choices[label][0]}: {choices[text][0]}\n{choices[label][1]}: {choices[text][1]}\n"
        "{choices[label][2]}: {choices[text][2]}\n{choices[label][3]}: {choices[text][3]}\n\n"
        "You must respond with the letter corresponding to the correct choice without any explanation."
    ),
    "hellaswag": (
        "You are provided with an incomplete passage below as well as 4 choices of continuation "
        "with only one of them being the correct ending. "
        "Treat the endings as being labelled 0, 1, 2, 3 in order.\n\n"
        "Passage: {ctx}\n\n"
        "0: {endings[0]}\n"
        "1: {endings[1]}\n"
        "2: {endings[2]}\n"
        "3: {endings[3]}\n\n"
        "You must respond with the only number corresponding to the correct ending (0,1,2,3) for the passage "
        "without any explanation."
    ),
    "boolq": "{passage}\n\nQuestion: {question}?\n\nPlease answer with either `true` or `false` without any explanation.",
    "gsm8k": "Please answer the following question: {question}\n\n",
}

# 3-shot ICL, used only when ``use_icl=True`` (matches the T2L paper's Table 8/Gemma
# eval protocol, which prepends these to every prompt and forces an "Answer:"
# generation prefix — see HFDownstreamEvaluator's ``use_icl``/prefill handling).
_IN_CONTEXT_EXAMPLES = {
    "gsm8k": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

## Example 2
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

## Example 3
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5
""",
    "boolq": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Passage: Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.

Question: do iran and afghanistan speak the same language

Answer: True

## Example 2
Passage: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable.

Question: do good samaritan laws protect those who help at an accident

Answer: True

## Example 3
Passage: Windows Movie Maker (formerly known as Windows Live Movie Maker in Windows 7) is a discontinued video editing software by Microsoft. It is a part of Windows Essentials software suite and offers the ability to create and edit videos as well as to publish them on OneDrive, Facebook, Vimeo, YouTube, and Flickr.

Question: is windows movie maker part of windows essentials

Answer: True
""",
    "hellaswag": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Passage: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then

0: , the man adds wax to the windshield and cuts it.

1: , a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.

2: , the man puts on a christmas coat, knitted with netting.

3: , the man continues removing the snow on his car.

Answer: 3

## Example 2
Passage: A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. the pans

0: contain egg yolks and baking soda.

1: are then sprinkled with brown sugar.

2: are placed in a strainer on the counter.

3: are filled with pastries and loaded into the oven.

Answer: 3

## Example 3
Passage: A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. The pans are filled with pastries and loaded into the oven. a knife

0: is seen moving on a board and cutting out its contents.

1: hits the peeled cheesecake, followed by sliced custard and still cooked ice cream.

2: etches a shape into the inside of the baked pans.

3: is used to cut cylinder shaped dough into rounds.

Answer: 3
""",
    "arc_easy": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: Which factor will most likely cause a person to develop a fever?
A: a leg muscle relaxing after exercise
B: a bacterial population in the bloodstream
C: several viral particles on the skin
D: carbohydrates being digested in the stomach

Answer: B

## Example 2
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
A: carbon dioxide
B: food
C: protection
D: water

Answer: B

## Example 3
Question: When a switch is used in an electrical circuit, the switch can
A: cause the charge to build.
B: increase and decrease the voltage.
C: cause the current to change direction.
D: stop and start the flow of current.

Answer: D
""",
    "arc_challenge": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
A: dry palms
B: wet palms
C: palms covered with oil
D: palms covered with lotion

Answer: A

## Example 2
Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?
A: The refrigerator door is smooth.
B: The refrigerator door contains iron.
C: The refrigerator door is a good conductor.
D: The refrigerator door has electric wires in it.

Answer: B

## Example 3
Question: A fold observed in layers of sedimentary rock most likely resulted from the
A: cooling of flowing magma.
B: converging of crustal plates.
C: deposition of river sediments.
D: solution of carbonate minerals.

Answer: B
""",
}


def load_task_descriptions(args_yaml_path: str | Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(Path(args_yaml_path).read_text())
    return {task: info["descriptions"] for task, info in payload["eval_ds_info"].items()}


def build_conditions(
    descriptions: dict[str, list[str]], task_ids: list[str], variant: int = 0
) -> dict[str, str]:
    return {task_id: descriptions[task_id][variant] for task_id in task_ids}


def _icl_prefix(task_id: str, use_icl: bool) -> str:
    return f"{_IN_CONTEXT_EXAMPLES[task_id]}\n\n" if use_icl else ""


def _pad_arc_choices(choices: dict) -> dict:
    """Pad to exactly 4 choices with an "N/A" filler, matching upstream's own ARC
    preprocessing (some ARC-Easy questions have only 3 answer choices, but the
    template hardcodes 4 slots)."""
    text = list(choices["text"])
    label = list(choices["label"])
    n_to_fill = 4 - len(text)
    if n_to_fill > 0:
        text += ["N/A"] * n_to_fill
        if label and label[0].isdigit():
            label += [str(len(label) + i + 1) for i in range(n_to_fill)]
        else:
            label += ["A", "B", "C", "D"][len(label) : len(label) + n_to_fill]
    return {"label": label, "text": text}


def _arc_examples(task_id: str, condition: str, rows, limit: int, variant: int, use_icl: bool = False) -> list[TaskExample]:
    examples = []
    prefix = _icl_prefix(task_id, use_icl)
    for i, row in enumerate(rows):
        if len(examples) >= limit:
            break
        # Validate against the row's own labels before padding: the ABCD padding
        # below always fills up to "D", which would otherwise make an out-of-range
        # answerKey look spuriously valid.
        if row["answerKey"] not in row["choices"]["label"]:
            continue
        choices = _pad_arc_choices(row["choices"])
        prompt = _TASK_TEMPLATES["arc"].format(question=row["question"], choices=choices)
        examples.append(
            TaskExample(
                task_id=f"{task_id}::{i}",
                condition=condition,
                input_text=prefix + prompt,
                target_text=row["answerKey"],
                family=task_id,
                metadata={
                    "choices": choices["text"],
                    "answer_index": choices["label"].index(row["answerKey"]),
                    "condition_variant": variant,
                },
            )
        )
    return examples


def _boolq_examples(task_id: str, condition: str, rows, limit: int, variant: int, use_icl: bool = False) -> list[TaskExample]:
    examples = []
    prefix = _icl_prefix(task_id, use_icl)
    for i, row in enumerate(rows):
        if len(examples) >= limit:
            break
        prompt = _TASK_TEMPLATES["boolq"].format(passage=row["passage"], question=row["question"])
        answer_index = 1 if row["answer"] else 0
        examples.append(
            TaskExample(
                task_id=f"{task_id}::{i}",
                condition=condition,
                input_text=prefix + prompt,
                target_text=["false", "true"][answer_index],
                family=task_id,
                metadata={"choices": ["false", "true"], "answer_index": answer_index, "condition_variant": variant},
            )
        )
    return examples


def _hellaswag_examples(
    task_id: str, condition: str, rows, limit: int, variant: int, use_icl: bool = False
) -> list[TaskExample]:
    examples = []
    prefix = _icl_prefix(task_id, use_icl)
    for i, row in enumerate(rows):
        if len(examples) >= limit:
            break
        endings = row["endings"]
        try:
            answer_index = int(row["label"])
        except (TypeError, ValueError):
            continue
        if not (0 <= answer_index < len(endings)):
            continue
        prompt = _TASK_TEMPLATES["hellaswag"].format(ctx=row["ctx"], endings=endings)
        examples.append(
            TaskExample(
                task_id=f"{task_id}::{i}",
                condition=condition,
                input_text=prefix + prompt,
                target_text=str(answer_index),
                family=task_id,
                metadata={"choices": endings, "answer_index": answer_index, "condition_variant": variant},
            )
        )
    return examples


def _gsm8k_examples(
    task_id: str, condition: str, rows, limit: int, variant: int, use_icl: bool = False
) -> list[TaskExample]:
    examples = []
    prefix = _icl_prefix(task_id, use_icl)
    for i, row in enumerate(rows):
        if len(examples) >= limit:
            break
        match = _GSM8K_ANSWER_RE.search(row["answer"])
        if not match:
            continue
        prompt = _TASK_TEMPLATES["gsm8k"].format(question=row["question"])
        examples.append(
            TaskExample(
                task_id=f"{task_id}::{i}",
                condition=condition,
                input_text=prefix + prompt,
                target_text=match.group(1).replace(",", ""),
                family=task_id,
                metadata={"condition_variant": variant},
            )
        )
    return examples


_BUILDERS = {
    "arc_easy": _arc_examples,
    "arc_challenge": _arc_examples,
    "boolq": _boolq_examples,
    "hellaswag": _hellaswag_examples,
    "gsm8k": _gsm8k_examples,
}


def build_task_examples(
    task_id: str, descriptions: dict[str, list[str]], limit: int, variant: int = 0, use_icl: bool = False
) -> list[TaskExample]:
    from datasets import load_dataset

    if task_id not in DATASET_SOURCES:
        raise ValueError(f"unknown task_id: {task_id}")
    condition = descriptions[task_id][variant]
    dataset_id, config, split = DATASET_SOURCES[task_id]
    rows = load_dataset(dataset_id, config, split=split) if config else load_dataset(dataset_id, split=split)
    return _BUILDERS[task_id](task_id, condition, rows, limit, variant, use_icl)


def build_all_task_examples(
    task_ids: list[str], descriptions: dict[str, list[str]], limit: int, variant: int = 0, use_icl: bool = False
) -> dict[str, list[TaskExample]]:
    return {
        task_id: build_task_examples(task_id, descriptions, limit, variant, use_icl) for task_id in task_ids
    }
