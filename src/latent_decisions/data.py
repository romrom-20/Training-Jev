"""Fully crossed factors; split by scenario, never by activation/question row."""

import hashlib
import itertools
import random

FACTORS = ("color", "shape", "animal")
VALUES = (("blue", "red"), ("square", "circle"), ("dog", "cat"))
QUESTIONS = (
    (
        "Is the configured color red?",
        "Is the configured shape circle?",
        "Is the configured animal cat?",
    ),
    (
        "Was red selected as the color?",
        "Was circle selected as the shape?",
        "Was cat selected as the animal?",
    ),
)
TEMPLATES = (
    "Configuration for record {record}: {settings}. Reply with the configured color only, in lowercase. Do not output the shape or animal.",
    "Record {record} has these settings: {settings}. Your answer must be just the lowercase color word. The other settings are irrelevant to your answer.",
    "Read this registry entry ({record}): {settings}. Return only the value of its color field, in lowercase, with no explanation.",
)


def make_dataset(counts, seed=42):
    rng = random.Random(seed)
    rows = []
    for split, n in counts.items():
        if n < 1:
            raise ValueError("Each split needs at least one group")
        for group in range(n):
            # A group contains all eight label combinations; label prevalence is exactly 1/2.
            record = hashlib.sha256(f"{seed}/{split}/{group}".encode()).hexdigest()[:10]
            template = 2 if split == "ood" else group % 2
            order = list(range(3))
            rng.shuffle(order)
            for bits in itertools.product((0, 1), repeat=3):
                settings = "; ".join(f"{FACTORS[j]}={VALUES[j][bits[j]]}" for j in order)
                rows.append(
                    dict(
                        id=f"{split}-{group}-" + "".join(map(str, bits)),
                        group=f"{split}-{group}",
                        split=split,
                        template=template,
                        system=TEMPLATES[template].format(record=record, settings=settings),
                        user="Read the record and give your answer.",
                        labels=list(bits),
                    )
                )
    return rows


def validate_splits(rows):
    seen = {}
    prompts = set()
    for row in rows:
        if row["group"] in seen and seen[row["group"]] != row["split"]:
            raise ValueError("A scenario group leaked across splits")
        seen[row["group"]] = row["split"]
        prompt = (row["system"], row["user"])
        if prompt in prompts:
            raise ValueError("Duplicate prompt")
        prompts.add(prompt)
