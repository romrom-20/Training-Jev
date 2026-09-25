"""Pure data preparation helpers for Experiment 040."""

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

SOURCE = Path(".context/trabl/test.jsonl")
SOURCE_REVISION = "e9329d05a722000eb7cabaab97c59049d52b94fd"
SOURCE_SHA256 = "3632412d3e8003364ed67e6877777bb51a474f30917037d36a8f7cedbd1e7884"
STIMULI = Path("docs/experiments/040-stimuli.json")
DATASETS = "https://huggingface.co/datasets/Booking-com/absa-dataset"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_source(path=SOURCE):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; download the pinned TRABL test split")
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("TRABL test split hash differs from the preregistered revision")
    rows = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        labels = ast.literal_eval(row["label"])
        if not isinstance(labels, list):
            raise ValueError(f"Malformed annotation list on source row {line_index}")
        rows.append(
            {
                "line_index": line_index,
                "source_id": row["id_source"],
                "annotator": int(row["annotator"]),
                "text": row["text"],
                "labels": labels,
            }
        )
    return rows


def _normalized(value):
    return str(value or "").strip().casefold()


def _tuple_key(annotation):
    return tuple(
        _normalized(annotation.get(field))
        for field in ("aspect term", "aspect category", "opinion span", "sentiment", "snippet")
    )


def _unique_span(text, phrase):
    phrase = str(phrase or "").strip()
    if not phrase:
        return None
    matches = list(re.finditer(re.escape(phrase), text, flags=re.IGNORECASE))
    return (matches[0].start(), matches[0].end()) if len(matches) == 1 else None


def _eligible(annotation, text):
    polarity = _normalized(annotation.get("sentiment"))
    aspect = str(annotation.get("aspect term") or "").strip()
    category = str(annotation.get("aspect category") or "").strip()
    opinion = str(annotation.get("opinion span") or "").strip()
    snippet = str(annotation.get("snippet") or "").strip()
    aspect_span = _unique_span(text, aspect)
    opinion_span = _unique_span(text, opinion)
    if polarity not in {"positive", "negative"} or not aspect_span or not opinion_span:
        return None
    if not snippet or snippet.casefold() not in text.casefold():
        return None
    if aspect_span[0] >= opinion_span[0]:
        return None
    return {
        "aspect": aspect,
        "category": category,
        "gold": polarity,
        "opinion": opinion,
        "opinion_span": list(opinion_span),
        "aspect_span": list(aspect_span),
    }


def select_pairs(rows):
    """Return one exact-agreement opposite-polarity pair per eligible review."""
    grouped = defaultdict(dict)
    for row in rows:
        if row["annotator"] in grouped[row["source_id"]]:
            raise ValueError("Duplicate annotator row for one source review")
        grouped[row["source_id"]][row["annotator"]] = row

    selected = []
    for source_id, annotators in grouped.items():
        if set(annotators) != {1, 2}:
            continue
        first, second = annotators[1], annotators[2]
        if first["text"] != second["text"]:
            continue
        by_key_1 = {_tuple_key(label): label for label in first["labels"]}
        by_key_2 = {_tuple_key(label): label for label in second["labels"]}
        consensus = [by_key_1[key] for key in sorted(set(by_key_1) & set(by_key_2))]
        eligible = [candidate for label in consensus if (candidate := _eligible(label, first["text"]))]
        positives = sorted(
            (candidate for candidate in eligible if candidate["gold"] == "positive"),
            key=lambda candidate: (candidate["opinion_span"][0], candidate["aspect"].casefold()),
        )
        negatives = sorted(
            (candidate for candidate in eligible if candidate["gold"] == "negative"),
            key=lambda candidate: (candidate["opinion_span"][0], candidate["aspect"].casefold()),
        )
        pair = next(
            (
                (positive, negative)
                for positive in positives
                for negative in negatives
                if positive["aspect"].casefold() != negative["aspect"].casefold()
                and max(positive["opinion_span"][0], negative["opinion_span"][0])
                >= min(positive["opinion_span"][1], negative["opinion_span"][1])
            ),
            None,
        )
        if pair is None:
            continue
        ordered = sorted(pair, key=lambda candidate: candidate["opinion_span"][0])
        cluster_number = min(first["line_index"], second["line_index"])
        pair_items = []
        for role_index, candidate in enumerate(ordered):
            prior = ordered[role_index - 1] if role_index else None
            item = {
                "id": f"trabl-review-{cluster_number:04d}-{candidate['gold'][:3]}",
                "cluster_id": f"trabl-review-{cluster_number:04d}",
                "source_rows": [first["line_index"], second["line_index"]],
                "role": "later" if prior else "earlier",
                "aspect": candidate["aspect"],
                "category": candidate["category"],
                "gold": candidate["gold"],
                "aspect_span": candidate["aspect_span"],
                "opinion_span": candidate["opinion_span"],
                "prior_gold": prior["gold"] if prior else None,
                "prior_opinion_span": prior["opinion_span"] if prior else None,
                "_text": first["text"],
                "_source_id": source_id,
            }
            pair_items.append(item)
        if pair_items[1]["prior_gold"] == pair_items[1]["gold"]:
            raise ValueError("Selected target pair does not have opposite polarities")
        selected.extend(pair_items)
    return sorted(selected, key=lambda item: (item["source_rows"][0], item["role"]))


def public_stimuli(items):
    return [
        {
            key: item[key]
            for key in (
                "id",
                "cluster_id",
                "source_rows",
                "role",
                "category",
                "gold",
                "aspect_span",
                "opinion_span",
                "prior_gold",
                "prior_opinion_span",
            )
        }
        for item in items
    ]


def make_jobs(items):
    jobs = []
    for item in items:
        text = item["_text"]
        opinion_start, opinion_end = item["opinion_span"]
        common = {
            key: item[key]
            for key in ("id", "cluster_id", "role", "aspect", "category", "gold", "prior_gold")
        }
        jobs.extend(
            [
                common | {"condition": "aspect_only", "visible_text": ""},
                common
                | {
                    "condition": "natural_prefix",
                    "visible_text": text[:opinion_start].rstrip(),
                },
                common
                | {
                    "condition": "opinion_visible",
                    "visible_text": text[:opinion_end].rstrip(),
                },
            ]
        )
        if item["role"] == "later":
            prior_start, prior_end = item["prior_opinion_span"]
            prefix = text[:opinion_start]
            if prior_start < 0 or prior_end > len(prefix) or prior_start >= prior_end:
                raise ValueError("Prior opinion span is not visible in the later-target prefix")
            deleted = (prefix[:prior_start] + prefix[prior_end:]).strip()
            jobs.append(
                common
                | {
                    "condition": "prior_opinion_deleted",
                    "visible_text": deleted,
                }
            )
    return jobs
