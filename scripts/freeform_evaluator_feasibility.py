"""Exploratory check of a cheap aspect-aware sentiment evaluator.

Fits only on MAMS-ATSA train. MAMS val/test and SemEval 2014 Restaurants test
are evaluation sets. The script writes metrics and labels only; it never writes
review text or a fitted model.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

MAMS = Path(".context/datasets/MAMS-for-ABSA/data/MAMS-ATSA/raw")
SEMEVAL = Path(".context/datasets/semeval2014/Restaurants_Test_Gold.xml")
OUT = Path("results/freeform-evaluator-feasibility-v1")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aspect_input(aspect, sentence):
    """Serialize an aspect and its review without exposing labels to features."""
    return f"Aspect: {aspect}. Review: {sentence}"


def read_mams(split):
    rows = []
    root = ET.parse(MAMS / f"{split}.xml").getroot()
    for sentence in root.findall(".//sentence"):
        text = (sentence.findtext("text") or "").strip()
        for aspect in sentence.findall("./aspectTerms/aspectTerm"):
            label = aspect.get("polarity")
            if label in {"positive", "negative"}:
                rows.append(
                    {
                        "id": f"{split}:{len(rows)}",
                        "input": aspect_input(aspect.get("term", ""), text),
                        "label": label,
                        "sentence_id": sentence.get("id", ""),
                    }
                )
    return rows


def read_semeval():
    rows = []
    root = ET.parse(SEMEVAL).getroot()
    for sentence in root.findall(".//sentence"):
        text = (sentence.findtext("text") or "").strip()
        categories = sentence.find("aspectCategories")
        if not text or categories is None:
            continue
        labels = {}
        for category in categories.findall("aspectCategory"):
            name, polarity = category.get("category"), category.get("polarity")
            if name in {"food", "service", "price"} and polarity in {
                "positive",
                "negative",
            }:
                labels[name] = polarity
        if len(labels) < 2:
            continue
        sentence_id = sentence.get("id", "")
        for name, polarity in labels.items():
            rows.append(
                {
                    "id": f"{sentence_id}|{name}",
                    "input": aspect_input(name, text),
                    "label": polarity,
                    "sentence_id": sentence_id,
                }
            )
    return rows


def score_rows(model, rows):
    expected = [row["label"] for row in rows]
    predicted = model.predict([row["input"] for row in rows]).tolist()
    matrix = confusion_matrix(expected, predicted, labels=["negative", "positive"])
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=["negative", "positive"], average="macro")
        ),
        "negative_recall": float(matrix[0, 0] / matrix[0].sum()),
        "positive_recall": float(matrix[1, 1] / matrix[1].sum()),
        "confusion_matrix_negative_positive": matrix.tolist(),
        "items": [
            {
                "id": row["id"],
                "sentence_id": row["sentence_id"],
                "gold": row["label"],
                "prediction": prediction,
            }
            for row, prediction in zip(rows, predicted)
        ],
    }


def run(output=OUT):
    train = read_mams("train")
    validations = {split: read_mams(split) for split in ("val", "test")}
    validations["semeval2014_restaurants_test"] = read_semeval()
    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=150_000,
        ),
        LogisticRegression(C=1.0, max_iter=1_500, class_weight="balanced"),
    )
    model.fit([row["input"] for row in train], [row["label"] for row in train])
    report = {
        "experiment": "exploratory_freeform_evaluator_feasibility",
        "fit_data": "MAMS-ATSA train; positive and negative aspects only",
        "features": "TF-IDF word 1-2 grams of aspect term/category plus full sentence",
        "classifier": "balanced logistic regression, C=1.0",
        "evaluation": {split: score_rows(model, rows) for split, rows in validations.items()},
        "data_sha256": {
            **{
                f"mams_atsa_{split}": sha256(MAMS / f"{split}.xml")
                for split in ("train", "val", "test")
            },
            "semeval2014_restaurants_test": sha256(SEMEVAL),
        },
        "interpretation": (
            "Source-review classification is only a proxy validation for judging generated text. "
            "No generated answer was evaluated in this pilot."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "evaluation"}, indent=2))
    for split, metrics in report["evaluation"].items():
        print(
            f"{split}: n={metrics['n']} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"negative_recall={metrics['negative_recall']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
