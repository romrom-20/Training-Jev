"""Run a leave-one-SemEval-subset-out lexical baseline for Experiment 039."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

from run_natural_opinion_span_abstention_037 import (
    PROTOCOL as PROTOCOL_037,
)
from run_natural_opinion_span_abstention_037 import (
    SOURCE_REVISION,
    SOURCE_SHA256,
    load_source_items,
    sha256,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROTOCOL = Path("docs/experiments/039-cross-subset-context-lexical-baseline.md")
STIMULI_INDEX = Path("docs/experiments/037-stimuli.json")
PARENT_037 = Path("results/natural-opinion-span-abstention-v1")
PARENT_038 = Path("results/aspect-only-prior-control-v1")
OUT = Path(".context/lexical-context-generalization-039")
DATASETS = ("14res", "14lap", "15res", "16res")
SEED = 20260939
EXPECTED_PROTOCOL_SHA256 = "c92a8c840f9e11214ec56243a662df07e1eb617f8cf960d6878d3d3515ce6bdf"
EXPECTED_037_PROTOCOL_SHA256 = "820fa15fb757d75d2191e0dfea022364e371fe6a4eeed6c52317e3d33b6c346f"
EXPECTED_STIMULI_SHA256 = "22dea039b2b235f8cfeb09b3f104d811f7097176f4a640d37b6fc3b9ef27d2d6"


def build_features(item):
    tokens = item["_tokens"]
    opinion_start = min(item["opinion_token_indices"])
    aspect_indices = set(item["aspect_token_indices"])
    prefix = tokens[:opinion_start]
    masked = []
    for index, token in enumerate(prefix):
        if index not in aspect_indices:
            masked.append(token)
        elif index == min(aspect_indices):
            masked.append("ASPECT")
    aspect = item["_aspect"]
    return {
        "aspect_only": f"aspect {aspect}",
        "masked_context": "review prefix " + " ".join(masked),
        "context_plus_aspect": f"aspect {aspect} review prefix {' '.join(prefix)}",
    }


def load_frozen_rows():
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Experiment 039 protocol changed after freeze")
    if sha256(PROTOCOL_037) != EXPECTED_037_PROTOCOL_SHA256:
        raise ValueError("Parent Experiment 037 protocol hash changed")
    if sha256(STIMULI_INDEX) != EXPECTED_STIMULI_SHA256:
        raise ValueError("Experiment 037 ID/span index changed")
    items = load_source_items()
    public_index = json.loads(STIMULI_INDEX.read_text())
    public_items = [
        {key: item[key] for key in public_index[0]}
        for item in items
    ]
    if public_items != public_index:
        raise ValueError("Current ASTE source selection differs from the frozen index")
    return items


def vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        norm="l2",
    )


def classifier():
    return LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    items = load_frozen_rows()
    parent_037_manifest = json.loads((PARENT_037 / "manifest.json").read_text())
    parent_038_manifest = json.loads((PARENT_038 / "manifest.json").read_text())
    if sha256(PARENT_037 / "predictions.json") != parent_037_manifest["predictions_sha256"]:
        raise ValueError("Experiment 037 parent prediction hash mismatch")
    if sha256(PARENT_038 / "predictions.json") != parent_038_manifest["predictions_sha256"]:
        raise ValueError("Experiment 038 parent prediction hash mismatch")

    features = [build_features(item) for item in items]
    gold = [item["polarity"] for item in items]
    datasets = [item["dataset"] for item in items]
    ids = [item["id"] for item in items]
    predictions = {variant: {} for variant in ("aspect_only", "masked_context", "context_plus_aspect")}
    fold_sizes = {}

    for test_dataset in DATASETS:
        train_indices = [i for i, dataset in enumerate(datasets) if dataset != test_dataset]
        test_indices = [i for i, dataset in enumerate(datasets) if dataset == test_dataset]
        if not train_indices or not test_indices:
            raise ValueError(f"Empty train/test fold for {test_dataset}")
        if set(gold[i] for i in train_indices) != {"positive", "negative"}:
            raise ValueError(f"Training fold is missing a class for {test_dataset}")
        if set(gold[i] for i in test_indices) != {"positive", "negative"}:
            raise ValueError(f"Test fold is missing a class for {test_dataset}")
        fold_sizes[test_dataset] = {"train": len(train_indices), "test": len(test_indices)}

        for variant in predictions:
            train_text = [features[i][variant] for i in train_indices]
            test_text = [features[i][variant] for i in test_indices]
            fitted_vectorizer = vectorizer()
            train_matrix = fitted_vectorizer.fit_transform(train_text)
            test_matrix = fitted_vectorizer.transform(test_text)
            fitted_classifier = classifier()
            fitted_classifier.fit(train_matrix, [gold[i] for i in train_indices])
            labels = fitted_classifier.predict(test_matrix)
            predictions[variant].update(
                {ids[i]: str(label) for i, label in zip(test_indices, labels)}
            )
        print(f"039 held-out {test_dataset}: {len(test_indices)} predictions per variant", flush=True)

    rows = [
        {
            "id": item["id"],
            "cluster_id": item["cluster_id"],
            "dataset": item["dataset"],
            "gold": item["polarity"],
            "aspect_only_label": predictions["aspect_only"][item["id"]],
            "masked_context_label": predictions["masked_context"][item["id"]],
            "context_plus_aspect_label": predictions["context_plus_aspect"][item["id"]],
        }
        for item in items
    ]
    public_stimuli = [
        {
            key: item[key]
            for key in (
                "id",
                "cluster_id",
                "dataset",
                "line_index",
                "aspect_token_indices",
                "opinion_token_indices",
                "polarity",
            )
        }
        for item in items
    ]
    output.mkdir(parents=True)
    (output / "stimuli.json").write_text(json.dumps(public_stimuli, indent=2) + "\n")
    (output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest = {
        "experiment": "039",
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__)),
        "parent_037_protocol_sha256": sha256(PROTOCOL_037),
        "parent_037_predictions_sha256": sha256(PARENT_037 / "predictions.json"),
        "parent_038_predictions_sha256": sha256(PARENT_038 / "predictions.json"),
        "source_revision": SOURCE_REVISION,
        "source_files_sha256": SOURCE_SHA256,
        "stimuli_sha256": sha256(output / "stimuli.json"),
        "predictions_sha256": sha256(output / "predictions.json"),
        "stimuli": len(items),
        "clusters": len({item["cluster_id"] for item in items}),
        "fold_sizes": fold_sizes,
        "variants": list(predictions),
        "tfidf": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 1,
            "norm": "l2",
        },
        "logistic_regression": {
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 2000,
            "random_state": SEED,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": version("scikit-learn"),
        },
        "raw_review_text_published": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
