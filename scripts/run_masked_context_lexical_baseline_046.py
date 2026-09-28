"""Train per-language sparse VA baselines on official DimABSA splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from run_expanded_opinion_mask_repair_043 import (
    MANIFEST as MANIFEST_043,
)
from run_expanded_opinion_mask_repair_043 import (
    OUT as OUT_043,
)
from run_expanded_opinion_mask_repair_043 import (
    PROTOCOL as PROTOCOL_043,
)
from run_expanded_opinion_mask_repair_043 import (
    select_repair_cases,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    FILES as TEST_FILES,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    LANGS,
    read_source,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    SOURCE_SHA256 as TEST_SOURCE_SHA256,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, make_pipeline

PROTOCOL = Path("docs/experiments/046-masked-context-lexical-baseline.md")
PROTOCOL_SHA256 = "afee6a9085a1693b411af1973dc30b4599dd467277b36786c13d40b324c77bbb"
SOURCE_REVISION = "bdc93be1224106ae7d3eb95739c02a76ed4ae8a1"
TRAIN_FILES = {
    "rus": "rus_restaurant_train_alltasks.jsonl",
    "ukr": "ukr_restaurant_train_alltasks.jsonl",
    "tat": "tat_restaurant_train_alltasks.jsonl",
}
TRAIN_SOURCE_SHA256 = {
    "rus": "a81c2e737a0ed8e3fcfee48864403a3339ab388b7ed545436f45a11ae878b92a",
    "ukr": "6c1e4843fb9af233a1103c751b56b99596f724ef9309ff6156f4041e4560bff1",
    "tat": "1f374e08dfd3fac61d9ca6935e185958f889a24a6690dacec6ff6497d4d54917",
}
PARENT_043_SHA256 = "1e93d359220c533a01f9cccca827f30611a713b70d857a898f59b6560303c7c8"
PARENT_043_PROTOCOL_SHA256 = "526a0de535a36deda9a2b0a4fb412967fad9720f5a39f492959ab3f71e0133a8"
SEED = 20260946
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FOLDS = 5
PRIVATE_DIR = Path(".context/exp046-private")
PREDICTIONS = PRIVATE_DIR / "predictions.jsonl"
MANIFEST = PRIVATE_DIR / "manifest.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mask_annotated_opinions(text: str, annotations: list[dict]) -> str:
    spans = set()
    for annotation in annotations:
        phrase = annotation.get("Opinion")
        if phrase in (None, "NULL", ""):
            continue
        start = 0
        while True:
            start = text.find(phrase, start)
            if start < 0:
                break
            spans.add((start, start + len(phrase)))
            start += max(1, len(phrase))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    result = text
    for start, end in reversed(merged):
        result = result[:start] + " [MASKED] " + result[end:]
    return " ".join(result.split())


def group_id(instance_id: str) -> str:
    return instance_id.split(":", 1)[0]


def build_pipeline(alpha: float):
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
        ],
        transformer_weights={"word": 1.0, "char": 1.0},
    )
    return make_pipeline(features, Ridge(alpha=alpha))


def choose_alpha(texts: list[str], y: np.ndarray, groups: list[str]) -> tuple[float, dict]:
    n_splits = min(FOLDS, len(set(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two distinct review groups for alpha selection")
    splitter = GroupKFold(n_splits=n_splits)
    scores = {}
    for alpha in ALPHAS:
        squared = []
        for train_index, valid_index in splitter.split(texts, y, groups):
            model = build_pipeline(alpha)
            model.fit([texts[i] for i in train_index], y[train_index])
            predicted = np.asarray(model.predict([texts[i] for i in valid_index]), dtype=float)
            squared.extend(((predicted - y[valid_index]) ** 2).reshape(-1).tolist())
        scores[str(alpha)] = float(np.sqrt(np.mean(squared)))
    chosen = min(ALPHAS, key=lambda alpha: (scores[str(alpha)], alpha))
    return chosen, scores


def train_examples(rows: list[dict]) -> tuple[dict[str, list[dict]], int]:
    examples = {lang: [] for lang in LANGS}
    for lang in LANGS:
        for row in rows[lang]:
            annotations = row.get("Quadruplet", row.get("Triplet", []))
            masked = mask_annotated_opinions(row["Text"], annotations)
            for item in annotations:
                va = item.get("VA")
                aspect = item.get("Aspect")
                if not aspect or aspect == "NULL" or not va or va == "NULL":
                    continue
                valence, arousal = (float(part) for part in va.split("#"))
                examples[lang].append(
                    {
                        "id": row["ID"],
                        "group": group_id(row["ID"]),
                        "aspect": aspect,
                        "masked_text": masked,
                        "gold": [valence, arousal],
                    }
                )
    return examples, sum(len(rows[lang]) for lang in LANGS)


def load_data(source_dir: Path) -> tuple[dict, dict, dict]:
    train_rows, train_hashes = {}, {}
    test_rows, test_hashes = {}, {}
    for lang in LANGS:
        train_path = source_dir / TRAIN_FILES[lang]
        train_bytes = train_path.read_bytes()
        train_hashes[lang] = sha256(train_bytes)
        if train_hashes[lang] != TRAIN_SOURCE_SHA256[lang]:
            raise ValueError(f"Official {lang} train hash mismatch")
        train_rows[lang] = read_jsonl(train_path)
        test_rows[lang], test_hashes[lang] = read_source(
            source_dir / Path(TEST_FILES[lang]).name, lang
        )
    if test_hashes != TEST_SOURCE_SHA256:
        raise ValueError("Pinned DimABSA test files changed")
    return train_rows, test_rows, {"train": train_hashes, "test": test_hashes}


def selected_targets(test_rows: dict, parent_hash: str) -> list[dict]:
    if sha256(PROTOCOL_043.read_bytes()) != PARENT_043_PROTOCOL_SHA256:
        raise ValueError("Experiment 043 parent protocol changed")
    if sha256(OUT_043.read_bytes()) != PARENT_043_SHA256:
        raise ValueError("Experiment 043 parent output changed")
    if parent_hash != PARENT_043_SHA256:
        raise ValueError("Experiment 043 parent hash is not the frozen value")
    parent_manifest = json.loads(MANIFEST_043.read_text())
    if parent_manifest.get("output_sha256") != parent_hash:
        raise ValueError("Experiment 043 manifest/output mismatch")
    selected = select_repair_cases(test_rows)
    output = []
    for case in selected:
        for lang in LANGS:
            row = case["rows"][lang]
            item = row["Triplet"][case["target_index"]]
            output.append(
                {
                    "case_id": case["case_id"],
                    "lang": lang,
                    "aspect": item["Aspect"],
                    "masked_text": mask_annotated_opinions(row["Text"], row["Triplet"]),
                    "gold": [float(value) for value in item["VA"].split("#")],
                    "aspect_only_text": f"Aspect: {item['Aspect']}",
                }
            )
    if len(selected) != 217 or len(output) != 651:
        raise ValueError("Experiment 043's frozen target sample changed")
    expected = {
        (json.loads(line)["case_id"], json.loads(line)["lang"]): json.loads(line)["gold"]
        for line in OUT_043.read_text().splitlines()
        if json.loads(line)["condition"] == "aspect_only"
    }
    for row in output:
        if expected.get((row["case_id"], row["lang"])) != row["gold"]:
            raise ValueError("Selected test gold VA differs from Experiment 043")
    return output


def run(source_dir: Path = Path(".context/dimabsa")) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 046 protocol hash mismatch")
    started = time.monotonic()
    train_rows, test_rows, source_hashes = load_data(source_dir)
    parent_hash = sha256(OUT_043.read_bytes())
    targets = selected_targets(test_rows, parent_hash)
    examples, _ = train_examples(train_rows)
    parent_ids = {row["case_id"] for row in targets}
    predictions = []
    tuning = {}
    counts = {}
    for lang in LANGS:
        train = examples[lang]
        train_ids = {row["id"] for row in train_rows[lang]}
        train_texts = {row["Text"] for row in train_rows[lang]}
        test_source = {row["ID"]: row for row in test_rows[lang]}
        selected_for_lang = [row for row in targets if row["lang"] == lang]
        test_texts = {test_source[row["case_id"]]["Text"] for row in selected_for_lang}
        if train_ids & parent_ids or train_texts & test_texts:
            raise ValueError(f"Train/test overlap detected for {lang}")
        counts[lang] = {
            "train_sentence_rows": len(train_rows[lang]),
            "train_aspect_examples": len(train),
            "test_clusters": len(selected_for_lang),
            "id_overlap": 0,
            "exact_text_overlap": 0,
        }
        for condition in ("aspect_only", "masked_context"):
            train_text = [
                f"Aspect: {row['aspect']}" if condition == "aspect_only"
                else f"Aspect: {row['aspect']} Context: {row['masked_text']}"
                for row in train
            ]
            y = np.asarray([row["gold"] for row in train], dtype=float)
            groups = [row["group"] for row in train]
            alpha, cv_scores = choose_alpha(train_text, y, groups)
            tuning[f"{lang}_{condition}"] = {
                "alpha": alpha,
                "grouped_cv_rmse_by_alpha": cv_scores,
                "n_training_examples": len(train),
                "n_training_review_groups": len(set(groups)),
            }
            model = build_pipeline(alpha)
            model.fit(train_text, y)
            target_texts = [
                row["aspect_only_text"] if condition == "aspect_only"
                else f"Aspect: {row['aspect']} Context: {row['masked_text']}"
                for row in selected_for_lang
            ]
            scores = np.asarray(model.predict(target_texts), dtype=float)
            for target, prediction in zip(selected_for_lang, scores):
                predictions.append(
                    {
                        "case_id": target["case_id"],
                        "lang": lang,
                        "condition": condition,
                        "gold": target["gold"],
                        "prediction": [float(x) for x in prediction],
                    }
                )
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    manifest = {
        "experiment": "046-masked-context-lexical-baseline",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "parent_043_output_sha256": parent_hash,
        "source_revision": SOURCE_REVISION,
        "source_hashes": source_hashes,
        "training_counts": counts,
        "alpha_grid": list(ALPHAS),
        "cv_folds_max": FOLDS,
        "feature_map": "word unigram/bigram TF-IDF + char_wb 2-5gram TF-IDF; multi-output ridge",
        "alpha_selection": tuning,
        "n_clusters": len(parent_ids),
        "n_predictions": len(predictions),
        "runtime_seconds": time.monotonic() - started,
        "predictions_sha256": sha256(PREDICTIONS.read_bytes()),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(".context/dimabsa"))
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
