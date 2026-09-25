"""Analyze the independent TripR corpus prefix comparison."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_prompt_format_score_generation import MODELS
from tripr_layer_confirmation import DATA, SOURCE_REPOSITORY_REVISION, load_tripr

PRIVATE = Path(".context/tripr-prefix-threshold-034")
OUT = Path("results/tripr-prefix-threshold-v1")
BUDGETS = (8, 12, 32)
JUDGES = ("laya", "qwen2.5-3b", "phi3-mini")
REPS = 10_000
SEED = 20260934
AMBIGUOUS = {"mixed", "unclear"}


def correct(label, gold):
    if label is None:
        return False
    prediction = label if label in (0, 1) else int(label == "positive")
    return prediction == gold


def paired_bootstrap(rows, difference, reps=REPS, seed=SEED):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sentence_id"]].append(row)
    clusters = np.asarray(sorted(grouped))
    if not len(clusters):
        return {"estimate": None, "sentence_cluster_bootstrap_95_ci": None, "n": 0}

    def estimate(sample):
        return float(np.mean([difference(row) for row in sample]))

    value = estimate(rows)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for index in range(reps):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in selected for row in grouped[cluster]]
        draws[index] = estimate(sample)
    return {
        "estimate": value,
        "sentence_cluster_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n": len(rows),
        "n_source_sentence_clusters": len(clusters),
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def load_and_validate(private=PRIVATE):
    manifest = json.loads((private / "manifest.json").read_text())
    predictions_path = private / "predictions.json"
    generation_path = private / "private-generations.json"
    if hashlib.sha256(predictions_path.read_bytes()).hexdigest() != manifest[
        "predictions_sha256"
    ]:
        raise ValueError("Prediction hash does not match the private manifest")
    if hashlib.sha256(generation_path.read_bytes()).hexdigest() != manifest[
        "private_generations_sha256"
    ]:
        raise ValueError("Generation hash does not match the private manifest")
    stimuli, _labels, conflicts = load_tripr(DATA)
    if (len(stimuli), len({row["sentence_id"] for row in stimuli}), conflicts) != (
        385,
        187,
        29,
    ):
        raise ValueError("Frozen TripR source filter changed")
    generations = json.loads(generation_path.read_text())
    predictions = json.loads(predictions_path.read_text())
    expected_generations = len(MODELS) * 385
    expected_predictions = len(MODELS) * 385 * len(BUDGETS) * len(JUDGES)
    if len(generations) != expected_generations:
        raise ValueError(f"Expected {expected_generations} TripR generations")
    if len(predictions) != expected_predictions:
        raise ValueError(f"Expected {expected_predictions} TripR prefix judgments")
    if manifest["dataset_repository_revision"] != SOURCE_REPOSITORY_REVISION:
        raise ValueError("TripR revision mismatch")
    return manifest, generations, predictions


def normalize_predictions(predictions):
    indexed = {}
    for row in predictions:
        key = (row["judge"], row["model"], row["id"], row["budget"])
        if key in indexed:
            raise ValueError(f"Duplicate TripR prefix outcome {key}")
        indexed[key] = row
    rows = []
    for judge in JUDGES:
        for model in MODELS:
            ids = sorted(
                key[2]
                for key in indexed
                if key[0] == judge and key[1] == model and key[3] == 8
            )
            if len(ids) != 385:
                raise ValueError(f"Incomplete {judge}/{model} 8-token rows")
            for item_id in ids:
                baseline = indexed[(judge, model, item_id, 8)]
                for budget in BUDGETS:
                    row = indexed[(judge, model, item_id, budget)]
                    if (row["sentence_id"], row["gold"], row["category"]) != (
                        baseline["sentence_id"],
                        baseline["gold"],
                        baseline["category"],
                    ):
                        raise ValueError(
                            "Mismatched paired metadata for "
                            f"{judge}/{model}/{item_id} at {budget} tokens"
                        )
                    rows.append(row)
    return rows


def series_metrics(rows):
    indexed = {(r["judge"], r["model"], r["id"], r["budget"]): r for r in rows}
    by_budget = {}
    for budget in BUDGETS:
        laya = [r for r in rows if r["judge"] == "laya" and r["budget"] == budget]
        qwen = [r for r in rows if r["judge"] == "qwen2.5-3b" and r["budget"] == budget]
        phi = [r for r in rows if r["judge"] == "phi3-mini" and r["budget"] == budget]
        joint = []
        for q in qwen:
            p = indexed[("phi3-mini", q["model"], q["id"], budget)]
            if q["label"] is not None and p["label"] is not None:
                joint.append(
                    {
                        "sentence_id": q["sentence_id"],
                        "agreement": int(q["label"] == p["label"]),
                    }
                )
        by_budget[str(budget)] = {
            "mean_actual_generated_tokens": float(np.mean([r["actual_tokens"] for r in laya])),
            "completed_before_budget_fraction": float(
                np.mean([r["actual_tokens"] < budget for r in laya])
            ),
            "laya_mixed_unclear": paired_bootstrap(
                laya,
                lambda r: int(r["label"] in AMBIGUOUS),
                seed=SEED + budget,
            ),
            "qwen_phi_agreement": paired_bootstrap(
                joint, lambda r: r["agreement"], seed=SEED + 100 + budget
            ),
            "qwen_coverage": sum(r["label"] is not None for r in qwen) / len(qwen),
            "qwen_accuracy_vs_tripr_polarity": paired_bootstrap(
                qwen,
                lambda r: int(correct(r["label"], r["gold"])),
                seed=SEED + 200 + budget,
            ),
            "phi_coverage": sum(r["label"] is not None for r in phi) / len(phi),
            "phi_accuracy_vs_tripr_polarity": paired_bootstrap(
                phi,
                lambda r: int(correct(r["label"], r["gold"])),
                seed=SEED + 300 + budget,
            ),
        }

    agreement_pairs = []
    laya_pairs = []
    accuracy_pairs = {judge: [] for judge in ("qwen2.5-3b", "phi3-mini")}
    for model in MODELS:
        ids = sorted(
            key[2]
            for key in indexed
            if key[1] == model and key[3] == 8 and key[0] == "qwen2.5-3b"
        )
        for item_id in ids:
            qw8 = indexed[("qwen2.5-3b", model, item_id, 8)]
            qw12 = indexed[("qwen2.5-3b", model, item_id, 12)]
            ph8 = indexed[("phi3-mini", model, item_id, 8)]
            ph12 = indexed[("phi3-mini", model, item_id, 12)]
            l8 = indexed[("laya", model, item_id, 8)]
            l12 = indexed[("laya", model, item_id, 12)]
            if all(r["label"] is not None for r in (qw8, qw12, ph8, ph12)):
                agreement_pairs.append(
                    {
                        "sentence_id": qw8["sentence_id"],
                        "delta": int(qw12["label"] == ph12["label"])
                        - int(qw8["label"] == ph8["label"]),
                    }
                )
            laya_pairs.append(
                {
                    "sentence_id": l8["sentence_id"],
                    "delta": int(l12["label"] in AMBIGUOUS) - int(l8["label"] in AMBIGUOUS),
                }
            )
            for judge, short, long in (
                ("qwen2.5-3b", qw8, qw12),
                ("phi3-mini", ph8, ph12),
            ):
                accuracy_pairs[judge].append(
                    {
                        "sentence_id": short["sentence_id"],
                        "delta": int(correct(long["label"], long["gold"]))
                        - int(correct(short["label"], short["gold"])),
                    }
                )
    paired_effects = {
        "primary_qwen_phi_agreement_12_minus_8": paired_bootstrap(
            agreement_pairs, lambda r: r["delta"], seed=SEED + 500
        ),
        "laya_mixed_unclear_rate_12_minus_8": paired_bootstrap(
            laya_pairs, lambda r: r["delta"], seed=SEED + 501
        ),
        "strict_accuracy_vs_source_label_12_minus_8": {
            judge: paired_bootstrap(
                rows_for_judge, lambda r: r["delta"], seed=SEED + 510 + index
            )
            for index, (judge, rows_for_judge) in enumerate(accuracy_pairs.items())
        },
    }

    by_target = {}
    for model in MODELS:
        target_rows = [r for r in rows if r["model"] == model]
        target_indexed = {
            (r["judge"], r["id"], r["budget"]): r for r in target_rows
        }
        target_effect = []
        for item_id in sorted(
            key[1] for key in target_indexed if key[0] == "qwen2.5-3b" and key[2] == 8
        ):
            qw8 = target_indexed[("qwen2.5-3b", item_id, 8)]
            qw12 = target_indexed[("qwen2.5-3b", item_id, 12)]
            ph8 = target_indexed[("phi3-mini", item_id, 8)]
            ph12 = target_indexed[("phi3-mini", item_id, 12)]
            if all(r["label"] is not None for r in (qw8, qw12, ph8, ph12)):
                target_effect.append(
                    {
                        "sentence_id": qw8["sentence_id"],
                        "delta": int(qw12["label"] == ph12["label"])
                        - int(qw8["label"] == ph8["label"]),
                    }
                )
        target_agreement_by_budget = {}
        for budget in BUDGETS:
            joint = []
            for q in target_rows:
                if q["judge"] != "qwen2.5-3b" or q["budget"] != budget:
                    continue
                p = target_indexed[("phi3-mini", q["id"], budget)]
                if q["label"] is not None and p["label"] is not None:
                    joint.append(
                        {
                            "sentence_id": q["sentence_id"],
                            "agreement": int(q["label"] == p["label"]),
                        }
                    )
            target_agreement_by_budget[str(budget)] = paired_bootstrap(
                joint,
                lambda r: r["agreement"],
                seed=SEED + 700 + MODELS.index(model) * 10 + budget,
            )
        by_target[model] = {
            "n_item_queries": len({r["id"] for r in target_rows if r["budget"] == 8}),
            "paired_agreement_change_12_minus_8": paired_bootstrap(
                target_effect, lambda r: r["delta"], seed=SEED + 600 + MODELS.index(model)
            ),
            "agreement_by_budget": target_agreement_by_budget,
        }

    # Post-hoc diagnostic: the frozen source corpus is polarity-imbalanced.
    # Separate class-specific paired accuracy shifts to check whether pooled
    # gains conceal asymmetric difficulty on negative examples.
    class_diagnostics = {}
    for judge in ("qwen2.5-3b", "phi3-mini"):
        per_class = {}
        for gold in (0, 1):
            paired_rows = []
            by_budget_class = {}
            for budget in BUDGETS:
                budget_rows = [
                    r
                    for r in rows
                    if r["judge"] == judge and r["budget"] == budget and r["gold"] == gold
                ]
                by_budget_class[str(budget)] = paired_bootstrap(
                    budget_rows,
                    lambda r: int(correct(r["label"], gold)),
                    seed=SEED + 800 + (0 if judge == "qwen2.5-3b" else 100) + gold * 10 + budget,
                )
            for model in MODELS:
                item_ids = sorted(
                    r["id"]
                    for r in rows
                    if r["judge"] == judge
                    and r["model"] == model
                    and r["budget"] == 8
                    and r["gold"] == gold
                )
                for item_id in item_ids:
                    short = indexed[(judge, model, item_id, 8)]
                    long = indexed[(judge, model, item_id, 12)]
                    paired_rows.append(
                        {
                            "sentence_id": short["sentence_id"],
                            "delta": int(correct(long["label"], gold))
                            - int(correct(short["label"], gold)),
                        }
                    )
            per_class["positive" if gold else "negative"] = {
                "accuracy_by_budget": by_budget_class,
                "paired_accuracy_change_12_minus_8": paired_bootstrap(
                    paired_rows,
                    lambda r: r["delta"],
                    seed=SEED + 900 + (0 if judge == "qwen2.5-3b" else 100) + gold,
                ),
            }
        class_diagnostics[judge] = per_class

    return {
        "by_budget": by_budget,
        "paired_8_to_12_effects": paired_effects,
        "by_target": by_target,
        "exploratory_gold_stratified_accuracy": class_diagnostics,
    }


def run(private=PRIVATE, output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest, generations, predictions = load_and_validate(private)
    rows = normalize_predictions(predictions)
    if len(rows) != len(MODELS) * 385 * len(BUDGETS) * len(JUDGES):
        raise ValueError("Normalized factorial size changed")
    analysis = {
        "experiment": "034",
        "status": "independent-corpus replication",
        "n_item_target_judge_prefix_outcomes": len(rows),
        "primary_endpoint": "paired change in Qwen/Phi agreement from 8 to 12 tokens, jointly parseable at both lengths",
        "analysis": series_metrics(rows),
        "n_source_sentence_clusters": len({r["sentence_id"] for r in rows}),
        "limitations": [
            "TripR is a different manually annotated restaurant-review corpus, not an unrelated domain.",
            "TripR sentiment is source-text gold and only a proxy for the meaning of the generated answer.",
            "All three evaluators are automated; their agreement is not human validation of generated-answer semantics.",
            "Raw review and generated-answer text remain private in ignored .context/.",
            "The category filter follows experiment 018 and inherits the TripR annotation's reported 0.66 inter-annotator agreement.",
        ],
    }
    output.mkdir(parents=True)
    public_predictions = [
        {key: value for key, value in row.items() if key not in {"answer", "text", "review", "user", "token_ids"}}
        for row in rows
    ]
    prediction_path = output / "predictions.json"
    prediction_path.write_text(json.dumps(public_predictions, indent=2, sort_keys=True) + "\n")
    public_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"private_generations_sha256", "predictions_sha256"}
    }
    public_manifest["predictions_sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(public_manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    audit = {
        "checks": {
            "complete_10395_prefix_judge_outcomes": len(public_predictions)
            == len(MODELS) * 385 * len(BUDGETS) * len(JUDGES),
            "all_target_models_cover_all_385_items": all(
                len({r["id"] for r in public_predictions if r["model"] == model and r["budget"] == 8})
                == 385
                for model in MODELS
            ),
            "no_raw_text_or_token_ids_in_public_predictions": all(
                not ({"answer", "text", "review", "user", "token_ids"} & row.keys())
                for row in public_predictions
            ),
            "protocol_hash_matches": hashlib.sha256(
                Path("docs/experiments/034-tripr-prefix-threshold.md").read_bytes()
            ).hexdigest()
            == manifest["protocol_sha256"],
        }
    }
    audit["checks_passed"] = all(audit["checks"].values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["checks_passed"]:
        raise ValueError("TripR prefix replication audit failed")
    readme = """# Experiment 034 — TripR prefix threshold\n\nThis independent-corpus follow-up applies the three frozen judges to nested 8/12/32-token prefixes of open answers generated from the TripR-2020Large annotations. Raw review/answer text and token IDs remain only in ignored `.context/`. See [`analysis.json`](analysis.json), [`predictions.json`](predictions.json), and the frozen [protocol](../../docs/experiments/034-tripr-prefix-threshold.md). TripR labels remain proxies for generated-answer meaning.\n"""
    (output / "README.md").write_text(readme)
    print(json.dumps(analysis, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.private, args.output)


if __name__ == "__main__":
    main()
