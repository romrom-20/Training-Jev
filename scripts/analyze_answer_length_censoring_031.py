"""Analyze paired truncation labels; raw text remains in ignored .context."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRIVATE = Path(".context/length-censoring-031")
OUT = Path("results/answer-length-censoring-v1")
PRIOR = Path(".context/laya-decision-audit-030/predictions.json")
MODELS = ("granite-3.1-2b", "qwen-1.5b", "smollm2-1.7b")
REPS = 10_000
SEED = 20260931
AMBIGUOUS = {"mixed", "unclear"}


def bootstrap_difference(rows, reps=REPS, seed=SEED):
    eligible = [row for row in rows if row["capped_at_8"]]
    grouped = defaultdict(list)
    for row in eligible:
        grouped[row["sentence_id"]].append(row)
    clusters = np.asarray(sorted(grouped))
    if not len(clusters):
        return {"n_pairs": 0, "difference": None, "sentence_cluster_bootstrap_95_ci": None}

    def difference(sample):
        return float(np.mean([row["ambiguous_32"] - row["ambiguous_8"] for row in sample]))

    point = difference(eligible)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for index in range(reps):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in chosen for row in grouped[cluster]]
        draws[index] = difference(sample)
    return {
        "n_pairs": len(eligible),
        "n_source_sentence_clusters": len(clusters),
        "difference_ambiguous_32_minus_8": point,
        "sentence_cluster_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def make_pairs(predictions, generations):
    generated = {
        (row["model"], row["id"], row["budget"]): row
        for row in generations
    }
    predicted = {
        (row["model"], row["id"], row["budget"]): row
        for row in predictions
    }
    pairs = []
    for model in MODELS:
        ids = sorted(
            key[1] for key in generated if key[0] == model and key[2] == 8
        )
        for item_id in ids:
            short = generated[(model, item_id, 8)]
            long = generated[(model, item_id, 32)]
            label8 = predicted[(model, item_id, 8)]["laya_label"]
            label32 = predicted[(model, item_id, 32)]["laya_label"]
            validate_generation_pair(short, long, model, item_id)
            pairs.append(
                {
                    "model": model,
                    "id": item_id,
                    "sentence_id": short["sentence_id"],
                    "category": short["category"],
                    "gold": short["gold"],
                    "capped_at_8": short["capped"],
                    "tokens_at_8": short["token_count"],
                    "tokens_at_32": long["token_count"],
                    "capped_at_32": long["capped"],
                    "label_8": label8,
                    "label_32": label32,
                    "ambiguous_8": int(label8 in AMBIGUOUS),
                    "ambiguous_32": int(label32 in AMBIGUOUS),
                    "label_changed": label8 != label32,
                }
            )
    if len(pairs) != len(MODELS) * 233:
        raise ValueError(f"Expected 699 pairs, received {len(pairs)}")
    return pairs


def validate_generation_pair(short, long, model="target", item_id="item"):
    if short["capped"]:
        if len(short["token_ids"]) != 8 or long["token_ids"][:8] != short["token_ids"]:
            raise ValueError(f"Non-prefix cap-hit pair: {model}/{item_id}")
    elif short["token_ids"] != long["token_ids"]:
        raise ValueError(f"Naturally ended answer changed with higher cap: {model}/{item_id}")


def summarize(rows):
    by_model = {}
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        capped = [row for row in model_rows if row["capped_at_8"]]
        uncapped = [row for row in model_rows if not row["capped_at_8"]]
        by_model[model] = {
            "n_items": len(model_rows),
            "capped_at_8": len(capped),
            "cap_hit_rate_at_8": len(capped) / len(model_rows),
            "capped_at_32": sum(row["capped_at_32"] for row in model_rows),
            "labels_all_pairs_8": dict(Counter(row["label_8"] for row in model_rows)),
            "labels_all_pairs_32": dict(Counter(row["label_32"] for row in model_rows)),
            "primary_cap_hit_effect": bootstrap_difference(capped),
            "ambiguous_rate_on_cap_hit_pairs_8": (
                float(np.mean([row["ambiguous_8"] for row in capped])) if capped else None
            ),
            "ambiguous_rate_on_cap_hit_pairs_32": (
                float(np.mean([row["ambiguous_32"] for row in capped])) if capped else None
            ),
            "label_change_rate_among_cap_hit_pairs": (
                float(np.mean([row["label_changed"] for row in capped])) if capped else None
            ),
            "naturally_ended_at_8_negative_control": {
                "n": len(uncapped),
                "label_change_count": sum(row["label_changed"] for row in uncapped),
                "four_way_labels_8": dict(Counter(row["label_8"] for row in uncapped)),
                "four_way_labels_32": dict(Counter(row["label_32"] for row in uncapped)),
            },
            "four_way_transition_counts": {
                label: {
                    other: sum(row["label_8"] == label and row["label_32"] == other for row in model_rows)
                    for other in ("positive", "negative", "mixed", "unclear")
                }
                for label in ("positive", "negative", "mixed", "unclear")
            },
            "secondary_strict_binary_accuracy_vs_review_gold": {
                "budget_8": sum(
                    row["label_8"] in {"positive", "negative"}
                    and int(row["label_8"] == "positive") == row["gold"]
                    for row in model_rows
                )
                / len(model_rows),
                "budget_32": sum(
                    row["label_32"] in {"positive", "negative"}
                    and int(row["label_32"] == "positive") == row["gold"]
                    for row in model_rows
                )
                / len(model_rows),
            },
        }
    pooled = bootstrap_difference(rows)
    pooled["ambiguous_rate_on_cap_hit_pairs_8"] = float(
        np.mean([row["ambiguous_8"] for row in rows if row["capped_at_8"]])
    )
    pooled["ambiguous_rate_on_cap_hit_pairs_32"] = float(
        np.mean([row["ambiguous_32"] for row in rows if row["capped_at_8"]])
    )
    pooled["label_change_rate_among_cap_hit_pairs"] = float(
        np.mean([row["label_changed"] for row in rows if row["capped_at_8"]])
    )
    return {"pooled_across_targets": pooled, "by_target": by_model}


def reproducibility_check(predictions):
    prior = json.loads(PRIOR.read_text())
    old = {
        (row["model"], row["id"]): row["laya_label"]
        for row in prior
        if row["phase"] == "generated"
    }
    fresh = {
        (row["model"], row["id"]): row["laya_label"]
        for row in predictions
        if row["budget"] == 8
    }
    matching = [key for key in old if key in fresh]
    return {
        "n_matching_experiment_030_rows": len(matching),
        "exact_label_agreement": sum(old[key] == fresh[key] for key in matching) / len(matching),
        "n_label_disagreements": sum(old[key] != fresh[key] for key in matching),
    }


def run(private=PRIVATE, output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((private / "manifest.json").read_text())
    predictions = json.loads((private / "predictions.json").read_text())
    generations = json.loads((private / "private-generations.json").read_text())
    if hashlib.sha256((private / "private-generations.json").read_bytes()).hexdigest() != manifest[
        "private_generation_sha256"
    ]:
        raise ValueError("Private generation hash does not match run manifest")
    rows = make_pairs(predictions, generations)
    analysis = {
        "experiment": "031",
        "n_paired_item_target_outcomes": len(rows),
        "primary_endpoint": "P(Laya label in {mixed, unclear} at 32) - P(same at 8), among pairs whose 8-token generation hit cap",
        "analysis": summarize(rows),
        "reproducibility_against_030": reproducibility_check(predictions),
        "limitations": [
            "Laya is an automated evaluator, not human answer-level ground truth.",
            "The causal treatment is a decoding cap; any effect is specific to these checkpoints, prompts, and task.",
            "The benchmark has 233 items from a limited restaurant-review domain, with repeated sentence clusters handled in the bootstrap.",
            "Review-level gold is only a proxy for the meaning of generated answers.",
            "Length and additional generated content change together, so the result does not isolate verbosity preference from semantic completion.",
        ],
    }
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n")
    public_manifest = {key: value for key, value in manifest.items() if key != "private_generation_sha256"}
    public_manifest["predictions_sha256"] = hashlib.sha256(
        (output / "predictions.json").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n")
    audit = {
        "checks": {
            "complete_699_pairs": len(rows) == 699,
            "pairwise_prefix_and_eos_checks": True,
            "no_raw_text_or_token_ids_in_public_predictions": all(
                not ({"text", "answer", "review", "user", "token_ids"} & row.keys())
                for row in predictions
            ),
            "protocol_hash_matches": hashlib.sha256(
                Path("docs/experiments/031-answer-length-censoring.md").read_bytes()
            ).hexdigest()
            == manifest["protocol_sha256"],
        }
    }
    audit["checks_passed"] = all(audit["checks"].values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["checks_passed"]:
        raise ValueError("Audit failed")
    readme = """# Experiment 031 — answer-length censoring\n\nThis bundle reports a paired local 8-vs-32 generated-token-cap intervention. Raw answer text and token IDs remain private in the ignored `.context/length-censoring-031/` directory. The primary outcome is the difference in Laya's mixed/unclear rate among answers that hit the 8-token cap. See [`analysis.json`](analysis.json), [`predictions.json`](predictions.json), and the frozen [protocol](../../docs/experiments/031-answer-length-censoring.md).\n\nLaya labels are automated judgments, not answer-level human truth. The result is specific to this task and evaluator.\n"""
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
