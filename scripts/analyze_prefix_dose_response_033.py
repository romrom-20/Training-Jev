"""Analyze evaluator agreement and stability on nested continuation prefixes."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRIVATE_031 = Path(".context/length-censoring-031")
PRIVATE_032 = Path(".context/answer-length-crossjudge-032")
PRIVATE_033 = Path(".context/prefix-dose-response-033")
OUT = Path("results/prefix-dose-response-v1")
BUDGETS = (4, 8, 12, 16, 24, 32)
MODELS = ("granite-3.1-2b", "qwen-1.5b", "smollm2-1.7b")
JUDGES = ("laya", "qwen2.5-3b", "phi3-mini")
REPS = 10_000
SEED = 20260933
AMBIGUOUS = {"mixed", "unclear"}


def load_all_predictions(private=PRIVATE_033):
    new_manifest = json.loads((private / "manifest.json").read_text())
    new_predictions_path = private / "predictions.json"
    if hashlib.sha256(new_predictions_path.read_bytes()).hexdigest() != new_manifest[
        "predictions_sha256"
    ]:
        raise ValueError("Experiment 033 private predictions hash mismatch")
    rows = json.loads(new_predictions_path.read_text())
    generations = json.loads((PRIVATE_031 / "private-generations.json").read_text())
    lengths = {
        (row["model"], row["id"], row["budget"]): row["token_count"]
        for row in generations
    }
    for row in json.loads((PRIVATE_031 / "predictions.json").read_text()):
        normalized = dict(row)
        normalized.update(
            judge="laya",
            label=row["laya_label"],
            actual_tokens=lengths[(row["model"], row["id"], row["budget"])],
        )
        rows.append(normalized)
    for row in json.loads((PRIVATE_032 / "predictions.json").read_text()):
        normalized = dict(row)
        normalized.update(
            label=row["judge_label"],
            actual_tokens=lengths[(row["model"], row["id"], row["budget"])],
        )
        rows.append(normalized)
    indexed = {(row["judge"], row["model"], row["id"], row["budget"]): row for row in rows}
    expected = len(JUDGES) * len(MODELS) * 233 * len(BUDGETS)
    if len(indexed) != expected:
        raise ValueError(f"Expected {expected} unique prefix/judge rows, got {len(indexed)}")
    return list(indexed.values()), new_manifest


def is_correct(label, gold):
    if label is None:
        return False
    prediction = label if label in (0, 1) else int(label == "positive")
    return prediction == gold


def cluster_interval(rows, value, condition=lambda row: True, reps=REPS, seed=SEED):
    eligible = [row for row in rows if condition(row)]
    grouped = defaultdict(list)
    for row in eligible:
        grouped[row["sentence_id"]].append(row)
    clusters = np.asarray(sorted(grouped))
    if not len(clusters):
        return {"estimate": None, "sentence_cluster_bootstrap_95_ci": None, "n": 0}

    def estimate(sample):
        return float(np.mean([value(row) for row in sample]))

    point = estimate(eligible)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for index in range(reps):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in selected for row in grouped[cluster]]
        draws[index] = estimate(sample)
    return {
        "estimate": point,
        "sentence_cluster_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n": len(eligible),
        "n_source_sentence_clusters": len(clusters),
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def build_series(rows):
    by_key = {(row["judge"], row["model"], row["id"], row["budget"]): row for row in rows}
    series = []
    for judge in JUDGES:
        for model in MODELS:
            ids = sorted(
                key[2]
                for key in by_key
                if key[0] == judge and key[1] == model and key[3] == 32
            )
            for item_id in ids:
                base = by_key[(judge, model, item_id, 32)]
                for budget in BUDGETS:
                    row = by_key[(judge, model, item_id, budget)]
                    if row["sentence_id"] != base["sentence_id"] or row["gold"] != base["gold"]:
                        raise ValueError(f"Mismatched nested-prefix metadata for {judge}/{model}/{item_id}")
                    series.append(row)
    return series


def verify_reused_labels(rows):
    values = {(row["judge"], row["model"], row["id"], row["budget"]): row["label"] for row in rows}
    expected_sources = [
        ("laya", PRIVATE_031 / "predictions.json", {8, 32}),
        ("qwen2.5-3b", PRIVATE_032 / "predictions.json", {8, 32}),
        ("phi3-mini", PRIVATE_032 / "predictions.json", {8, 32}),
    ]
    comparisons = {}
    for judge, path, budgets in expected_sources:
        source = json.loads(path.read_text())
        if judge == "laya":
            source = [row for row in source if row["budget"] in budgets]
            label_field = "laya_label"
        else:
            source = [row for row in source if row["judge"] == judge and row["budget"] in budgets]
            label_field = "judge_label"
        matched = [
            row
            for row in source
            if values[(judge, row["model"], row["id"], row["budget"])] == row[label_field]
        ]
        comparisons[judge] = {
            "n_reused_rows": len(source),
            "exact_label_match_fraction": len(matched) / len(source),
            "n_label_mismatches": len(source) - len(matched),
        }
    return comparisons


def summarize(rows):
    by_key = {(row["judge"], row["model"], row["id"], row["budget"]): row for row in rows}
    by_budget = {}
    for budget in BUDGETS:
        laya_rows = [row for row in rows if row["judge"] == "laya" and row["budget"] == budget]
        qwen_rows = [row for row in rows if row["judge"] == "qwen2.5-3b" and row["budget"] == budget]
        phi_rows = [row for row in rows if row["judge"] == "phi3-mini" and row["budget"] == budget]
        both = []
        for qwen in qwen_rows:
            phi = by_key[("phi3-mini", qwen["model"], qwen["id"], budget)]
            if qwen["label"] is not None and phi["label"] is not None:
                both.append(
                    {
                        "sentence_id": qwen["sentence_id"],
                        "agree": int(qwen["label"] == phi["label"]),
                    }
                )
        by_budget[str(budget)] = {
            "actual_tokens_in_prefix": {
                "mean": float(np.mean([row["actual_tokens"] for row in laya_rows])),
                "max": max(row["actual_tokens"] for row in laya_rows),
                "reused_eos_completed_text_fraction": float(
                    np.mean([row["actual_tokens"] < budget for row in laya_rows])
                ),
            },
            "laya": {
                "n": len(laya_rows),
                "four_way_counts": dict(Counter(row["label"] for row in laya_rows)),
                "mixed_unclear_rate": cluster_interval(
                    laya_rows,
                    lambda row: int(row["label"] in AMBIGUOUS),
                    seed=SEED + budget,
                ),
            },
            "qwen_phi_agreement": {
                "n_both_parseable": len(both),
                "agreement": cluster_interval(
                    both, lambda row: row["agree"], seed=SEED + 100 + budget
                ),
            },
            "qwen2_5_3b": {
                "coverage": sum(row["label"] is not None for row in qwen_rows) / len(qwen_rows),
                "strict_accuracy_vs_review_proxy": cluster_interval(
                    qwen_rows,
                    lambda row: int(is_correct(row["label"], row["gold"])),
                    seed=SEED + 200 + budget,
                ),
            },
            "phi3_mini": {
                "coverage": sum(row["label"] is not None for row in phi_rows) / len(phi_rows),
                "strict_accuracy_vs_review_proxy": cluster_interval(
                    phi_rows,
                    lambda row: int(is_correct(row["label"], row["gold"])),
                    seed=SEED + 300 + budget,
                ),
            },
        }

    stabilization = {}
    for judge in ("qwen2.5-3b", "phi3-mini"):
        rows_by_model = {}
        labels_for = defaultdict(dict)
        for row in rows:
            if row["judge"] == judge:
                labels_for[(row["model"], row["id"])][row["budget"]] = row["label"]
        for model in MODELS:
            observations = []
            for (target, item_id), labels in labels_for.items():
                if target != model or labels[32] is None:
                    continue
                stable_from = 32
                for index, budget in enumerate(BUDGETS):
                    if all(labels[later] == labels[32] for later in BUDGETS[index:]):
                        stable_from = budget
                        break
                observations.append(stable_from)
            rows_by_model[model] = {
                "n_parseable_32_token_endpoints": len(observations),
                "first_budget_stable_to_32_distribution": {
                    str(budget): observations.count(budget) for budget in BUDGETS
                },
                "stable_by_budget_fraction": {
                    str(budget): sum(value <= budget for value in observations) / len(observations)
                    if observations
                    else None
                    for budget in BUDGETS
                },
            }
        stabilization[judge] = rows_by_model
    return {"by_budget": by_budget, "judge_label_stability": stabilization}


def run(private=PRIVATE_033, output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows, manifest = load_all_predictions(private)
    series = build_series(rows)
    analysis = {
        "experiment": "033",
        "status": "post-result dose-response follow-up",
        "n_item_target_judge_prefix_outcomes": len(series),
        "primary_endpoint": "Qwen/Phi label agreement among jointly parseable outputs over nested prefix budgets",
        "analysis": summarize(series),
        "reused_label_checks": verify_reused_labels(series),
        "limitations": [
            "This reuses the exact 233 items and 32-token continuations from 031/032; it is not an independent sample replication.",
            "Agreement and stability do not establish semantic correctness; review labels are proxies and judges are automated.",
            "Prefix length adds semantic evidence together with token count and does not isolate generic verbosity bias.",
            "Later budgets repeat a naturally ended answer when EOS occurred before that prefix boundary.",
        ],
    }
    output.mkdir(parents=True)
    prediction_path = output / "predictions.json"
    prediction_path.write_text(json.dumps(series, indent=2, sort_keys=True) + "\n")
    public_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"source_generation_sha256", "source_031_manifest_sha256", "source_032_manifest_sha256"}
    }
    public_manifest["predictions_sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(public_manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    audit = {
        "checks": {
            "complete_4194_nested_prefix_decisions": len(series) == 4194,
            "six_unique_prefixes_per_judge_target_item": len(
                {(r["judge"], r["model"], r["id"], r["budget"]) for r in series}
            )
            == 4194,
            "all_reused_labels_match_source_runs": all(
                row["n_label_mismatches"] == 0
                for row in analysis["reused_label_checks"].values()
            ),
            "no_answer_text_or_token_ids_in_public_rows": all(
                not ({"answer", "text", "review", "user", "token_ids"} & row.keys())
                for row in series
            ),
            "protocol_hash_matches": hashlib.sha256(
                Path("docs/experiments/033-prefix-dose-response.md").read_bytes()
            ).hexdigest()
            == manifest["protocol_sha256"],
        }
    }
    audit["checks_passed"] = all(audit["checks"].values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["checks_passed"]:
        raise ValueError("Experiment 033 audit failed")
    readme = """# Experiment 033 — prefix dose response\n\nThis follow-up evaluates nested token prefixes of the exact 32-token continuations from experiments 031/032. Raw review/answer text and token IDs remain private in ignored `.context/`. See [`analysis.json`](analysis.json), [`predictions.json`](predictions.json), and the frozen [protocol](../../docs/experiments/033-prefix-dose-response.md). Agreement is not human validation.\n"""
    (output / "README.md").write_text(readme)
    print(json.dumps(analysis, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, default=PRIVATE_033)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.private, args.output)


if __name__ == "__main__":
    main()
