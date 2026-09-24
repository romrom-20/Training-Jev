"""Audit and package the Granite generated-answer dose-response experiment."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from analyze_granite_tripr_generation_dose_response import summarize
from granite_tripr_generation_dose_response import CONTROL_SEEDS, DOSES, MODEL, PROTOCOL, ROOT_021
from prompt_effect_forecast import sha
from tripr_layer_confirmation import DATA as TRIPR_DATA
from tripr_layer_confirmation import load_tripr

ROOT = Path("runs/granite-tripr-generation-dose-response-v1")
RUNNER = Path("scripts/granite_tripr_generation_dose_response.py")
ANALYZER = Path("scripts/analyze_granite_tripr_generation_dose_response.py")


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit(analysis):
    folder = ROOT / "evaluation"
    outcomes = json.loads((folder / "outcomes.json").read_text())
    baseline = json.loads((folder / "baseline.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    stimuli, labels, _ = load_tripr()
    conflicts = {sid for sid, aspects in labels.items() if len(set(aspects.values())) > 1}
    expected_ids = {row["id"] for row in stimuli if row["sentence_id"] in conflicts}
    expected_count = len(DOSES) * len(expected_ids) * (len(CONTROL_SEEDS) + 1)
    baseline_021 = ROOT_021 / "evaluation" / "generation.json"
    checks = {
        "pinned_model_and_layer": manifest["model"] == MODEL and manifest["layer"] == 23,
        "frozen_doses_and_seed_set": tuple(manifest["dose_fractions"]) == DOSES
        and tuple(manifest["control_seeds"]) == CONTROL_SEEDS,
        "tripr_and_021_baseline_hashes": manifest["dataset_xml_sha256"] == sha(TRIPR_DATA)
        and manifest["baseline_generation_sha256"] == sha(baseline_021),
        "training_activations_match_021": manifest["training_activation_sha256"]
        == sha(ROOT_021 / "training" / "activations.npz"),
        "protocol_and_runner_hashes": manifest["protocol_sha256"] == sha(PROTOCOL)
        and manifest["runner_sha256"] == sha(RUNNER),
        "complete_prompt_dose_seed_factorial": len(outcomes) == expected_count
        and {row["id"] for row in baseline} == expected_ids
        and {row["dose_fraction"] for row in outcomes} == set(DOSES)
        and {row["seed"] for row in outcomes if row["condition"] == "random"}
        == set(CONTROL_SEEDS),
        "all_queries_present_per_arm": all(
            {row["id"] for row in outcomes if row["dose_fraction"] == dose and row["condition"] == "trained"}
            == expected_ids
            and all(
                {row["id"] for row in outcomes if row["dose_fraction"] == dose and row["condition"] == "random" and row["seed"] == seed}
                == expected_ids
                for seed in CONTROL_SEEDS
            )
            for dose in DOSES
        ),
        "no_text_or_generated_answers_saved": all(
            all(key not in row for key in ("text", "answer", "user"))
            for row in outcomes + baseline
        ),
        "local_inference_and_clean_capture": manifest["device"] in ("mps", "cpu")
        and not manifest["provenance"]["dirty_worktree"],
        "analysis_recomputes": summarize(ROOT) == analysis,
    }
    report = {
        "experiment": 22,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "analysis_code_sha256": sha(ANALYZER),
        "manifest": manifest,
    }
    if not report["checks_passed"]:
        raise ValueError("022 integrity audit failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/granite-tripr-generation-dose-response-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    analysis = summarize(ROOT)
    report = audit(analysis)
    args.output.mkdir(parents=True)
    folder = ROOT / "evaluation"
    for filename in ("baseline.json", "outcomes.json"):
        with (folder / filename).open("rb") as source, gzip.open(args.output / f"{filename}.gz", "wb") as target:
            shutil.copyfileobj(source, target)
    json_write(args.output / "analysis.json", analysis)
    json_write(args.output / "audit.json", report)
    primary = analysis["dose_results"]["0.4"]
    lines = [
        "# Granite generated-answer dose response (experiment 022)",
        "",
        "This follow-up tests target-matched learned residuals against 20 random residual seeds on the 63 TripR conflict prompts. It stores answer validity and correctness only, not generated text.",
        "",
        f"Baseline generated accuracy: {analysis['baseline_generated_accuracy']:.1%}.",
        f"Primary 40% dose: trained-minus-random utility {primary['trained_minus_random_utility']:+.4f} (nested 95% CI [{primary['nested_sentence_seed_bootstrap_95_ci'][0]:+.4f}, {primary['nested_sentence_seed_bootstrap_95_ci'][1]:+.4f}]); random-seed rank p={primary['random_seed_monte_carlo_one_sided_p']:.4f}; behavioral gate {'PASS' if primary['behavioral_gate'] else 'FAIL'}.",
        "",
        "| Dose | Trained accuracy | Exact one-word | Label flips | Gains / harms | Trained − random utility | 95% CI | Rank p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dose in DOSES:
        row = analysis["dose_results"][str(dose)]
        ci = row["nested_sentence_seed_bootstrap_95_ci"]
        lines.append(
            f"| {dose:.0%} | {row['trained_generated_strict_accuracy']:.1%} | {row['trained_exact_one_word_rate']:.1%} | {row['trained_label_flips_vs_baseline']} | {row['trained_accuracy_gains']} / {row['trained_accuracy_harms']} | {row['trained_minus_random_utility']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['random_seed_monte_carlo_one_sided_p']:.4f} |"
        )
    lines += [
        "",
        "The 40% dose is the sole primary endpoint; lower doses are descriptive. This reuses the same checkpoint and prompts as 021 and is not an independent replication. See the [frozen protocol](../../docs/experiments/022-granite-generated-dose-response.md), seed-level analysis, provenance, and audit.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines))
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (args.output / "SHA256SUMS").write_text(checksums)
    print(f"Packaged experiment 022 in {args.output}; audit passed={report['checks_passed']}")


if __name__ == "__main__":
    main()
