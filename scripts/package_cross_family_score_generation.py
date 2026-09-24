"""Audit and package cross-family SemEval score/generation outcomes."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from analyze_cross_family_score_generation import analyze
from cross_family_score_generation import MODELS, PROTOCOL, ROOT
from cross_family_score_generation import __file__ as runner_file
from natural_aspect_selectivity import DATA, load_stimuli
from prompt_effect_forecast import sha
from task_ladder import MODEL_SPECS

RUNNER = Path(runner_file)
ANALYZER = Path("scripts/analyze_cross_family_score_generation.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit(result):
    stimuli = load_stimuli(DATA)
    expected_ids = {row["id"] for row in stimuli}
    checks = {
        "frozen_dataset_hash": all(
            json.loads((ROOT / name / "manifest.json").read_text())["dataset_xml_sha256"] == sha(DATA)
            for name in MODELS
        ),
        "pinned_models_layers_and_revisions": all(
            json.loads((ROOT / name / "manifest.json").read_text())["model"]["revision"]
            == MODEL_SPECS[name]["revision"]
            for name in MODELS
        ),
        "complete_prompt_sets": all(
            len(json.loads((ROOT / name / "outcomes.json").read_text())) == 233
            and {
                row["id"] for row in json.loads((ROOT / name / "outcomes.json").read_text())
            }
            == expected_ids
            for name in MODELS
        ),
        "protocol_runner_hashes": all(
            json.loads((ROOT / name / "manifest.json").read_text())["protocol_sha256"]
            == sha(PROTOCOL)
            and json.loads((ROOT / name / "manifest.json").read_text())["runner_sha256"]
            == sha(RUNNER)
            for name in MODELS
        ),
        "clean_local_capture": all(
            json.loads((ROOT / name / "manifest.json").read_text())["device"] in ("mps", "cpu")
            and not json.loads((ROOT / name / "manifest.json").read_text())["provenance"]["dirty_worktree"]
            for name in MODELS
        ),
        "no_review_or_generated_text": all(
            all(
                all(key not in row for key in ("text", "answer", "user"))
                for row in json.loads((ROOT / name / "outcomes.json").read_text())
            )
            for name in MODELS
        ),
        "analysis_recomputes": analyze(ROOT) == result,
    }
    report = {
        "experiment": 24,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "analysis_code_sha256": sha(ANALYZER),
    }
    if not report["checks_passed"]:
        raise ValueError("024 integrity audit failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/cross-family-score-generation-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = analyze(ROOT)
    report = audit(result)
    args.output.mkdir(parents=True)
    for name in MODELS:
        with (ROOT / name / "outcomes.json").open("rb") as source, gzip.open(
            args.output / f"{name}-outcomes.json.gz", "wb"
        ) as target:
            shutil.copyfileobj(source, target)
        write_json(
            args.output / f"{name}-manifest.json",
            json.loads((ROOT / name / "manifest.json").read_text()),
        )
    write_json(args.output / "analysis.json", result)
    write_json(args.output / "audit.json", report)
    (args.output / "DATA_ATTRIBUTION.md").write_text(
        Path("results/granite-semeval-margin-generation-v1/DATA_ATTRIBUTION.md").read_text()
    )
    lines = [
        "# Cross-family score/generation alignment (experiment 024)",
        "",
        "This observational replication applies the same 233 SemEval prompts to Qwen2.5-1.5B and SmolLM2-1.7B. It stores no review or generated text.",
        "",
        "| Model | Exact one-word | Candidate-pair accuracy | Generated accuracy | Pair/generation agreement (95% CI) | All-vocabulary top-1 | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in result["models"].items():
        ci = row["candidate_pair_generation_agreement_sentence_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {row['generated_exact_one_word_rate']:.1%} | {row['candidate_pair_accuracy']:.1%} | {row['generated_strict_accuracy']:.1%} | {row['candidate_pair_generation_agreement']:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] | {row['all_vocabulary_top1_accuracy']:.1%} | {'PASS' if row['measurement_gate_passed'] else 'FAIL'} |"
        )
    ci = result["paired_sentence_bootstrap_difference_95_ci"]
    lines += [
        "",
        f"SmolLM2 minus Qwen agreement: {result['smollm2_minus_qwen_agreement_difference']:+.3%} (paired sentence-bootstrap 95% CI [{ci[0]:+.3%}, {ci[1]:+.3%}]).",
        f"Cross-family replication gate: {'PASS' if result['cross_family_replication_passed'] else 'FAIL'}.",
        "",
        "This is a noncausal measurement replication on one dataset; it does not establish general calibration or intervention effects. See the [frozen protocol](../../docs/experiments/024-cross-family-score-generation.md), seed, audit, and data attribution.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines))
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (args.output / "SHA256SUMS").write_text(checksums)
    print(f"Packaged experiment 024; audit passed={report['checks_passed']}")


if __name__ == "__main__":
    main()
