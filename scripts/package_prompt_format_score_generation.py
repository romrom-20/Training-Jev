"""Audit and package the paired SemEval prompt-format score/generation study."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from analyze_prompt_format_score_generation import analyze
from natural_aspect_selectivity import DATA, load_stimuli
from prompt_effect_forecast import sha
from prompt_format_score_generation import CONFIGS, FORMATS, MODELS, PROTOCOL, ROOT

RUNNER = Path("scripts/prompt_format_score_generation.py")
ANALYZER = Path("scripts/analyze_prompt_format_score_generation.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit(result):
    expected = {row["id"] for row in load_stimuli(DATA)}
    checks = {
        "frozen_dataset_ids_and_hashes": all(
            len(json.loads((ROOT / name / "outcomes.json").read_text())) == 466
            and {
                row["id"] for row in json.loads((ROOT / name / "outcomes.json").read_text())
            }
            == expected
            and json.loads((ROOT / name / "manifest.json").read_text())["dataset_xml_sha256"]
            == sha(DATA)
            for name in MODELS
        ),
        "model_revision_matches_frozen_config": all(
            json.loads((ROOT / name / "manifest.json").read_text())["model"]["revision"]
            == CONFIGS[name]["revision"]
            for name in MODELS
        ),
        "format_and_protocol_runner_hashes": all(
            tuple(json.loads((ROOT / name / "manifest.json").read_text())["formats"])
            == FORMATS
            and json.loads((ROOT / name / "manifest.json").read_text())["protocol_sha256"]
            == sha(PROTOCOL)
            and json.loads((ROOT / name / "manifest.json").read_text())["runner_sha256"]
            == sha(RUNNER)
            for name in MODELS
        ),
        "local_clean_capture": all(
            json.loads((ROOT / name / "manifest.json").read_text())["device"] in ("mps", "cpu")
            and not json.loads((ROOT / name / "manifest.json").read_text())["provenance"]["dirty_worktree"]
            for name in MODELS
        ),
        "raw_review_and_generation_text_absent": all(
            all(
                all(key not in row for key in ("text", "answer", "user"))
                for row in json.loads((ROOT / name / "outcomes.json").read_text())
            )
            for name in MODELS
        ),
        "analysis_recomputes": analyze(ROOT) == result,
    }
    report = {
        "experiment": 25,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "analysis_code_sha256": sha(ANALYZER),
    }
    if not report["checks_passed"]:
        raise ValueError("025 integrity audit failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/prompt-format-score-generation-v1"))
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
    ci = result["paired_sentence_bootstrap_95_ci"]
    lines = [
        "# Prompt-format effect on score/generation agreement (experiment 025)",
        "",
        "This paired ablation compares explicit one-word labels with an open sentiment question on 233 SemEval prompts across Granite, Qwen and SmolLM2. Generated text is not retained.",
        "",
        "| Model | Format | Exact one-word | Parseable polarity | Candidate-pair accuracy | Pair/generation agreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, formats in result["models"].items():
        for fmt, row in formats.items():
            agreement = row["pair_generation_agreement_among_parseable"]
            agreement_text = f"{agreement:.1%}" if agreement is not None else "n/a"
            lines.append(
                f"| {name} | {fmt} | {row['exact_one_word_rate']:.1%} | {row['parseable_polarity_rate']:.1%} | {row['candidate_pair_accuracy']:.1%} | {agreement_text} |"
            )
    point = result["open_minus_forced_pair_generation_agreement"]
    point_text = f"{point:+.3%}" if point is not None else "n/a"
    ci_text = f"[{ci[0]:+.3%}, {ci[1]:+.3%}]" if ci[0] is not None else "n/a"
    lines += [
        "",
        f"Open minus forced-choice agreement: {point_text} (paired sentence-bootstrap 95% CI {ci_text}).",
        f"Open-label extraction coverage >=90% in all families: {result['all_models_open_parseable_coverage_at_least_90_percent']}. Format effect detected under the frozen rule: {result['format_effect_detected']}.",
        "",
        "This is a prompt-format measurement test, not a causal steering experiment. See the [frozen protocol](../../docs/experiments/025-prompt-format-score-generation.md), per-model manifests, audit, and data attribution.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines))
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (args.output / "SHA256SUMS").write_text(checksums)
    print(f"Packaged experiment 025; audit passed={report['checks_passed']}")


if __name__ == "__main__":
    main()
