"""Audit and package the Granite SemEval score/generation alignment study."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from analyze_granite_semeval_margin_generation import analyze
from granite_semeval_margin_generation import DATA, PROTOCOL, ROOT
from natural_aspect_selectivity import load_stimuli
from prompt_effect_forecast import sha

RUNNER = Path("scripts/granite_semeval_margin_generation.py")
ANALYZER = Path("scripts/analyze_granite_semeval_margin_generation.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit(analysis):
    outcomes = json.loads((ROOT / "outcomes.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    stimuli = load_stimuli(DATA)
    checks = {
        "frozen_stimulus_filter_and_unique_ids": len(stimuli) == 233
        and len({row["sentence_id"] for row in stimuli}) == 112
        and {row["id"] for row in outcomes} == {row["id"] for row in stimuli},
        "model_checkpoint_and_local_device": manifest["model"]["revision"]
        == "bbc2aed595bd38bd770263dc3ab831db9794441d"
        and manifest["device"] in ("mps", "cpu"),
        "dataset_and_protocol_hashes": manifest["dataset_xml_sha256"] == sha(DATA)
        and manifest["protocol_sha256"] == sha(PROTOCOL),
        "runner_hash_matches": manifest["runner_sha256"] == sha(RUNNER),
        "clean_capture_commit": not manifest["provenance"]["dirty_worktree"],
        "raw_review_and_generation_text_absent": all(
            all(key not in row for key in ("text", "answer", "user")) for row in outcomes
        ),
        "analysis_recomputes": analyze(ROOT) == analysis,
    }
    report = {
        "experiment": 23,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "analysis_code_sha256": sha(ANALYZER),
        "manifest": manifest,
    }
    if not report["checks_passed"]:
        raise ValueError("023 integrity audit failed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/granite-semeval-margin-generation-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = analyze(ROOT)
    report = audit(result)
    args.output.mkdir(parents=True)
    with (ROOT / "outcomes.json").open("rb") as source, gzip.open(
        args.output / "outcomes.json.gz", "wb"
    ) as target:
        shutil.copyfileobj(source, target)
    write_json(args.output / "analysis.json", result)
    write_json(args.output / "audit.json", report)
    (args.output / "DATA_ATTRIBUTION.md").write_text(
        "# Data attribution\n\n"
        "SemEval-2014 Task 4 restaurant test annotations, described at the "
        "[official task page](https://alt.qcri.org/semeval2014/task4/). The frozen XML "
        "was read from the public mirror [HSLCY/ABSA-BERT-pair](https://github.com/HSLCY/ABSA-BERT-pair/blob/master/data/semeval2014/Restaurants_Test_Gold.xml), "
        "SHA-256 `f21509cfa37e16534cd5b2da043be487355b64ef48fe8d6aaacaeca6b49cc0fb`. "
        "No review text or generated text is redistributed in this bundle. Consult the "
        "upstream source and task organizers for the original data terms.\n"
    )
    lines = [
        "# Granite candidate scores and generated answers (experiment 023)",
        "",
        "This observational test compares the sign of positive-versus-negative candidate logits with greedy generated labels on 233 SemEval aspect prompts (112 sentences). It tests score/generation alignment, not steering or causality.",
        "",
        f"- Exact one-word generation rate: {result['generated_exact_one_word_rate']:.1%}.",
        f"- Candidate-pair accuracy: {result['candidate_pair_accuracy']:.1%}; generated accuracy: {result['generated_strict_accuracy']:.1%}; unrestricted next-token top-1 accuracy: {result['all_vocabulary_top1_accuracy']:.1%}.",
        f"- Candidate-pair/generation agreement: {result['candidate_pair_generation_agreement']:.1%} (sentence-bootstrap 95% CI [{result['candidate_pair_generation_agreement_sentence_bootstrap_95_ci'][0]:.1%}, {result['candidate_pair_generation_agreement_sentence_bootstrap_95_ci'][1]:.1%}]).",
        f"- Generation errors: {result['n_generation_errors']}; secondary error-detection AUC: {result['gold_aligned_margin_error_detection_auc_secondary']}.",
        f"- Frozen measurement gate: {'PASS' if result['primary_measurement_gate_passed'] else 'FAIL'}.",
        "",
        "See the [frozen protocol](../../docs/experiments/023-granite-score-generation-alignment.md), analysis, audit, and data attribution.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines))
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (args.output / "SHA256SUMS").write_text(checksums)
    print(f"Packaged experiment 023 in {args.output}; audit passed={report['checks_passed']}")


if __name__ == "__main__":
    main()
