"""Analyze how removing answer-format constraints changes score/answer agreement."""

import argparse
import json
from pathlib import Path

import numpy as np
from prompt_format_score_generation import FORMATS, MODELS, ROOT

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261026


def analyze(root=ROOT):
    model_results = {}
    paired_by_sentence = {}
    for name in MODELS:
        outcomes = json.loads((root / name / "outcomes.json").read_text())
        by_format = {
            fmt: {row["id"]: row for row in outcomes if row["format"] == fmt}
            for fmt in FORMATS
        }
        model_results[name] = {}
        for fmt in FORMATS:
            rows = list(by_format[fmt].values())
            parseable = [row for row in rows if row["parseable_polarity"]]
            agreements = [
                row["candidate_pair_prediction"] == row["generated_label"]
                for row in parseable
            ]
            model_results[name][fmt] = {
                "n_prompts": len(rows),
                "exact_one_word_rate": float(np.mean([row["exact_one_word"] for row in rows])),
                "parseable_polarity_rate": float(np.mean([row["parseable_polarity"] for row in rows])),
                "candidate_pair_accuracy": float(
                    np.mean([row["candidate_pair_prediction"] == row["label"] for row in rows])
                ),
                "generated_accuracy_among_parseable": float(
                    np.mean([row["generated_strict_correct"] for row in parseable])
                )
                if parseable
                else None,
                "pair_generation_agreement_among_parseable": float(np.mean(agreements))
                if agreements
                else None,
                "n_parseable": len(parseable),
            }
        common = set(by_format[FORMATS[0]]).intersection(by_format[FORMATS[1]])
        for stimulus_id in common:
            forced = by_format["forced_choice"][stimulus_id]
            opened = by_format["open_question"][stimulus_id]
            if forced["parseable_polarity"] and opened["parseable_polarity"]:
                paired_by_sentence.setdefault(forced["sentence_id"], []).append(
                    {
                        "forced": int(
                            forced["candidate_pair_prediction"] == forced["generated_label"]
                        ),
                        "open": int(
                            opened["candidate_pair_prediction"] == opened["generated_label"]
                        ),
                    }
                )

    sentences = sorted(paired_by_sentence)
    if sentences:
        point = float(
            np.mean(
                [row["open"] - row["forced"] for sid in sentences for row in paired_by_sentence[sid]]
            )
        )
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = np.empty(BOOTSTRAP_REPS)
        for index in range(BOOTSTRAP_REPS):
            selected = rng.choice(sentences, size=len(sentences), replace=True)
            sampled = [row for sid in selected for row in paired_by_sentence[sid]]
            draws[index] = np.mean([row["open"] - row["forced"] for row in sampled])
        ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    else:
        point = None
        ci = [None, None]
    all_open_coverage = [model_results[name]["open_question"]["parseable_polarity_rate"] for name in MODELS]
    result = {
        "models": model_results,
        "n_paired_sentences": len(sentences),
        "open_minus_forced_pair_generation_agreement": point,
        "paired_sentence_bootstrap_95_ci": ci,
        "all_models_open_parseable_coverage_at_least_90_percent": all(
            value >= 0.90 for value in all_open_coverage
        ),
        "format_effect_detected": all(value >= 0.90 for value in all_open_coverage)
        and ci[0] is not None
        and (ci[0] > 0 or ci[1] < 0),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "noncausal_observational_test": True,
    }
    write_json(root / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2), flush=True)


if __name__ == "__main__":
    main()
