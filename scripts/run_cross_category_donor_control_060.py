"""Run polarity-matched donor swaps constrained to a different aspect category."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import run_category_matched_context_swap_057 as exp057
from run_expanded_opinion_mask_repair_043 import candidates, generate_batch
from run_laptop_decoder_transfer_049 import SOURCE_REVISION, prompt_with_registered_grid
from run_small_model_decoder_factorial_048 import (
    DEFAULT_BATCH,
    MAX_NEW_TOKENS,
    build_trie,
    candidate_sequences,
    generate_free_batch,
    homogeneous_batches,
    parse_finite_va,
    parse_free_va,
    sha256,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/060-cross-category-donor-control.md")
PROTOCOL_SHA256 = "cfb193ecbdb0a5c58ce815c37d23ef00820f6cd24ec4713ae760d6a1c511aa6d"
EXP050_OUTPUT = Path(".context/exp050-private-predictions.jsonl")
EXP050_OUTPUT_SHA256 = "890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f"
EXP057_OUTPUT = Path(".context/exp057-private-predictions.jsonl")
EXP057_OUTPUT_SHA256 = "21e7fe02b43433fa4b61b5dfe790efef38c57e8a0e240e4443ffe40ffb110e31"
SEEDS = (20260601, 20260602, 20260603)
DECODERS = ("finite_grid", "free_greedy")
CONDITION = "cross_category_polarity_matched_context"
OUT = Path(".context/exp060-private-predictions.jsonl")
MANIFEST = Path(".context/exp060-run-manifest.json")


def seeded_key(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()


def cross_category_maps(cases: list[dict], tokenizer) -> tuple[dict[int, dict[str, str]], dict]:
    lengths = {
        case["case_id"]: len(tokenizer.encode(
            exp057.exp053.mask_opinions(case["row"]), add_special_tokens=False
        )) for case in cases
    }
    categories = {case["case_id"]: case["category"] for case in cases}
    polarities = {case["case_id"]: exp057.exp055.bucket(case) for case in cases}
    polarity_groups = {
        polarity: [case["case_id"] for case in cases if polarities[case["case_id"]] == polarity]
        for polarity in ("neg", "pos")
    }
    maps = {}
    for permutation, seed in enumerate(SEEDS, start=1):
        assignment = {}
        for polarity, ids in polarity_groups.items():
            cell_sizes = Counter(categories[case_id] for case_id in ids)
            recipients = sorted(ids, key=lambda case_id: (
                cell_sizes[categories[case_id]], lengths[case_id], seeded_key(seed, case_id), case_id
            ))
            eligible_donors = {
                case_id: sorted(
                    (candidate for candidate in ids if categories[candidate] != categories[case_id]),
                    key=lambda candidate: (
                        abs(lengths[case_id] - lengths[candidate]),
                        seeded_key(seed, candidate), candidate,
                    ),
                ) for case_id in ids
            }
            donor_to_recipient = {}

            def augment(recipient: str, visited: set[str]) -> bool:
                for donor in eligible_donors[recipient]:
                    if donor in visited:
                        continue
                    visited.add(donor)
                    previous_recipient = donor_to_recipient.get(donor)
                    if previous_recipient is None or augment(previous_recipient, visited):
                        donor_to_recipient[donor] = recipient
                        return True
                return False

            for recipient in recipients:
                if not augment(recipient, set()):
                    raise ValueError(f"No complete cross-category matching for {polarity}")
            assignment.update({recipient: donor for donor, recipient in donor_to_recipient.items()})
        if set(assignment) != set(categories) or set(assignment.values()) != set(categories):
            raise ValueError(f"Assignment {permutation} is not a full bijection")
        for recipient, donor in assignment.items():
            if donor == recipient or polarities[recipient] != polarities[donor]:
                raise ValueError("Cross-category donor violates self/polarity constraint")
            if categories[recipient] == categories[donor]:
                raise ValueError("Cross-category donor retained recipient's official category")
        maps[permutation] = assignment
    unique_count = len({tuple(sorted(mapping.items())) for mapping in maps.values()})
    if unique_count != 3:
        raise ValueError(f"Expected three globally distinct donor maps, got {unique_count}")
    return maps, {
        "token_lengths": lengths,
        "n_unique_assignment_maps": unique_count,
        "all_donors_same_polarity": True,
        "all_donors_different_category": True,
        "polarity_counts": dict(Counter(polarities.values())),
    }


def verify_parents(cases: list[dict]) -> tuple[dict, str, str]:
    if sha256(EXP050_OUTPUT.read_bytes()) != EXP050_OUTPUT_SHA256:
        raise ValueError("Experiment 050 baseline output hash mismatch")
    if sha256(EXP057_OUTPUT.read_bytes()) != EXP057_OUTPUT_SHA256:
        raise ValueError("Experiment 057 same-category output hash mismatch")
    own_rows = [json.loads(line) for line in EXP050_OUTPUT.read_text().splitlines() if line.strip()]
    own_by_key = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in own_rows if row["condition"] == "opinion_masked"
    }
    same_rows = [json.loads(line) for line in EXP057_OUTPUT.read_text().splitlines() if line.strip()]
    same_by_id = {}
    for row in same_rows:
        if row["decoder"] == "finite_grid" and row["permutation"] == 1:
            same_by_id[row["case_id"]] = row
    for case in cases:
        case_id = case["case_id"]
        for decoder in DECODERS:
            own = own_by_key.get((case_id, "opinion_masked", decoder))
            same = next((row for row in same_rows if row["case_id"] == case_id
                         and row["permutation"] == 1 and row["decoder"] == decoder), None)
            if own is None or same is None or own["gold"] != case["gold"] or same["gold"] != case["gold"]:
                raise ValueError(f"Missing or mismatched frozen gold/baseline for {case_id}/{decoder}")
        row = same_by_id.get(case_id)
        if row is None or row["donor_category"] != case["category"]:
            raise ValueError(f"Experiment 057 row is not in the same-category arm: {case_id}")
    return own_by_key, EXP050_OUTPUT_SHA256, EXP057_OUTPUT_SHA256


def build_jobs(cases: list[dict], maps: dict[int, dict[str, str]]) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for permutation, mapping in maps.items():
        for case in cases:
            donor = by_id[mapping[case["case_id"]]]
            target = case["row"]["Triplet"][case["target_index"]]
            prompt = prompt_with_registered_grid(
                exp057.exp053.mask_opinions(donor["row"]), target["Aspect"]
            )
            for decoder in DECODERS:
                jobs.append({
                    "case_id": case["case_id"], "permutation": permutation,
                    "condition": CONDITION, "decoder": decoder, "gold": case["gold"],
                    "donor_id": donor["case_id"], "donor_category": donor["category"],
                    "donor_polarity": exp057.exp055.bucket(donor), "prompt": prompt,
                })
    if len(jobs) != 184 * 3 * 2:
        raise ValueError(f"Expected 1,104 jobs, got {len(jobs)}")
    return sorted(jobs, key=lambda row: (row["permutation"], row["decoder"], row["case_id"]))


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        batch_size: int = DEFAULT_BATCH, preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 060 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    task2_rows, task2_hash = exp057.exp053.read_source(source_dir)
    if task2_hash != exp057.exp053.SOURCE_SHA256:
        raise ValueError("Pinned Task 2 source hash mismatch")
    task3_rows, task3_hash = exp057.read_task3(source_dir)
    fresh = exp057.exp055.fresh_sample(exp057.exp053.select_cases(task2_rows))
    cases, sample_stats = exp057.select_category_cases(exp057.attach_categories(fresh, task3_rows))
    _, own_hash, same_hash = verify_parents(cases)
    config = dict(MODEL_SPECS["qwen-3b"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    maps, donor_meta = cross_category_maps(cases, tokenizer)
    jobs = build_jobs(cases, maps)
    if preflight_only:
        token_sequences = candidate_sequences(tokenizer, candidates())
        prompt_lengths = [len(tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=True, add_generation_prompt=True,
        )) for job in jobs]
        return {
            "experiment": "060-cross-category-donor-control",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "task2_sha256": task2_hash, "task3_sha256": task3_hash,
            "exp050_output_sha256": own_hash, "exp057_output_sha256": same_hash,
            **sample_stats,
            "n_jobs": len(jobs), "n_permutations": len(maps),
            "donor_assignment_sha256": {
                permutation: sha256(json.dumps(mapping, sort_keys=True).encode())
                for permutation, mapping in maps.items()
            },
            "n_unique_assignment_maps": donor_meta["n_unique_assignment_maps"],
            "donor_constraints": {key: donor_meta[key] for key in (
                "all_donors_same_polarity", "all_donors_different_category", "polarity_counts"
            )},
            "n_candidate_values": len(candidates()),
            "n_unique_candidate_token_sequences": len(token_sequences),
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "model_revision": config["revision"],
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["permutation"], row["condition"], row["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate output row: {key}")
            previous[key] = row
    allowed = {
        (job["case_id"], job["permutation"], job["condition"], job["decoder"])
        for job in jobs
    }
    if not set(previous).issubset(allowed):
        raise ValueError("Resume file contains rows outside the frozen design")
    remaining = [job for job in jobs if (
        job["case_id"], job["permutation"], job["condition"], job["decoder"]
    ) not in previous]
    del tokenizer
    model, tokenizer, actual_device = load_target(config, device, offline=True)
    tokenizer.padding_side = "left"
    trie = build_trie(candidate_sequences(tokenizer, candidates()))
    set_seed(20260960)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        processed = 0
        for batch in homogeneous_batches(remaining, batch_size):
            decoder = batch[0]["decoder"]
            raws = (generate_batch(model, tokenizer, batch, trie, actual_device)
                    if decoder == "finite_grid"
                    else generate_free_batch(model, tokenizer, batch, actual_device))
            for job, raw in zip(batch, raws):
                prediction = parse_finite_va(raw) if decoder == "finite_grid" else parse_free_va(raw)
                row = {key: job[key] for key in (
                    "case_id", "permutation", "condition", "decoder", "gold", "donor_id",
                    "donor_category", "donor_polarity",
                )}
                row.update({"raw": raw, "prediction": prediction})
                key = (row["case_id"], row["permutation"], row["condition"], decoder)
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"060 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "060-cross-category-donor-control",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_revision": SOURCE_REVISION,
        "source_sha256": task2_hash, "task2_sha256": task2_hash, "task3_sha256": task3_hash,
        "exp050_output_sha256": own_hash, "exp057_output_sha256": same_hash,
        "parent_output_sha256": same_hash,
        "model": config["model"], "model_revision": config["revision"],
        "device": actual_device, "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS, **sample_stats,
        "n_permutations": len(maps),
        "donor_assignment_sha256": {
            permutation: sha256(json.dumps(mapping, sort_keys=True).encode())
            for permutation, mapping in maps.items()
        },
        "n_unique_assignment_maps": donor_meta["n_unique_assignment_maps"],
        "all_donors_same_polarity": donor_meta["all_donors_same_polarity"],
        "all_donors_different_category": donor_meta["all_donors_different_category"],
        "n_outputs": len(previous),
        "invalid_by_decoder": {
            decoder: sum(row["decoder"] == decoder and row["prediction"] is None
                         for row in previous.values()) for decoder in DECODERS
        },
        "generation_seconds_this_process_only": time.monotonic() - started,
        "resumed_rows": len(jobs) - len(remaining),
        "output_sha256": sha256(output.read_bytes()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(".context/dimabsa"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=MANIFEST)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
