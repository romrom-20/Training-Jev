"""Prepare the private, blinded human coding form for experiment 029."""

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import torch
from analyze_prompt_format_score_generation import MODELS
from natural_aspect_selectivity import DATA, load_stimuli
from prompt_format_score_generation import CONFIGS, prompt_for

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/029-blinded-human-audit.md")
OUT = Path(".context/human-coding-029")
SEED = 20260925
QUOTAS = {
    ("food", 1): 4,
    ("food", 0): 3,
    ("service", 1): 3,
    ("service", 0): 4,
    ("price", 1): 3,
    ("price", 0): 3,
}


def select_sample(stimuli, seed=SEED):
    rng = random.Random(seed)
    chosen, used_sentences = [], set()
    for (category, label), quota in QUOTAS.items():
        candidates = [
            item for item in stimuli if item["category"] == category and item["label"] == label
        ]
        rng.shuffle(candidates)
        selected = []
        for item in candidates:
            if item["sentence_id"] in used_sentences:
                continue
            selected.append(item)
            used_sentences.add(item["sentence_id"])
            if len(selected) == quota:
                break
        if len(selected) != quota:
            raise ValueError(f"Could not fill frozen quota for {category}/{label}")
        chosen.extend(selected)
    if len(chosen) != 20 or len({item["sentence_id"] for item in chosen}) != 20:
        raise ValueError("Sample must contain 20 distinct source sentences")
    return chosen


def generate_answers(stimuli):
    all_rows = []
    for model_name in MODELS:
        model, tokenizer, device = load_target(CONFIGS[model_name], "auto", offline=True)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_for(item, "open_question")}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in stimuli
        ]
        answers = []
        for start in range(0, len(prompts), 4):
            batch = tokenizer(prompts[start : start + 4], padding=True, return_tensors="pt").to(
                device
            )
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            answers.extend(
                tokenizer.batch_decode(
                    generated[:, batch.input_ids.shape[1] :], skip_special_tokens=True
                )
            )
        all_rows.extend(
            {"model": model_name, "stimulus": item, "answer": answer.strip()}
            for item, answer in zip(stimuli, answers)
        )
        print(f"Generated {model_name}: {len(answers)}/{len(stimuli)}", flush=True)
        del model, tokenizer
        gc.collect()
        if str(device).startswith("mps"):
            torch.mps.empty_cache()
    if len(all_rows) != 60:
        raise ValueError(f"Expected 60 paired answers, got {len(all_rows)}")
    return all_rows


def write_form(rows):
    rng = random.Random(SEED + 1)
    rng.shuffle(rows)
    private_rows = []
    blind_rows = []
    for index, row in enumerate(rows, start=1):
        blind_id = f"A{index:03d}"
        item = row["stimulus"]
        private_rows.append(
            {
                "blind_id": blind_id,
                "model": row["model"],
                "stimulus_id": item["id"],
                "sentence_id": item["sentence_id"],
                "category": item["category"],
                "gold": item["label"],
                "answer": row["answer"],
            }
        )
        blind_rows.append({"id": blind_id, "aspect": item["category"], "answer": row["answer"]})
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "private-key.json").write_text(json.dumps(private_rows, indent=2) + "\n")
    payload = json.dumps(blind_rows, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Answer coding</title>
<style>body{font:16px system-ui;max-width:850px;margin:2rem auto;padding:0 1rem;color:#18212b}article{border:1px solid #ccd3da;border-radius:10px;padding:1rem;margin:1rem 0}blockquote{white-space:pre-wrap;background:#f5f7f9;padding:1rem;border-radius:6px}.choices{display:flex;gap:1rem;flex-wrap:wrap}button{padding:.6rem 1rem;margin:.5rem .5rem .5rem 0}small{color:#566}</style>
<h1>Code the answer</h1><p>For each answer, label the sentiment it communicates about the named aspect. Use <b>mixed</b> if it clearly expresses both polarities; use <b>unclear</b> for no answer, irrelevant content, or an answer whose polarity is not clear. Judge only the answer shown.</p><p id="progress"></p><main id="items"></main><button id="export">Download labels.json</button><button id="csv">Download labels.csv</button><p><small>Your progress is saved in this browser on this device. Export before closing.</small></p>
<script>
const rows=PAYLOAD, key='human-coding-029', codes=['positive','negative','mixed','unclear'];
let saved=JSON.parse(localStorage.getItem(key)||'{}');
function render(){const root=document.querySelector('#items');root.replaceChildren();for(const row of rows){const card=document.createElement('article');card.id=row.id;const title=document.createElement('h2');title.textContent=row.id+' · Aspect: '+row.aspect;const answer=document.createElement('blockquote');answer.textContent=row.answer||'(empty answer)';card.append(title,answer);const choices=document.createElement('div');choices.className='choices';for(const code of codes){const label=document.createElement('label'),input=document.createElement('input');input.type='radio';input.name=row.id;input.value=code;input.checked=saved[row.id]===code;input.onchange=()=>{saved[row.id]=code;localStorage.setItem(key,JSON.stringify(saved));update()};label.append(input,document.createTextNode(' '+code));choices.append(label)}card.append(choices);root.append(card)}update()}
function update(){document.querySelector('#progress').textContent=Object.keys(saved).length+' / '+rows.length+' coded'}
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();URL.revokeObjectURL(a.href)}
document.querySelector('#export').onclick=()=>download('labels.json',JSON.stringify({experiment:'029',annotations:saved},null,2),'application/json');
document.querySelector('#csv').onclick=()=>download('labels.csv','blind_id,code\\n'+rows.filter(r=>saved[r.id]).map(r=>r.id+','+saved[r.id]).join('\\n'),'text/csv');render();
</script></html>""".replace("PAYLOAD", payload)
    (OUT / "index.html").write_text(page)
    sample = [
        {
            "id": item["id"],
            "sentence_id": item["sentence_id"],
            "category": item["category"],
            "gold": item["label"],
        }
        for item in {r["stimulus"]["id"]: r["stimulus"] for r in rows}.values()
    ]
    manifest = {
        "experiment": "029",
        "seed": SEED,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "n_sentences": len(sample),
        "n_answers": len(private_rows),
        "private_files": ["private-key.json"],
        "sample_strata": {
            f"{c}_{'positive' if y else 'negative'}": sum(
                r["category"] == c and r["gold"] == y for r in sample
            )
            for c, y in QUOTAS
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote blinded form: {OUT / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Print the frozen sample strata without loading models",
    )
    args = parser.parse_args()
    stimuli = load_stimuli(DATA)
    if len(stimuli) != 233:
        raise ValueError("Frozen SemEval item filter changed")
    sample = select_sample(stimuli)
    print(
        "Sample strata:",
        {
            f"{c}_{y}": sum(x["category"] == c and x["label"] == y for x in sample)
            for c, y in QUOTAS
        },
        flush=True,
    )
    if args.select_only:
        return
    write_form(generate_answers(sample))


if __name__ == "__main__":
    main()
