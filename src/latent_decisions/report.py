"""Offline, shareable report generated exclusively from saved measurements."""

import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import write_json


def render(run):
    data = json.loads((run / "metrics.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    causal_path = run / "interventions.json"
    causal = json.loads(causal_path.read_text()) if causal_path.exists() else None
    selected = data["selected_layer"]
    methods = ["bilinear", "independent_linear", "query_only", "shuffled_labels"]
    colors = ["#167d8d", "#25354b", "#a28b71", "#9aafb7"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#faf9f6",
            "axes.facecolor": "#faf9f6",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for method, color in zip(methods, colors):
        means = []
        for layer in data["config"]["layers"]:
            records = [r for r in data["records"] if r["method"] == method and r["layer"] == layer]
            means.append(
                np.mean([r["evaluations"]["test"]["calibrated"]["brier"] for r in records])
            )
        axes[0].plot(
            data["config"]["layers"], means, "o-", color=color, label=method.replace("_", " ")
        )
    axes[0].set(
        title="Test Brier by extraction layer",
        xlabel="Transformer block (1-indexed)",
        ylabel="Binary Brier score (lower is better)",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], "--", color="#999999", label="Perfect calibration")
    record = next(
        r
        for r in data["records"]
        if r["method"] == "bilinear"
        and r["layer"] == selected
        and r["seed"] == data["config"]["seeds"][0]
    )
    for split, color in (("test", colors[0]), ("ood", colors[1])):
        bins = record["evaluations"][split]["calibrated"]["reliability"]
        axes[1].plot(
            [b["probability"] for b in bins],
            [b["frequency"] for b in bins],
            "o-",
            color=color,
            label=split,
        )
    axes[1].set(
        title=f"Reliability · block {selected}, seed {record['seed']}",
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].legend(fontsize=8)
    if causal:
        names = ["color", "shape", "animal"] + [f"random_{j}" for j in range(8)]
        means = [causal["summary"][n]["mean"] for n in names]
        axes[2].barh(names, means, color=[colors[0]] + [colors[1]] * 2 + [colors[3]] * 8)
        axes[2].axvline(0, color="#999999", linewidth=0.6)
        axes[2].set(
            title=f"Steering diagnostic · block {causal['layer']}",
            xlabel="Mean Δ log[P(red) / P(blue)] (+dose − −dose)",
            ylabel="Unit intervention direction",
        )
    else:
        axes[2].remove()
    fig.suptitle("Latent Decisions / controlled real-model pilot", fontsize=15, fontweight="bold")
    fig.savefig(run / "overview.png", dpi=160)
    plt.close(fig)
    # Small, fully offline data payload for the interactive reader.
    summary = dict(
        selected_layer=selected,
        records=[
            {
                k: r[k]
                for k in ("method", "layer", "seed", "parameters", "temperature", "evaluations")
            }
            for r in data["records"]
        ],
    )
    payload = json.dumps(summary).replace("<", "\\u003c")
    collection = manifest["collection"]
    budget = f"{collection['seconds']:.1f}s capture · {data['training_seconds']:.1f}s probe fitting · {collection['peak_process_rss_gib']:.2f} GiB capture process RSS"

    def mean_accuracy(method, split):
        return np.mean(
            [
                r["evaluations"][split]["calibrated"]["accuracy"]
                for r in data["records"]
                if r["method"] == method and r["layer"] == selected
            ]
        )

    findings = (
        f"At validation-selected block {selected}, the shared probe achieves "
        f"{mean_accuracy('bilinear', 'test'):.1%} held-out accuracy versus "
        f"{mean_accuracy('independent_linear', 'test'):.1%} for independent linear probes. "
        f"On the unseen prompt template it reaches {mean_accuracy('bilinear', 'ood'):.1%}. "
        f"The privileged full-prompt text baseline reaches "
        f"{data['text_baselines']['full_prompt_text']['test']['calibrated']['accuracy']:.1%}. "
        "This establishes an executable pipeline, not a gain over existing probes."
    )
    causal_note = ""
    if causal:
        causal_note = f"""<section><p class="eyebrow">INTERVENTIONS</p><h2>Reading and influencing are separate tests.</h2>
        <p>We added and subtracted fixed-norm directions at block {causal["layer"]} on {causal["prompts"]} held-out prompts.
        The outcome is a change in the target model’s red-versus-blue log odds. Eight equal-norm random directions
        and two other properties act as comparators. Per-prompt vocabulary KL and color-token mass are saved.</p>
        <p class="note">Exploratory result. Steering sufficiency does not establish necessity, natural use, or a uniquely identified mechanism.
        This pilot does not evaluate deception or hidden intent.</p></section>"""
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Latent Decisions — local research pilot</title><style>
:root{color-scheme:light;--ink:#202c3b;--muted:#61707c;--line:#d7dddc;--accent:#167d8d;--paper:#faf9f6}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:42px 30px}nav{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:18px;font-size:13px}.eyebrow{font-size:11px;letter-spacing:.15em;font-weight:700;color:var(--accent)}h1{font-size:clamp(36px,5vw,62px);line-height:1.05;letter-spacing:-.05em;max-width:850px;margin:24px 0}h2{font-size:25px;letter-spacing:-.025em}header{padding:34px 0 32px}.lead{font-size:20px;max-width:750px;color:var(--muted)}.note{border-left:3px solid var(--accent);padding:12px 20px;background:#f0f3f1;color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:30px;border-block:1px solid var(--line);padding:24px 0}.stat b{font-size:27px;display:block}.stat span{font-size:13px;color:var(--muted)}section{margin:42px 0}img{width:100%;height:auto}label{font-size:13px;margin-right:20px;display:inline-block}select{display:block;background:white;border:1px solid var(--line);padding:8px 28px 8px 10px;color:var(--ink);font:inherit}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted)}.scroll{overflow-x:auto}footer{border-top:1px solid var(--line);padding-top:20px;font-size:12px;color:var(--muted)}a{color:var(--accent)}.two{display:grid;grid-template-columns:1fr 1fr;gap:48px}@media(max-width:650px){.stats,.two{grid-template-columns:1fr}main{padding:22px 18px}}
</style><main><nav><strong>LATENT DECISIONS</strong><span>RESEARCH NOTE / 001 · SEPTEMBER 2026</span></nav>
<header><p class="eyebrow">SMALL PROBES. TESTABLE CLAIMS.</p><h1>What can we read?<br>What actually matters?</h1><p class="lead">A laptop-scale experiment in asking precise questions of a language model’s activations—and checking the answers.</p><p class="note">Status: controlled real-model pilot. Three known instruction properties; no claim of general semantic transfer, deception detection, or novelty over activation oracles.</p></header>
<div class="stats"><div class="stat"><b>416</b><span>fully crossed prompts · five isolated splits</span></div><div class="stat"><b>0.5B / frozen</b><span>Qwen2.5 target · CPU probe training</span></div><div class="stat"><b>24 GB laptop</b><span>Apple Silicon · no cloud compute</span></div></div>
<section><p class="eyebrow">MEASUREMENTS, NOT MOCK DATA</p><h2>A small experiment with explicit controls.</h2><p>__BUDGET__. Process RSS is not total unified-memory usage. Capture excludes probe fitting and interventions; these are wall-clock measurements from this run, not performance guarantees.</p><img src="overview.png" alt="Measured held-out Brier score by transformer block, calibration reliability, and target-model steering effects"><p style="font-size:12px;color:var(--muted)">Source: local Qwen2.5-0.5B-Instruct run, 22 September 2026. Brier curves average three training seeds; reliability uses the first seed. Steering averages fixed test prompts, with a dose of 5% of the median training activation norm.</p></section>
<section><p class="eyebrow">EXPLORE THE SAVED RUN</p><h2>Compare every baseline.</h2><label>Layer<select id="layer"></select></label><label>Evaluation<select id="split"><option value="test">Held-out scenarios</option><option value="ood">Held-out prompt template</option></select></label><label>Probability readout<select id="readout"><option value="calibrated">Temperature-scaled</option><option value="raw">Raw</option></select></label><div class="scroll"><table><thead><tr><th>Method</th><th>Parameters</th><th>Accuracy</th><th>Brier ↓</th><th>NLL ↓</th><th>ECE ↓</th></tr></thead><tbody id="metrics"></tbody></table></div><p style="font-size:13px;color:var(--muted)">Means over three seeds. Layer selected using validation NLL only. Temperature fitted on a separate calibration split. Small ECE does not prove calibration under general distribution shift.</p></section>
__CAUSAL__
<section class="two"><div><p class="eyebrow">WHAT THIS SUPPORTS</p><h2>An executable starting point.</h2><p>The system captures actual residual-stream states, fits a rank-constrained question-conditioned readout, checks leakage controls, and measures interventions on the original target.</p></div><div><p class="eyebrow">WHAT COMES NEXT</p><h2>A harder test of transfer.</h2><p>Hold out entire property families, reverse irrelevant correlations, compare against a trained activation oracle, and measure whether improvements survive new models and intervention distributions.</p></div></section>
<footer>Artifacts: <a href="metrics.json">metrics</a> · <a href="manifest.json">manifest</a> · <a href="interventions.json">interventions</a> · <a href="predictions.json">per-example predictions</a><br>Labels describe explicit prompt settings. These are not labels for beliefs or intent. Full-prompt text is a privileged baseline and can directly read the settings.</footer>
<script>const data=__DATA__;const layer=document.querySelector('#layer');for(const n of [...new Set(data.records.map(r=>r.layer))]){const option=document.createElement('option');option.value=n;option.textContent='Block '+n;option.selected=n===data.selected_layer;layer.append(option)}function render(){const split=document.querySelector('#split').value;const readout=document.querySelector('#readout').value;const rows=data.records.filter(r=>r.layer===Number(layer.value));const body=document.querySelector('#metrics');body.replaceChildren();for(const method of [...new Set(rows.map(r=>r.method))]){const group=rows.filter(r=>r.method===method);const mean=key=>group.reduce((sum,r)=>sum+r.evaluations[split][readout][key],0)/group.length;const tr=document.createElement('tr');const values=[method.replaceAll('_',' '),group[0].parameters.toLocaleString(),(100*mean('accuracy')).toFixed(1)+'%',mean('brier').toFixed(4),mean('nll').toFixed(4),mean('ece').toFixed(4)];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}body.append(tr)}}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',render));render();</script></main></html>"""
    # Derive display counts instead of baking in one configuration.
    document = document.replace("<b>416</b>", f"<b>{sum(collection['split_counts'].values())}</b>")
    document = document.replace(
        "<b>0.5B / frozen</b>", f"<b>{collection['model_parameters'] / 1e9:.2f}B / frozen</b>"
    )
    document = document.replace(
        "Qwen2.5 target", html.escape(data["config"]["model"].split("/")[-1])
    )
    document = document.replace(
        "<b>24 GB laptop</b><span>Apple Silicon · no cloud compute</span>",
        f"<b>{collection['device'].upper()} capture</b><span>Local execution · no model API</span>",
    )
    document = document.replace(
        "<h2>Compare every baseline.</h2>", "<h2>Compare activation-head baselines.</h2>"
    )
    document = document.replace(
        "<h2>A small experiment with explicit controls.</h2>",
        "<h2>A small experiment with explicit controls.</h2><p>" + findings + "</p>",
    )
    nseeds = len(data["config"]["seeds"])
    document = document.replace("three training seeds", f"{nseeds} training seeds").replace(
        "Means over three seeds", f"Means over {nseeds} seeds"
    )
    date = manifest.get("collected_at", "date unrecorded")[:10]
    document = document.replace("22 September 2026", html.escape(date)).replace(
        "SEPTEMBER 2026", html.escape(date)
    )
    document = document.replace(
        "Source: local Qwen2.5-0.5B-Instruct run",
        "Source: local " + html.escape(data["config"]["model"]) + " run",
    )
    if causal:
        document = document.replace("dose of 5%", f"dose of {causal['dose_fraction']:.0%}")
    else:
        document = document.replace(' · <a href="interventions.json">interventions</a>', "")
        document = document.replace(
            " Steering averages fixed test prompts, with a dose of 5% of the median training activation norm.",
            "",
        )
    document = (
        document.replace("__BUDGET__", html.escape(budget))
        .replace("__CAUSAL__", causal_note)
        .replace("__DATA__", payload)
    )
    (run / "report.html").write_text(document)
    write_json(run / "report-summary.json", summary)
