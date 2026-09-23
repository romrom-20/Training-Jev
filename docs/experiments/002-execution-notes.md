# Execution notes for the remapping study

The prospective protocol was committed as `186ba51` before outcome collection.
Both models completed capture and probe fitting on 22 September 2026. The 0.5B
intervention run finished. The 1.5B process stopped after 16 of 21 complete conditions;
no process remained on 23 September. Its partial log was preserved, validated for
complete condition coverage, and resumed for the remaining five conditions.

The implementation now refuses to overwrite a partial log silently. Resume checks
that every retained condition contains exactly the selected prompt IDs with no
duplicates. The completed 1.5B manifest records 6,144 retained rows. Its `seconds`
field covers the resumed portion only; total intervention runtime is not known.
The original 0.5B timing covers the complete stage.

Both targets failed the prospective task-competence gate. Their interventions are
therefore descriptive measurements, not evidence that a semantic computation was
causally identified. No failing task was repaired in place and no rows were dropped.

One control needs a narrower label than its shorthand suggests: `output_A` was trained
on the **task-correct A label**, not on the model's observed A/B choice. On a competent
target these largely agree, but here they do not. It must be called an *intended-output
label control*, not an actual answer-writing direction. It cannot establish the
answer-writing interpretation in this run. The recorded condition name is retained
for artifact consistency. A future study should include a model-logit gradient or a
probe trained on observed outputs as the explicit answer-writing control.

The benchmark also changes nuisance-record diversity between training regimes:
fixed-code training uses twice as many record IDs as balanced-code training at the
same label budget. It does not isolate answer counterbalancing from that diversity
change. These limitations, the failed competence gates, and direct answer-remapping
prior art prevent a strong causal or novelty claim from this experiment.
