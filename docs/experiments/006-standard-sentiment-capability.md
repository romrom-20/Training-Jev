# Experiment 006: capability check with standard sentiment labels

**Prospective local diagnostic, 23 September 2026.** Diagnostic 005 failed when
the small targets had to remap positive and negative judgments to arbitrary A/B
codes. This experiment removes that code remapping and uses the conventional
`positive` and `negative` labels. It tests task viability only.

Use the same 16 plainly positive/negative reviews frozen in
`scripts/capability_diagnostic.py`, balanced eight per class. Cross each review
with four fixed user-message templates, giving 64 prompts per model. Each prompt
asks for exactly one of the standard labels. No demonstrations or prompt tuning.
Generate greedily for at most four tokens. Report strict first-word accuracy,
conditional next-token accuracy over positive/negative, and the two-label
probability mass per format and pooled.

Continuation requires at least 90% strict accuracy in each of four formats for a
model. Repeated texts mean formats are paired prompt conditions, not independent
examples. Passing selects a viable behavior task for a new probe protocol; it does
not test the probe hypothesis. Failure ends this task candidate without wording
changes. Explicitly report that these lexical reviews are trivially available to
text-only baselines.
