# Experiment 005: find a behavior task the laptop targets can actually do

**Prospective local diagnostic, 23 September 2026.** Experiments 002–004 found
that the synthetic rule-following task failed the target capability gate. Before
training another readout or changing task prompts by trial and error, this frozen
diagnostic tests whether these same models can do a familiar binary judgment when
the task appears in the user message.

## Question and fixed design

Can Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct classify plainly positive versus
negative short reviews, across four fixed prompt phrasings and both A/B mappings?
This is task selection, not evidence about probes or a novelty test.

Use 16 fixed, short review texts (eight clearly positive and eight clearly
negative), each under four predetermined phrasings and both answer mappings: 128
prompts per model. Keep the review text and label identical across mappings. The
instruction appears in the user role; system content is empty. Generate greedily
for at most four new tokens. Do not add demonstrations, tune wording, exclude
examples, or select a prompt phrasing after seeing results.

Report strict generated-answer accuracy (first non-space output token must equal
the correct A/B letter), conditional A-vs-B next-token accuracy, and A+B mass,
separately for each prompt phrasing and code mapping, plus the pooled values. The
prospective continuation gate is at least 90% strict accuracy in every one of the
eight phrasing × mapping cells for a model. A pass licenses a separate, fully
specified probe experiment on sentiment; it does not license a general claim. A
failure ends this candidate without prompt repair. Any different task is a new
protocol and a new diagnostic.

This benchmark is lexical and deliberately easy. A text-only classifier should
solve it; it can only establish a viable target behavior task, not the value of
activation probes. No LessWrong post decision follows from this diagnostic alone.
