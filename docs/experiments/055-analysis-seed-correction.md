# Experiment 055 — bootstrap seed correction

After the 1,302 generations completed, a validation audit found that the 055
analyzer wrapper inherited Experiment 054's bootstrap seed (`20260954`) even though
the frozen 055 protocol specifies `20260955`. The point estimates and model outputs
are unaffected. The analyzer now sets the registered seed explicitly and the
summary intervals have been regenerated. The frozen protocol and its hash remain
unchanged; this correction records an analysis-code oversight rather than a change
to the registered estimand.
