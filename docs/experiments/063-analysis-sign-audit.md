# Experiment 063 analysis sign audit

After the registered run completed, an algebra check found that the expected-sign sentence in `063-output-key-order-restaurant-replication.md` was reversed. The frozen estimator itself is unchanged.

Experiment 063 defines a context gain as `RMSE(own review) - RMSE(donor review)`, its decoder interaction as finite-grid gain minus free-greedy gain, and its topic-match effect as same-category interaction minus cross-category interaction. Experiment 062 defined its condition interaction from donor-review error minus own-review error and reported cross-category minus same-category. Negating the component interaction and reversing the condition contrast cancel; therefore the same numerical order-moderation sign is expected across the two experiments. The expected primary sign from the 062 coordinate audit is **negative**, not positive as the protocol's final explanatory sentence stated.

The preregistered primary estimate remains -0.204 VA-RMSE points (95% recipient-bootstrap interval [-0.487, +0.067]); its interval includes zero. The coordinate-specific secondary order moderations are -0.419 for valence and -0.623 for arousal, both in the direction observed in 062, while their difference is uncertain. The secondary aggregate VA moderation is -0.528 (95% interval [-0.808, -0.254]). This correction records the formula-level sign audit and does not alter the sample, outputs, bootstrap, endpoints, or any estimate. The aggregate and coordinate outcomes remain secondary where the frozen protocol says so.

The corrective interpretation was written after seeing the outcome. It should be treated as a transparent reporting correction, not as a retroactive preregistration amendment.
