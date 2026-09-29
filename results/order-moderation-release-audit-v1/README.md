# Post-hoc release audit: output-order moderation across English and Chinese

This audit was added after Experiment 064 and is exploratory. It does not change the registered SIGHAN or model-size outcomes.

The all-VA order moderation estimates were -0.258 for English laptop reviews ([-0.447, -0.060]), -0.528 for English restaurant reviews ([-0.800, -0.252]), and -0.077 for Chinese SIGHAN reviews ([-0.334, 0.172]). The post-hoc SIGHAN-minus-weighted-English difference was 0.311 (95% independent cohort-bootstrap interval [0.003, 0.610]).

The nominal post-hoc interval narrowly excludes zero, but this only suggests that these selected benchmark cohorts differ under one prompt/model setup. Language, domain, dataset release, and cohort remain confounded. The two English sources share the DimABSA release; the Chinese source is a separate public release. Private rows and review text remain in ignored `.context/`; `summary.json` pins aggregate source checksums.
