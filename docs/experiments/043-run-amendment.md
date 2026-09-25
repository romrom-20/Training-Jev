# Experiment 043 execution amendment

The frozen run completed all 1,302 jobs with zero invalid outputs on MPS. The
runner's progress message incorrectly printed a cumulative denominator of 1,440,
left over from Experiment 041. The actual job list, sample, generation calls,
private output rows, and run manifest all contained exactly 1,302 jobs. The
display-only denominator is corrected in the post-run runner version; it does not
change the recorded run or its analysis. The code revision used for generation is
`74ab7db8bd4d9bd2a43ab4f98ca57e20ce02a115`.

The total generation time was 2,608.1 seconds after the model loaded. Model loading
time is not included. No item outputs were dropped, retried, or reinterpreted.
