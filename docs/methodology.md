# ModelAudit: methodology

The healthy fixture is a chronological split of a synthetic binary task with legitimate predictors. The flawed fixture adds the future label as a predictor and uses a random split despite a future-deployment protocol. Logistic regression can achieve near-perfect AUC in the flawed fixture, but metadata-driven leakage checks reject it. Supplied scores are evaluated without fitting or modifying the submitted model. Thresholds are explicit illustrative review policies, not externally mandated standards.

## Assumptions and limits

Metadata must be accurate; static checks cannot discover all hidden leakage or certify independence. High AUC and duplicate feature vectors are warnings, not proof of leakage. Temporal overlap is a failure for this future-deployment contract; other intended-use contracts may require different splits. Subgroup gaps are descriptive diagnostics, not legal findings or proof of unfairness. This prototype does not validate calibration uncertainty, labels, causal claims, security or regulatory compliance. PASS_WITHIN_SCOPE never means universally safe.

## Extension plan

Add signed provenance, independent reruns, calibration confidence intervals, entity/time split contracts and reviewer-approved policy configuration.
