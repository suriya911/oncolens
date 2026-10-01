# Experiment Registry
Rules: register an experiment BEFORE running it. Fix the metric and the comparison in advance.
Primary metric: macro AUROC on validation (model selection) and test (final, once).
Seeds: 0, 1, 2 for any result reported in REPORT.md.

| ID | RQ | Dataset | Model | Init | Strategy | GPUs | Precision | Global batch | Epochs | Seeds | Status | W&B group | Result |
|----|----|---------|-------|------|----------|------|-----------|--------------|--------|-------|--------|-----------|--------|
| e001 | - | pcam | M0 SimpleCNN | scratch | single | 4070 | bf16 | 256 | 3 | 0 | planned | sanity | |
| e002 | RQ1 | pcam | M1 ResNet-18 | scratch | single | 4070 | bf16 | 256 | 30 | 0,1,2 | planned | rq1-pcam | |
| e010 | RQ1/RQ3 | pcam | M1 ResNet-18 | scratch | ddp | 2×T4 | fp16 | 512 | 30 | 0,1 | planned | rq3-scaling | |
