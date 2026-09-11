# EWC sanity check — DEBUG ONLY

All functional checks passed using the existing quick seed 0, unchanged lambda=100.0 and learning rate=0.01. No tuning or full benchmark was performed. The original final accuracy **19.60%** and all checkpoint predictions were reproduced exactly; the existing benchmark results were not replaced.

| Check | Evidence |
|---|---|
| Nonzero Fisher | Task 1: 99,052/203,530 entries nonzero; all tasks finite and nonnegative. Per-layer statistics are in the JSON. |
| Penalty activates | Zero at the exact Task 1 optimum, correctly; before Task 2 sample 2 the loss is 0.00032210966 and gradient norm is 0.010833165. |
| Correct snapshots | Each snapshot equals the task-end parameters, shares no parameter memory, and stays unchanged through subsequent training. Four snapshots are retained. |
| Lambda is used | Actual model lambda=100.0, matching saved configuration; finite and positive. No candidate values were tested. |
| Penalty reaches SGD | All 500 actual gradient/update checks passed; 399 updates differ from a CE-only update at the same parameters in actual fp32 storage. |
| Relative strength | Across tasks 2–5, median penalty/CE gradient norm = 0.0805546; maximum = 9.10372. This is a global norm, not a per-parameter strength estimate. |

Lambda alone has no universal useful scale: its effect also depends on Fisher magnitude, loss normalization and parameter displacement. The effective diagonal factor lr × lambda × accumulated Fisher is recorded per task, and no non-finite gradients or parameters occurred. These checks establish that regularization is active and correctly applied, not that lambda=100 is optimal or that EWC should perform well on this Class-IL task. Keep the bad quick result.

Reproduce: `python experiments/check_ewc_sanity.py results/research_benchmark/quick`.
Detailed evidence: [ewc_sanity.json](ewc_sanity.json).
