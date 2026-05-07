# Experiment 2: Pre/Post Feasibility

This experiment isolates the physical feasibility role of the security
postprocess.

For each selected method, it compares:

- `pre`: preliminary posted-price dispatch before CVXPY security correction.
- `post / mt_uplift_enabled`: security postprocess where MT can be increased
  up to physical availability for security correction.
- `post / no_mt_uplift`: security postprocess capped by accepted offers for
  all DERs, used to test whether MT security uplift is necessary.

## Run

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP
python experiment\02_pre_post_feasibility\run_pre_post_feasibility.py --samples 24
```

Fast smoke test:

```powershell
python experiment\02_pre_post_feasibility\run_pre_post_feasibility.py --samples 2 --checkpoints model_best_constr.pth
```

## Outputs

- `results/pre_post_feasibility_summary.csv`: method-level feasibility and
  correction metrics.
- `results/pre_post_feasibility_summary.md`: Markdown version of the table.
- `results/pre_post_feasibility_by_time.csv`: hourly MT floor, dispatch,
  correction, line/voltage violation, and balance diagnostics for plotting.
- `results/pre_post_feasibility_config.json`: run settings.

## Key Metrics

- `mt_floor_gap_mwh`: physical MT floor violation after dispatch.
- `postprocess_mt_slack_mwh`: slack used by postprocess to satisfy the MT
  floor constraint; should be near zero for feasible postprocess rows.
- `line_violation_max_mw` and `voltage_violation_max_pu`: maximum security
  violations including ESS injections.
- `positive_adjustment_mwh`: upward DER correction required by postprocess.
- `mt_security_uplift_mwh`: MT dispatch above accepted offer cap; nonzero only
  when MT security uplift is used.
