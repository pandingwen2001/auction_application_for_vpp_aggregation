# Experiment 2: ERCOT Pre/Post Feasibility

This experiment compares the learned posted-price mechanism against the
constrained social optimum on ERCOT typical-day scenarios.

The default rows are intentionally narrow:

- `social_optimum`: full-information feasible reference.
- `ours_*:pre`: learned posted-price dispatch before security postprocess.
- `ours_*:post`: learned posted-price dispatch after security postprocess.

## Run

```powershell
cd C:\tcxp\2026_spring\auction_application_for_vpp_aggregation
python experiment\02_pre_post_feasibility\run_pre_post_feasibility.py --all-ercot-scenarios --samples 24 --checkpoints model_best.pth --out-dir experiment\02_pre_post_feasibility\results_ercot_paper
```

Fast smoke test:

```powershell
python experiment\02_pre_post_feasibility\run_pre_post_feasibility.py --samples 2 --scenario-idx 0 --checkpoints model_best.pth --out-dir experiment\02_pre_post_feasibility\results_smoke
```

## Outputs

- `pre_post_feasibility_detailed.csv`: one row per ERCOT scenario and method
  stage.
- `pre_post_feasibility_summary.csv`: aggregated paper-facing comparison.
- `pre_post_feasibility_summary.md`: Markdown version of the summary table.
- `pre_post_feasibility_by_time.csv`: hourly feasibility and correction
  diagnostics for plotting.
- `pre_post_feasibility_config.json`: run settings and checkpoint source.

## Key Metrics

- `feasible_rate_pct`: share of ERCOT scenario rows that satisfy the physical
  feasibility checks under the configured tolerance. The default tolerance is
  `1e-3`, and the postprocess uses `mt_slack_weight=1e7` to make this a safety
  verification experiment.
- `operation_cost_gap_pct`: true operation-cost gap relative to the constrained
  social optimum under the same ERCOT scenario and sampled types.
- `dispatch_l1_gap_mwh`: dispatch distance from the social optimum.
- `mt_floor_gap_mwh`, `line_violation_max_mw`, `voltage_violation_max_pu`, and
  `power_balance_residual_max_mw`: physical feasibility diagnostics.
- `correction_l1_mwh`, `positive_adjustment_mwh`, and
  `mt_security_uplift_mwh`: postprocess intervention burden.
