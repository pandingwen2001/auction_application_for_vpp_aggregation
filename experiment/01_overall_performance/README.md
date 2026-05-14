# Experiment 1: Overall Performance

This experiment builds the headline paper table. It evaluates all methods on
the same sampled DER type scenarios and reports feasibility, economic cost,
information rent, and dispatch distance to the constrained social optimum.

## Run

```powershell
cd C:\tcxp\2026_spring\auction_application_for_vpp_aggregation
python experiment\01_overall_performance\run_overall_performance.py --samples 24 --checkpoints model_best.pth
```

ERCOT paper-grade typical-day evaluation:

```powershell
python experiment\01_overall_performance\run_overall_performance.py --all-ercot-scenarios --samples 24 --checkpoints model_best.pth --skip-public-context-ablation --adjustment-weight 1000 --out-dir experiment\01_overall_performance\results_ercot_paper_v2
python experiment\01_overall_performance\plot_overall_performance.py --table experiment\01_overall_performance\results_ercot_paper_v2\overall_performance_table.csv
```

Useful faster smoke test:

```powershell
python experiment\01_overall_performance\run_overall_performance.py --samples 2 --scenario-idx 0 --checkpoints model_best.pth --skip-public-context-ablation --adjustment-weight 1000 --out-dir experiment\01_overall_performance\results_smoke
```

## Outputs

- `results/overall_performance_detailed.csv`: all pre/post rows and all diagnostics.
- `results/overall_performance_table.csv`: headline rows for the paper table.
- `results/overall_performance_table.md`: Markdown version of the same table.
- `results/overall_performance_main_figure.pdf/png`: three-panel paper figure
  from the headline table.
- `results/overall_performance_config.json`: run settings and selected checkpoint directory.

## Baseline Groups

- `oracle`: constrained social optimum with true types; efficiency lower bound.
- `cooperative_disaggregation`: efficient dispatch with cooperative surplus
  settlement by VCG marginal contribution, Shapley value, or nucleolus.
- `dlmp`: efficient dispatch settled by an active-power DLMP approximation,
  `lambda_i,t = pi_DA_t * (1 + LF_i)`, following the Liu-style DLMP loss-factor
  component available in the local network model.
- `bid_dependent_opf`: truthful bid-dependent OPF dispatch under pay-as-bid
  and uniform day-ahead-price settlement.
- `fixed_price`: optional non-learned posted prices `rho_i,t = ratio * pi_DA_t`
  when `--include-fixed-price` is provided. These are kept for strategic
  sensitivity experiments and are omitted from the main paper table by default.
- `learned_posted_price`: learned posted-price checkpoints, evaluated both
  with own-bid-excluded peer context and public-context-only ablation.

The cooperative baselines compute coalition values from a load-capped avoided
grid-cost game, then pay each DER its realized dispatch cost plus allocated
surplus. Physical feasibility metrics still come from the common
network-constrained dispatch used in the table.

## Headline Metrics

- `operation_cost`: true DER production cost plus grid procurement cost.
- `info_rent_cost`: DER payments minus true DER production cost.
- `total_procurement_cost`: operation cost plus information rent.
- `operation_cost_gap_pct`: efficiency loss relative to the constrained social
  optimum.
- `dispatch_l1_gap_mwh`: dispatch distance from the constrained social optimum.
- `renewable_share_pct` and `grid_import_mwh`: clean-resource use and grid
  reliance.
- `utility_min`, `utility_shortfall_cost`, and `feasible_rate_pct`: participation
  and physical feasibility safety checks.
