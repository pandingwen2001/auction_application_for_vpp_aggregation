# Experiment 3: ERCOT Ablation Study

This experiment tests which pieces of the learned posted-price rule matter on
ERCOT typical-day scenarios. The main table uses postprocessed rows only, plus
the constrained social optimum reference.

## Run

```powershell
cd C:\tcxp\2026_spring\auction_application_for_vpp_aggregation
python experiment\03_ablation_study\run_ablation_study.py --all-ercot-scenarios --samples 24 --checkpoint model_best.pth --out-dir experiment\03_ablation_study\results_ercot_paper
```

Fast smoke test:

```powershell
python experiment\03_ablation_study\run_ablation_study.py --samples 2 --scenario-idx 0 --checkpoint model_best.pth --out-dir experiment\03_ablation_study\results_smoke
```

## Ablations

- `A0 full_peer_context`: full learned posted-price rule.
- `A1 public_context_only`: instantiate the learned rule without peer-bid
  context.
- `A2 zero_peer_bid_component`: keep the architecture but zero `rho_peer_bid`.
- `A3 no_security_component`: zero `rho_security`.
- `A4 no_scarcity_component`: zero `rho_scarcity`.
- `A5 base_type_only`: keep only base and type/context components.

Optional checkpoint-selector comparisons can be added with
`--include-selector-comparison`, but they are not part of the default main
ablation table.

## Outputs

- `ablation_detailed.csv`: all pre/post rows and diagnostics for every ERCOT
  scenario.
- `ablation_table.csv`: aggregated paper-facing postprocess rows.
- `ablation_table.md`: Markdown version of the table.
- `ablation_config.json`: run settings and ablation definitions.

## Interpretation

Use the postprocess rows for the main table. Key metrics are
`total_procurement_cost`, `operation_cost_gap_pct`, `info_rent_cost`,
`positive_adjustment_mwh`, `dispatch_l1_gap_mwh`, and
`utility_shortfall_cost`.
