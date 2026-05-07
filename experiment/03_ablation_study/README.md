# Experiment 3: Ablation Study

This experiment tests which pieces of the posted-price mechanism matter for
economic performance, correction burden, and physical feasibility after
security postprocess.

## Run

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP
python experiment\03_ablation_study\run_ablation_study.py --samples 24
```

Fast smoke test:

```powershell
python experiment\03_ablation_study\run_ablation_study.py --samples 2 --checkpoint model_best_constr.pth --skip-selector-comparison
```

## Ablations

- `A0 full_peer_context`: full learned posted-price rule.
- `A1 public_context_only`: instantiate the learned rule without peer-bid context.
- `A2 zero_peer_bid_component`: keep the architecture but zero `rho_peer_bid`.
- `A3 base_type_only`: keep only base and type/context components.
- `A4 no_security_component`: remove `rho_security_main` and `rho_security_residual`.
- `A5 no_scarcity_component`: remove `rho_scarcity_main` and `rho_scarcity_residual`.
- `A6 no_residual_heads`: remove only the residual dual-guided heads.
- `S selector_full_*`: optional full-model comparisons across checkpoint
  selectors such as `model_best.pth`, `model_best_feasible_rent.pth`, and
  `model_best_correction.pth` when those files exist.

## Outputs

- `results/ablation_detailed.csv`: all pre/post rows and diagnostics.
- `results/ablation_table.csv`: paper-facing postprocess rows.
- `results/ablation_table.md`: Markdown version of the table.
- `results/ablation_config.json`: run settings and ablation definitions.

## Interpretation

Use postprocess rows for the main ablation table. The main question is whether
removing each component increases `info_rent`, `procurement_cost`,
`mt_offer_gap_mwh`, `positive_adjustment_mwh`, or `dispatch_l1_gap_mwh`.

This script performs evaluation-time component removal. If later you train
dedicated retrained ablation checkpoints, pass those through
`--extra-checkpoints` or extend the selector comparison block.
