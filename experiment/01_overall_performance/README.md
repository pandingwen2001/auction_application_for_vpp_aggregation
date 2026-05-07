# Experiment 1: Overall Performance

This experiment builds the headline paper table. It evaluates all methods on
the same sampled DER type scenarios and reports feasibility, economic cost,
information rent, and dispatch distance to the constrained social optimum.

## Run

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP
python experiment\01_overall_performance\run_overall_performance.py --samples 24
```

Useful faster smoke test:

```powershell
python experiment\01_overall_performance\run_overall_performance.py --samples 2 --checkpoints model_best_constr.pth
```

## Outputs

- `results/overall_performance_detailed.csv`: all pre/post rows and all diagnostics.
- `results/overall_performance_table.csv`: headline rows for the paper table.
- `results/overall_performance_table.md`: Markdown version of the same table.
- `results/overall_performance_config.json`: run settings and selected checkpoint directory.

## Baseline Groups

- `oracle`: constrained social optimum with true types; efficiency lower bound.
- `bid_dependent_opf`: truthful bid-dependent OPF dispatch under pay-as-bid
  and uniform day-ahead-price settlement.
- `fixed_price`: non-learned posted prices `rho_i,t = ratio * pi_DA_t`.
- `learned_posted_price`: learned posted-price checkpoints, evaluated both
  with own-bid-excluded peer context and public-context-only ablation.
