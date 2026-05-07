# Experiment 6: Robustness Under Load Uncertainty

This experiment evaluates whether the mechanism remains physically feasible
and economically stable under deterministic load-scale stress and random
hourly load-profile uncertainty.

For each perturbed load scenario, the script rebuilds the 33-bus line-flow and
voltage margins before evaluating the trained posted-price mechanism and
baselines.

## Run

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP
python experiment\06_robustness\run_robustness.py --samples 12
```

Fast smoke test:

```powershell
python experiment\06_robustness\run_robustness.py --samples 2 --scale-levels 0.9 1.1 --random-scenarios-per-band 0 --skip-bid-opf-baseline --skip-social-optimum --fixed-price-ratios 0.7
```

Paper-strength run:

```powershell
python experiment\06_robustness\run_robustness.py --samples 24 --eval-seeds 20260426 20260427 20260428 --scale-levels 0.8 0.9 1.0 1.1 1.2 --random-scenarios-per-band 5
```

## Scenario Types

- `scale_sweep`: deterministic uniform load scaling, useful for plotting a
  clean stress curve.
- `random_mild`: hourly multipliers sampled from `[0.95, 1.05]`.
- `random_medium`: hourly multipliers sampled from `[0.90, 1.10]`.
- `random_high`: hourly multipliers sampled from `[0.80, 1.20]`.

Random hourly profiles are lightly smoothed by default. Use
`--disable-smoothing` to keep raw hourly samples.

## Compared Methods

- `learned_peer_<checkpoint>`: proposed peer-context posted-price mechanism.
- `learned_public_only_<checkpoint>`: public-context-only posted-price baseline.
- `fixed_price_ratio_*`: simple fixed posted-price baseline.
- `bid_dependent_opf_pay_as_bid`: bid-dependent OPF with pay-as-bid settlement.
- `bid_dependent_opf_uniform_da`: bid-dependent OPF with uniform day-ahead-price settlement.
- `constrained_social_opt`: oracle efficiency lower bound.

## Outputs

- `results/robustness_detailed.csv`: per-method, per-stage, per-load-scenario rows.
- `results/robustness_summary.csv`: mean/std robustness summary by method, stage, and scenario group.
- `results/robustness_summary.md`: Markdown version of the summary table.
- `results/load_scale_curve.csv`: compact deterministic load-scale curve for plotting.
- `results/robustness_config.json`: run settings and exact load multipliers.

## Key Diagnostics

- `feasible_rate_mean`: fraction of sampled type scenarios satisfying physical checks.
- `procurement_cost_mean/std`: economic stability under load uncertainty.
- `info_rent_mean/std`: rent stability under load uncertainty.
- `mt_floor_gap_mwh_mean/max`: MT-floor robustness.
- `postprocess_mt_slack_mwh_mean`: whether postprocess needs slack.
- `positive_adjustment_mwh_mean/std`: correction burden.
- `line_violation_max_mw` and `voltage_violation_max_pu`: network security residuals.

Important wording for the paper: this experiment supports robustness under
load stress and forecast-like hourly perturbations. It is not a proof of
distribution-free robustness.
