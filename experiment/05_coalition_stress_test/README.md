# Experiment 5: Coalition and Collusion Stress Test

This experiment evaluates coordinated strategic behavior by groups of DERs.
The current 33-bus implementation treats physical availability as public, so
coalition behavior is represented through coordinated cost-bid manipulation and
economic withholding proxies.

## Run

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP
python experiment\05_coalition_stress_test\run_coalition_stress_test.py --samples 12
```

Fast smoke test:

```powershell
python experiment\05_coalition_stress_test\run_coalition_stress_test.py --samples 2 --coalition-groups MT --max-coalition-size 2 --max-combinations-per-bucket 3 --overreport-scales 1.25 --underreport-scales 0.8
```

More exhaustive paper run:

```powershell
python experiment\05_coalition_stress_test\run_coalition_stress_test.py --samples 24 --coalition-groups MT renewable PV WT --max-coalition-size 4 --max-combinations-per-bucket 1000 --overreport-scales 1.1 1.25 1.5 --underreport-scales 0.8
```

## Compared Methods

- `learned_peer_posted_price`: proposed own-bid-excluded peer-context posted price.
- `learned_public_only_posted_price`: public-context-only posted price baseline.
- `bid_dependent_opf_pay_as_bid`: bid-dependent OPF with pay-as-bid settlement.
- `bid_dependent_opf_uniform_da`: bid-dependent OPF with uniform day-ahead-price settlement.

## Coalition Groups

Default groups:

- `MT`: controllable MT coalition.
- `renewable`: PV and WT coalition.

Optional groups:

- `PV`
- `WT`
- `controllable`
- `all`

Use `--max-coalition-size` and `--max-combinations-per-bucket` to control the
coalition size sweep and enumeration cost.

## Strategy Candidates

- `group_cost_overreport`: all members multiply both cost parameters.
- `group_linear_cost_overreport`: all members multiply only the linear term.
- `group_cost_underreport`: all members reduce both cost parameters.
- `group_high_cost_withholding_proxy`: all members report the upper bid bound.
- `group_low_cost_pressure`: all members report the lower bid bound.

## Outputs

- `results/coalition_stress_detailed.csv`: per-method, per-coalition, per-strategy, per-stage results.
- `results/coalition_stress_summary.csv`: aggregate by method, coalition group, size, strategy, and stage.
- `results/worst_coalition_by_size.csv`: worst coalition/strategy for each method, group, size, and stage.
- `results/coalition_size_curve.csv`: compact table for plotting coalition size sensitivity.
- `results/worst_coalition_summary.csv`: headline worst-case table by method and stage.
  `n_size_buckets` counts coalition group-size buckets after taking the worst
  coalition in each bucket.
- `results/worst_coalition_summary.md`: Markdown version of the headline table.
- `results/coalition_stress_config.json`: run settings, coalition set, and strategy grid.

## Key Diagnostics

- `coalition_regret_mean`: positive gain of the coalition's aggregate utility.
- `per_member_regret_mean`: coalition regret divided by coalition size.
- `procurement_delta`: VPP cost impact of the collusive deviation.
- `info_rent_delta`: total information rent impact.
- `offer_cap_delta_mwh`: change in accepted posted-price offer capability.
- `coalition_rho_delta_mean`: price movement faced by coalition members.
- `outside_rho_delta_mean`: price spillover to nonmembers.

Important wording for the paper: this is a collusion stress test, not a formal
collusion-proofness guarantee.
