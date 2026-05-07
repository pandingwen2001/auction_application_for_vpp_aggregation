# Experiment 4: Strategic Behavior Stress Test

This experiment evaluates unilateral strategic behavior. For each DER, the
script changes only that DER's reported cost parameters while utility is always
computed using the true type.

## Run

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP
python experiment\04_strategic_behavior\run_strategic_behavior.py --samples 12
```

Fast smoke test:

```powershell
python experiment\04_strategic_behavior\run_strategic_behavior.py --samples 2 --overreport-scales 1.25 --underreport-scales 0.8
```

## Compared Methods

- `learned_peer_posted_price`: proposed own-bid-excluded peer-context posted price.
- `learned_public_only_posted_price`: public-context-only posted price baseline.
- `bid_dependent_opf_pay_as_bid`: bid-dependent OPF with pay-as-bid settlement.
- `bid_dependent_opf_uniform_da`: bid-dependent OPF with uniform day-ahead-price settlement.

## Strategy Candidates

- `cost_overreport`: multiply both quadratic and linear cost parameters.
- `linear_cost_overreport`: multiply only the linear term.
- `cost_underreport`: reduce both cost parameters.
- `high_cost_withholding_proxy`: set the DER's bid to its upper bound. In the
  current model, physical availability is public, so this is a cost-bid proxy
  for economic capacity withholding.
- `low_cost_quantity_pressure`: set the DER's bid to its lower bound.

## Outputs

- `results/strategic_behavior_detailed.csv`: per-method, per-DER, per-strategy,
  per-stage utility gains and system deltas.
- `results/strategic_behavior_summary.csv`: aggregate by method, strategy, and stage.
- `results/best_response_by_der.csv`: best strategy among the candidate set for
  each DER.
- `results/best_response_summary.csv`: headline regret statistics by method and stage.
- `results/best_response_summary.md`: Markdown version of the headline best-response summary.
- `results/strategic_behavior_config.json`: run settings and strategy grid.

## Key Diagnostics

- `regret_mean`: mean positive utility gain from the candidate misreport.
- `own_rho_delta_max`: should be near zero for the proposed posted-price method.
- `other_rho_delta_mean`: may be nonzero when peer-bid context is enabled.
- `procurement_delta`, `info_rent_delta`, `positive_adjustment_delta`: system
  consequences of the unilateral deviation.

Important wording for the paper: this is evidence of reduced individual price
manipulation channels, not a formal proof of full strategy-proofness.
