# Experiment 4: ERCOT Strategic Behavior

This is the main strategic-behavior experiment. For each DER, the script
changes only that DER's reported cost parameters while utility is always
computed using the true type.

Two attack classes are reported:

- `fixed`: cost inflation, cost shading, and upper/lower-bound bid proxies.
- `optimal_spsa`: projected black-box GD/SPSA best response over sampled bid
  reports, using 25-50 iterations.

## Run

```powershell
cd C:\tcxp\2026_spring\auction_application_for_vpp_aggregation
python experiment\04_strategic_behavior\run_strategic_behavior.py --scenario-idx 0 --samples 6 --gd-steps 30 --checkpoint model_best.pth --out-dir experiment\04_strategic_behavior\results_ercot_paper
```

Faster smoke test:

```powershell
python experiment\04_strategic_behavior\run_strategic_behavior.py --scenario-idx 0 --samples 2 --gd-steps 3 --methods ours_posted_price,bid_dependent_opf_pay_as_bid,bid_dependent_opf_uniform_da --spsa-methods ours_posted_price --out-dir experiment\04_strategic_behavior\results_smoke
```

To run a broader ERCOT subset:

```powershell
python experiment\04_strategic_behavior\run_strategic_behavior.py --all-ercot-scenarios --max-scenarios 6 --samples 4 --gd-steps 25 --checkpoint model_best.pth --out-dir experiment\04_strategic_behavior\results_ercot_subset
```

## Compared Methods

The fixed-strategy grid uses the same method set as the main overall
performance experiment:

- `ours_posted_price`
- `vcg_disaggregation`
- `shapley_value_disaggregation`
- `nucleolus_disaggregation`
- `dlmp_settlement`
- `bid_dependent_opf_pay_as_bid`
- `bid_dependent_opf_uniform_da`
- `constrained_social_opt`

By default, projected SPSA is run for `ours_posted_price`, `dlmp_settlement`,
`bid_dependent_opf_pay_as_bid`, and `bid_dependent_opf_uniform_da`. Add
`--include-cooperative-spsa` to also run it for VCG/Shapley/Nucleolus; this is
much slower because coalition values must be recomputed repeatedly.

## Outputs

- `strategic_behavior_detailed.csv`: per-method, per-DER, per-strategy utility
  gains and system deltas.
- `best_response_by_der.csv`: best attack for each DER.
- `best_response_summary.csv`: headline regret statistics by method and attack
  type.
- `best_response_summary.md`: Markdown version of the headline table.
- `strategic_behavior_config.json`: run settings and attack grid.

## Key Diagnostics

- `regret_mean_across_der`: average positive utility gain from misreporting.
- `regret_max_der`: worst DER-level mean regret.
- `regret_max_sample`: worst single-sample utility gain.
- `own_rho_delta_max`: should be near zero for the proposed own-bid-excluded
  posted-price mechanism.
- `procurement_delta_at_worst` and `info_rent_delta_at_worst`: system impact
  at the worst strategic deviation.
