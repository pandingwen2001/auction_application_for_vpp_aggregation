# Posted-Price VPP

Self-contained code for the current 33-bus VPP aggregation/disaggregation
posted-price mechanism.

The project focuses on a bid-aware but own-bid-excluded posted-price rule for
VPP procurement:

```text
public 24h system context + leave-one-out peer bid aggregates
  -> posted price rho_i,t
  -> DER offer cap under its reported cost curve
  -> preliminary dispatch
  -> security postprocess
  -> settlement at the same posted price
```

For DER `i`, its own report is excluded from the features used to compute
`rho_i,t`. Other DER prices may still use DER `i`'s bid through peer aggregate
market-state features.

## Project Layout

```text
data/                         Liu 2024 24h load, price, ESS, PV/WT profiles
network/vpp_network.py        Local single-period IEEE 33-bus network builder
network/vpp_network_multi.py  24h 33-bus network extension
network/opf_layer_multi.py    Approximate dispatch / OPF layer
our_method/                   Posted-price mechanism, training, evaluation
baseline/                     Constrained social optimum baseline
runs/                         Archived reference runs copied from original_with_bid
```

## Main Commands

Run from this folder:

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP

python our_method\check_peer_bid_exclusion.py
python our_method\run_phase1a.py
python our_method\run_phase1a_transformer.py
python our_method\evaluate_posted_price.py --run runs\phase1a_20260429_110448
```

`run_phase1a.py` keeps the original MLP posted-price heads. The Transformer
ablation in `run_phase1a_transformer.py` uses attention only over public
context/type tokens; leave-one-out peer-bid features remain pointwise to
preserve own-bid exclusion.

Evaluate a Transformer run with the matching architecture flag:

```powershell
python our_method\evaluate_posted_price.py --run runs\<phase1a_transformer_run> --price-arch transformer
```

Optional correction-feedback and dual-guided fine-tuning:

```powershell
python our_method\generate_correction_feedback.py --run runs\phase1a_20260429_110448 --checkpoint model_best_constr.pth
python our_method\run_dual_guided.py --run runs\phase1a_20260429_110448 --checkpoint model_best_constr.pth
```

## Notes

- The code now uses the local `network/vpp_network.py::build_33bus_network`.
  It no longer depends on `auction_VPP\bounded_code`.
- `include_dr=False` is still the default for the current paper experiments,
  giving 12 DERs: 4 PV, 4 WT, and 4 MT.
- Security postprocess settlement keeps the same posted price `rho`; no
  bid-dependent uplift price is introduced.
