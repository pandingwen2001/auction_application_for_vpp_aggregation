# Own-Bid-Excluded Bid-Aware Posted Price

This folder is the self-contained 33-bus posted-price VPP project extracted
from the earlier `auction_VPP/original_with_bid` branch.

The main change is in `our_method/vpp_mechanism_multi.py`: the posted-price
network can now use leave-one-out peer-bid aggregates.

For DER `i`, the mechanism computes

```text
rho_i,t = f_theta(public_context_t,i, DER_class_i, aggregate_peer_bids_-i)
offer_cap_i,t = argmax_y rho_i,t y - bid_cost_i(y)
payment_i = sum_t rho_i,t x_i,t
```

So `bid_i` affects DER `i`'s accepted quantity through `offer_cap_i,t`, but it
does not enter `rho_i,t`. Other DER prices may use `bid_i` through aggregate
market-state features.

Useful commands:

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP
python our_method\check_peer_bid_exclusion.py
python our_method\run_phase1a.py
python our_method\evaluate_posted_price.py --run runs\<run_name>
```

The new mechanism is enabled by default in `run_phase1a.py`,
`evaluate_posted_price.py`, `generate_correction_feedback.py`, and
`run_dual_guided.py`. Use `--disable-peer-bid-context` in the evaluation and
dual-guided scripts to instantiate the old public-context-only price rule.
