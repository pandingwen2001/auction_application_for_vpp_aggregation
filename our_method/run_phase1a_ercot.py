#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phase1a_ercot.py
--------------------
Phase 1a training over 24 ERCOT 2023 typical days.

Differences from `run_phase1a.py`:
  * Builds 24 net_multi dicts (one per typical day from data/ercot_2023_typical.npz)
  * Trainer samples one scenario uniformly at random each iteration and
    hot-swaps the mechanism's data buffers before forward/backward.
  * All scenarios share the same learnable parameters and the same DER prior.

Usage:
    cd auction_application_for_vpp_aggregation
    python our_method/run_phase1a_ercot.py
"""

import os
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "network"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "data"))
sys.path.insert(0, _THIS_DIR)

from vpp_network_multi import build_network_multi          # noqa: E402
from vpp_mechanism_multi import VPPMechanismMulti          # noqa: E402
from trainer_multi import VPPTrainerMulti, DERTypePriorMulti  # noqa: E402
from ercot_profiles import num_scenarios                    # noqa: E402


def main():
    print(f"Building {num_scenarios()} ERCOT scenario networks...")
    scenarios = [build_network_multi(scenario_idx=i,
                                     ctrl_min_ratio=0.15,
                                     pi_clip_factor=3.0)
                 for i in range(num_scenarios())]
    print(f"  Example dates: {scenarios[0]['scenario_date']}, "
          f"{scenarios[12]['scenario_date']}, {scenarios[23]['scenario_date']}")

    print("Constructing mechanism on scenario 0...")
    # Mechanism is built once on scenario 0; trainer will call
    # mech.set_scenario(...) at each iter to swap data buffers.
    mech = VPPMechanismMulti(
        scenarios[0],
        posted_price_cfg=dict(
            pi_buyback_ratio=0.1,
            use_peer_bid_context=True,
            peer_bid_scale=0.25,
            type_cap_ratio=dict(
                PV=0.70,
                WT=0.70,
                DG=0.80,
                MT=1.70,
                DR=0.80,
            ),
        ),
    )
    prior = DERTypePriorMulti(scenarios[0])

    cfg = dict(
        # ---- training length ----
        max_iter         = 5000,
        batch_size       = 64,
        num_batches      = 200,
        warmup_iters     = 500,
        tau_ramp         = 800,
        # ---- optim ----
        lr               = 1e-3,
        grad_clip_norm   = 1.0,
        gd_iter          = 20,
        mt_offer_penalty_w = 100.0,
        pi_buyback_ratio = 0.1,
        # ---- logging ----
        print_iter       = 50,
        log_every        = 25,
        save_iter        = 500,
        # ---- wandb ----
        use_wandb        = False,
        wandb_project    = "vpp-multi-period-ercot",
        wandb_tags       = ["phase1a", "ercot-2023", "24-typical-days",
                            "houston-hub", "posted-price",
                            "own-bid-excluded", "peer-bid-context"],
        wandb_mode       = "online",
    )

    trainer = VPPTrainerMulti(
        mechanism=mech,
        prior=prior,
        cfg=cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
        scenarios=scenarios,
    )
    trainer.train()


if __name__ == "__main__":
    main()
