#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phase1a.py
--------------
Full Phase 1a training entry point with wandb logging enabled.

Usage:
    cd multi_period_code
    python our_method/run_phase1a.py

The run will appear on https://wandb.ai under project "vpp-multi-period".
Make sure you have run `wandb login` at least once before invoking this.
"""

import os
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "network"))
sys.path.insert(0, _THIS_DIR)

from vpp_network_multi import build_network_multi          # noqa: E402
from vpp_mechanism_multi import VPPMechanismMulti          # noqa: E402
from trainer_multi import VPPTrainerMulti, DERTypePriorMulti  # noqa: E402


def main():
    print("Building multi-period network (Liu profiles, time-varying price)...")
    net = build_network_multi(
        constant_price=False,
        ctrl_min_ratio=0.15,
    )

    print("Constructing own-bid-excluded bid-aware posted-price mechanism...")
    mech = VPPMechanismMulti(
        net,
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
    prior = DERTypePriorMulti(net)

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
        # DERs can alternatively sell to the grid at a low buyback price.
        # We use a small ratio because grid buyback is uncertain/discounted.
        pi_buyback_ratio = 0.1,
        # ---- logging ----
        print_iter       = 50,
        log_every        = 25,
        save_iter        = 500,
        # ---- wandb ----
        use_wandb        = False,
        wandb_project    = "vpp-multi-period",
        wandb_tags       = ["phase1a", "liu-profiles", "posted-price",
                            "own-bid-excluded", "peer-bid-context",
                            "buyback-0.1", "mt-floor-0.15",
                            "mt-cap-1.7", "mt-offer-penalty"],
        wandb_mode       = "online",   # switch to "offline" if no internet
    )

    trainer = VPPTrainerMulti(
        mechanism=mech,
        prior=prior,
        cfg=cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.train()


if __name__ == "__main__":
    main()
