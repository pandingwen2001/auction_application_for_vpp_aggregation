#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_social_opt_multi.py
----------------------------
Baseline 1: Social Optimum (oracle).

Knows the TRUE types and dispatches to minimise true social cost.
Payment = true cost (zero information rent).

Theoretical lower bound — not achievable in practice because the planner
does not actually know the types. Used as the cost floor in plots.
"""

import torch
import torch.nn as nn
import numpy as np

from baseline.baseline_common_multi import JointQPMulti, true_cost_total


class SocialOptimumMechanismMulti(nn.Module):
    def __init__(self, net_multi: dict):
        super().__init__()
        self.net = net_multi
        self.T   = int(net_multi["T"])
        self.N   = int(net_multi["n_ders"])
        self.qp  = JointQPMulti(net_multi)

    def forward(self, bids: torch.Tensor):
        """
        bids here are interpreted as the TRUE types (oracle access).
        Returns x [B,T,N], price=None, p [B,N], P_VPP [B,T]
        """
        B = bids.shape[0]
        device = bids.device
        types_np = bids.detach().cpu().numpy()                     # [B, N, 2]

        x_out    = np.zeros((B, self.T, self.N), dtype=np.float32)
        pvpp_out = np.zeros((B, self.T),         dtype=np.float32)

        for s in range(B):
            a_i = types_np[s, :, 0]                                # [N]
            b_i = types_np[s, :, 1]                                # [N]
            a_tile = np.broadcast_to(a_i[None, :], (self.T, self.N))
            b_tile = np.broadcast_to(b_i[None, :], (self.T, self.N))
            x_np, pvpp_np, _ = self.qp.solve(a_tile, b_tile)
            x_out[s]    = x_np
            pvpp_out[s] = pvpp_np

        x     = torch.tensor(x_out,    device=device)
        P_VPP = torch.tensor(pvpp_out, device=device)
        # Payment = true cost  →  utility = 0 → zero info rent
        p = true_cost_total(bids, x)                                # [B, N]
        return x, None, p, P_VPP


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from network.vpp_network_multi import build_network_multi
    from our_method.trainer_multi import DERTypePriorMulti
    from baseline.baseline_common_multi import evaluate_baseline

    net = build_network_multi(constant_price=True)
    mech = SocialOptimumMechanismMulti(net)
    prior = DERTypePriorMulti(net)
    types = prior.sample(3, device="cpu")
    out = evaluate_baseline(mech, types, net, compute_regret=False)
    print("Social Optimum:", {k: v for k, v in out.items() if not isinstance(v, np.ndarray)})
