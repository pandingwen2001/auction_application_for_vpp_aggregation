#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity check for the own-bid-excluded peer-bid posted price.

The check changes one DER's reported bid and verifies that this DER's own
posted price is unchanged. Peer-bid aggregate features for other DERs should
still change, so the mechanism can use bid information without feeding a DER's
own bid into its own price.
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "network"))
sys.path.insert(0, _THIS_DIR)

from vpp_network_multi import build_network_multi          # noqa: E402
from vpp_mechanism_multi import VPPMechanismMulti          # noqa: E402
from trainer_multi import DERTypePriorMulti                # noqa: E402


def randomise_peer_head(mech: VPPMechanismMulti) -> None:
    torch.manual_seed(123)
    pp = mech.posted_price_net
    for module in (pp.peer_bid_mlp, pp.tr_peer_bid_mlp):
        for layer in module:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.05)
                torch.nn.init.normal_(layer.bias, mean=0.0, std=0.01)


def check_arch(net, prior, price_arch: str):
    mech = VPPMechanismMulti(
        net,
        posted_price_cfg=dict(
            price_arch=price_arch,
            pi_buyback_ratio=0.1,
            use_peer_bid_context=True,
            peer_bid_scale=0.25,
            type_cap_ratio=dict(PV=0.70, WT=0.70, DG=0.80, MT=1.70, DR=0.80),
        ),
    )
    randomise_peer_head(mech)

    bids = prior.sample(4, device="cpu")
    context = mech.context_builder.build(bids.shape[0], bids.device)
    der_idx = int(net["mt_indices"][0])

    bids_changed = bids.clone()
    bids_changed[:, der_idx, :] = prior.hi[der_idx].view(1, 2)

    pp = mech.posted_price_net
    with torch.no_grad():
        rho = pp(context, bids=bids)
        rho_changed = pp(context, bids=bids_changed)
        peer = pp._peer_bid_features(bids)
        peer_changed = pp._peer_bid_features(bids_changed)

    own_price_delta = (rho[:, :, der_idx] - rho_changed[:, :, der_idx]).abs().max()
    own_feature_delta = (peer[:, der_idx] - peer_changed[:, der_idx]).abs().max()

    other_mask = torch.ones(net["n_ders"], dtype=torch.bool)
    other_mask[der_idx] = False
    other_feature_delta = (peer[:, other_mask] - peer_changed[:, other_mask]).abs().max()

    print(f"\nprice_arch: {price_arch}")
    print(f"Changed DER index: {der_idx} ({net['der_labels'][der_idx]})")
    print(f"own price delta max   : {own_price_delta.item():.8f}")
    print(f"own feature delta max : {own_feature_delta.item():.8f}")
    print(f"other feature delta max: {other_feature_delta.item():.8f}")

    assert own_price_delta.item() < 1e-5
    assert own_feature_delta.item() < 1e-6
    assert other_feature_delta.item() > 1e-6


def main():
    torch.manual_seed(20260428)
    net = build_network_multi(constant_price=False, ctrl_min_ratio=0.15)
    prior = DERTypePriorMulti(net)
    for price_arch in ("mlp", "transformer"):
        check_arch(net, prior, price_arch)
    print("PASS: own bid is excluded from own posted price.")


if __name__ == "__main__":
    main()
