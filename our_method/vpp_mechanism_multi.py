#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vpp_mechanism_multi.py
----------------------
Multi-period (24h) VPP posted-price mechanism.

Architecture:

  Stage 1 — PostedPriceNetworkMulti
    Input:  public context [B, T, N, F_ctx], DER technology class,
            optional leave-one-out peer-bid aggregates
    Output: rho [B, T, N], own-bid-excluded posted price schedule

  Stage 2 — DC3OPFLayerMulti
    Input:  rho [B, T, N] plus accepted supply caps from reported bids
    Output: x [B, T, N], P_VPP [B, T]

  Stage 3 — posted-price payment rule
    p_i = Σ_t rho_{i,t} x_{i,t}        [B, N]
    Prices are bid-independent (own-bid excluded), so bids affect quantity
    but not the price paid for that quantity.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "network"))
from opf_layer_multi import DC3OPFLayerMulti                           # noqa: E402


# ===========================================================================
# Context feature builder
# ===========================================================================

class MultiPeriodContextBuilder:
    """
    Builds per-(t, i) context features from the net_multi dict.

    Features per (t, i):
      - x_bar_profile[t, i] / x_bar_nominal[i]          (normalized capacity)
      - load_profile[t] / load_max                       (normalized load)
      - pi_DA_profile[t] / pi_DA_max                     (normalized price)
      - sin(2π t/T), cos(2π t/T)                         (hour encoding)

    Total F_ctx = 5.

    These are deterministic given the network — computed once and reused
    for every training batch.
    """

    N_CTX_FEATURES = 5

    def __init__(self, net_multi: dict):
        T = net_multi["T"]
        N = net_multi["n_ders"]

        x_bar_nom = np.asarray(net_multi["x_bar"], dtype=np.float64) + 1e-9
        x_bar_prof = np.asarray(net_multi["x_bar_profile"], dtype=np.float64)
        load_prof  = np.asarray(net_multi["load_profile"], dtype=np.float64)
        pi_DA_prof = np.asarray(net_multi["pi_DA_profile"], dtype=np.float64)

        load_max  = max(float(load_prof.max()),  1e-9)
        pi_DA_max = max(float(pi_DA_prof.max()), 1e-9)

        ctx = np.zeros((T, N, self.N_CTX_FEATURES), dtype=np.float32)
        ctx[..., 0] = x_bar_prof / x_bar_nom[None, :]
        ctx[..., 1] = (load_prof  / load_max)[:, None]
        ctx[..., 2] = (pi_DA_prof / pi_DA_max)[:, None]

        hours = np.arange(T)
        ctx[..., 3] = np.sin(2 * np.pi * hours / T)[:, None]
        ctx[..., 4] = np.cos(2 * np.pi * hours / T)[:, None]

        self.context_static = torch.tensor(ctx, dtype=torch.float32)  # [T, N, F_ctx]
        self.T = T
        self.N = N

    def build(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return context [B, T, N, F_ctx] for the given batch size."""
        return self.context_static.to(device).unsqueeze(0).expand(
            batch_size, -1, -1, -1)

    def set_scenario(self, net_multi: dict) -> None:
        """Rebuild context_static from a new net_multi (time-varying data)."""
        T = net_multi["T"]
        N = net_multi["n_ders"]
        x_bar_nom = np.asarray(net_multi["x_bar"], dtype=np.float64) + 1e-9
        x_bar_prof = np.asarray(net_multi["x_bar_profile"], dtype=np.float64)
        load_prof  = np.asarray(net_multi["load_profile"], dtype=np.float64)
        pi_DA_prof = np.asarray(net_multi["pi_DA_profile"], dtype=np.float64)
        load_max  = max(float(load_prof.max()),  1e-9)
        pi_DA_max = max(float(pi_DA_prof.max()), 1e-9)
        ctx = np.zeros((T, N, self.N_CTX_FEATURES), dtype=np.float32)
        ctx[..., 0] = x_bar_prof / x_bar_nom[None, :]
        ctx[..., 1] = (load_prof  / load_max)[:, None]
        ctx[..., 2] = (pi_DA_prof / pi_DA_max)[:, None]
        hours = np.arange(T)
        ctx[..., 3] = np.sin(2 * np.pi * hours / T)[:, None]
        ctx[..., 4] = np.cos(2 * np.pi * hours / T)[:, None]
        new_ctx = torch.tensor(ctx, dtype=torch.float32,
                               device=self.context_static.device)
        self.context_static = new_ctx


# ===========================================================================
# Bid-independent posted-price network
# ===========================================================================

class PostedPriceNetworkMulti(nn.Module):
    """
    Structured posted price schedule with optional peer-bid context.

    A DER's own bid is intentionally excluded from its own posted price.
    When peer-bid context is enabled, prices may depend on leave-one-out
    aggregate bid features from other DERs, public grid/load/availability
    context, and DER technology class.

    The total posted price is represented as

        rho_total = rho_base + rho_type + rho_security + rho_scarcity + rho_peer_bid

    before the final floor/cap projection.
    """

    TYPE_ORDER = ("PV", "WT", "DG", "MT", "DR")

    @classmethod
    def _classify_der(cls, label: str, der_type: str) -> str:
        if label.startswith("PV"):
            return "PV"
        if label.startswith("WT"):
            return "WT"
        if label.startswith("MT"):
            return "MT"
        if der_type == "DR":
            return "DR"
        return "DG"

    @staticmethod
    def _logit(x: float) -> float:
        x = min(max(float(x), 1e-4), 1.0 - 1e-4)
        return math.log(x / (1.0 - x))

    def __init__(self, net_multi: dict, F_ctx: int = 5,
                 hidden: int = 64,
                 price_floor_ratio: float = 0.1,
                 type_cap_ratio: dict = None,
                 init_gate_by_type: dict = None,
                 type_embed_dim: int = 4,
                 use_peer_bid_context: bool = False,
                 peer_bid_scale: float = 0.25):
        super().__init__()
        self.T = net_multi["T"]
        self.N = net_multi["n_ders"]
        self.use_peer_bid_context = bool(use_peer_bid_context)
        self.peer_bid_scale = float(peer_bid_scale)

        type_cap_ratio = type_cap_ratio or {
            "PV": 0.70,
            "WT": 0.70,
            "DG": 0.80,
            "MT": 1.70,
            "DR": 0.80,
        }
        init_gate_by_type = init_gate_by_type or {
            "PV": 0.25,
            "WT": 0.25,
            "DG": 0.40,
            "MT": 0.65,
            "DR": 0.35,
        }

        labels = net_multi["der_labels"]
        der_types = net_multi["der_type"]
        type_names = [self._classify_der(lbl, dt)
                      for lbl, dt in zip(labels, der_types)]
        type_to_id = {name: i for i, name in enumerate(self.TYPE_ORDER)}
        type_ids = torch.tensor([type_to_id[t] for t in type_names],
                                dtype=torch.long)
        self.register_buffer("type_ids", type_ids, persistent=False)
        type_onehot = F.one_hot(type_ids, num_classes=len(self.TYPE_ORDER)).float()
        self.register_buffer("type_onehot", type_onehot, persistent=False)

        bid_lo = np.stack([
            np.asarray(net_multi["mc_a_lo"], dtype=np.float32),
            np.asarray(net_multi["mc_b_lo"], dtype=np.float32),
        ], axis=-1)
        bid_hi = np.stack([
            np.asarray(net_multi["mc_a_hi"], dtype=np.float32),
            np.asarray(net_multi["mc_b_hi"], dtype=np.float32),
        ], axis=-1)
        self.register_buffer("bid_lo",
                             torch.tensor(bid_lo, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("bid_hi",
                             torch.tensor(bid_hi, dtype=torch.float32),
                             persistent=False)
        # Per type: mean normalized a, mean normalized b, and peer-count fraction.
        self.peer_bid_feature_dim = len(self.TYPE_ORDER) * 3

        pi_DA = np.asarray(net_multi["pi_DA_profile"], dtype=np.float32)
        self._price_floor_ratio = float(price_floor_ratio)
        cap_ratio = np.asarray([type_cap_ratio[t] for t in type_names],
                               dtype=np.float32)
        self.register_buffer("_cap_ratio_per_type",
                             torch.tensor(cap_ratio, dtype=torch.float32),
                             persistent=False)
        floor = float(price_floor_ratio) * pi_DA
        cap = pi_DA[:, None] * cap_ratio[None, :]
        cap = np.maximum(cap, floor[:, None] + 1e-3)

        self.register_buffer("price_floor_T",
                             torch.tensor(floor, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("price_cap_TN",
                             torch.tensor(cap, dtype=torch.float32),
                             persistent=False)

        init_offsets = torch.zeros(len(self.TYPE_ORDER), dtype=torch.float32)
        for name, idx in type_to_id.items():
            init_offsets[idx] = self._logit(init_gate_by_type.get(name, 0.4))
        self.type_offset = nn.Parameter(init_offsets)
        self.type_embed = nn.Embedding(len(self.TYPE_ORDER), type_embed_dim)

        # Type/context adder. Kept under the legacy name `mlp` so older
        # checkpoints continue to load.
        self.mlp = nn.Sequential(
            nn.Linear(F_ctx + type_embed_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # System-level base component: shared across DERs.
        self.system_feature_idx = (1, 2, 3, 4)  # load, DA price, sin(hour), cos(hour)
        self.base_mlp = nn.Sequential(
            nn.Linear(len(self.system_feature_idx), hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.security_mlp = nn.Sequential(
            nn.Linear(F_ctx + type_embed_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.scarcity_mlp = nn.Sequential(
            nn.Linear(F_ctx + type_embed_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.peer_bid_mlp = nn.Sequential(
            nn.Linear(F_ctx + type_embed_dim + self.peer_bid_feature_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self._component_scale = {}
        self._init_weights()

    def reset_component_scale(self) -> None:
        """Restore all price components to their learned scale."""
        self._component_scale = {}

    def set_component_scale(self, **scale: float) -> None:
        """
        Evaluation-time ablation hook.

        Accepted keys are base, type, security, scarcity, and peer_bid.
        Missing keys keep scale 1.0.
        """
        aliases = {
            "rho_base": "base",
            "rho_type": "type",
            "rho_security": "security",
            "rho_scarcity": "scarcity",
            "rho_peer_bid": "peer_bid",
        }
        out = {}
        for key, value in scale.items():
            key = aliases.get(key, key)
            if key not in {"base", "type", "security", "scarcity", "peer_bid"}:
                continue
            out[key] = float(value)
        self._component_scale.update(out)

    def _scale_component(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return float(self._component_scale.get(name, 1.0)) * value

    def set_scenario(self, net_multi: dict) -> None:
        """Recompute price floor/cap from the new pi_DA_profile.

        Per-type cap ratios and price_floor_ratio are intrinsic to the
        learned mechanism and unchanged across scenarios.
        """
        pi_DA = torch.tensor(net_multi["pi_DA_profile"], dtype=torch.float32,
                             device=self.price_floor_T.device)
        floor = self._price_floor_ratio * pi_DA
        cap = pi_DA.unsqueeze(-1) * self._cap_ratio_per_type.unsqueeze(0)
        cap = torch.maximum(cap, floor.unsqueeze(-1) + 1e-3)
        self.price_floor_T.data.copy_(floor)
        self.price_cap_TN.data.copy_(cap)

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        last = self.mlp[-1]
        nn.init.xavier_uniform_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)

        for head in (self.base_mlp, self.security_mlp, self.scarcity_mlp,
                     self.peer_bid_mlp):
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            last = head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _detach_components(components: dict) -> dict:
        return {
            key: value.detach()
            for key, value in components.items()
            if torch.is_tensor(value)
        }

    def _normalise_bids(self, bids: torch.Tensor) -> torch.Tensor:
        denom = (self.bid_hi - self.bid_lo).clamp(min=1e-6)
        return ((bids - self.bid_lo) / denom).clamp(0.0, 1.0)

    def _peer_bid_features(self, bids: torch.Tensor) -> torch.Tensor:
        """
        Build leave-one-out type-level peer-bid features.

        For DER i, the returned feature excludes bid_i from every aggregate
        that will be used to price DER i. This keeps rho_i independent of
        DER i's own reported cost parameters while still exposing market-state
        information from other DERs.
        """
        B, N, _ = bids.shape
        assert N == self.N

        z = self._normalise_bids(bids)                         # [B, N, 2]
        type_onehot = self.type_onehot.to(device=bids.device,
                                          dtype=z.dtype)        # [N, K]
        K = type_onehot.shape[1]

        type_sum = torch.einsum("bnf,nk->bkf", z, type_onehot) # [B, K, 2]
        type_count = type_onehot.sum(dim=0).view(1, 1, K, 1)   # [1, 1, K, 1]

        own_type = type_onehot.view(1, N, K, 1)                # [1, N, K, 1]
        own_z = z.unsqueeze(2) * own_type                      # [B, N, K, 2]
        loo_sum = type_sum.unsqueeze(1) - own_z                # [B, N, K, 2]
        loo_count = type_count - own_type                      # [1, N, K, 1]

        peer_mean = torch.where(
            loo_count > 0.0,
            loo_sum / loo_count.clamp(min=1.0),
            torch.zeros_like(loo_sum),
        )
        count_frac = (loo_count.squeeze(-1)
                      / max(1.0, float(max(self.N - 1, 1))))
        count_frac = count_frac.expand(B, -1, -1)              # [B, N, K]

        return torch.cat([
            peer_mean.reshape(B, N, K * 2),
            count_frac.reshape(B, N, K),
        ], dim=-1)

    def forward(self, context: torch.Tensor,
                bids: torch.Tensor = None) -> torch.Tensor:
        """
        context : [B, T, N, F_ctx]
        bids    : [B, N, 2] optional reported costs for peer-bid context
        returns rho : [B, T, N] posted price in $/MWh
        """
        B, T, N, _ = context.shape
        assert T == self.T
        assert N == self.N

        type_emb = self.type_embed(self.type_ids).view(1, 1, N, -1)
        type_emb = type_emb.expand(B, T, -1, -1)
        features = torch.cat([context, type_emb], dim=-1)

        floor = self.price_floor_T.view(1, T, 1)
        cap = self.price_cap_TN.view(1, T, N)
        span = cap - floor

        # Base component: system-level public signal, shared across DERs.
        system_features = context[:, :, 0, list(self.system_feature_idx)]
        base_gate_delta = torch.tanh(self.base_mlp(system_features).squeeze(-1))
        rho_base = floor.expand(B, T, N) + span * base_gate_delta.unsqueeze(-1)
        rho_base = self._scale_component("base", rho_base)

        # Type/context adder.
        type_raw = self.mlp(features).squeeze(-1)
        type_raw = type_raw + self.type_offset[self.type_ids].view(1, 1, N)
        rho_type = span * torch.sigmoid(type_raw)
        rho_type = self._scale_component("type", rho_type)

        rho_security = span * torch.tanh(self.security_mlp(features).squeeze(-1))
        rho_scarcity = span * torch.tanh(self.scarcity_mlp(features).squeeze(-1))
        rho_security = self._scale_component("security", rho_security)
        rho_scarcity = self._scale_component("scarcity", rho_scarcity)

        if self.use_peer_bid_context:
            if bids is None:
                raise ValueError("bids are required when use_peer_bid_context=True")
            peer_bid = self._peer_bid_features(bids).unsqueeze(1)
            peer_bid = peer_bid.expand(-1, T, -1, -1)
            peer_features = torch.cat([features, peer_bid], dim=-1)
            rho_peer_bid = (self.peer_bid_scale * span * torch.tanh(
                self.peer_bid_mlp(peer_features).squeeze(-1)))
        else:
            rho_peer_bid = torch.zeros_like(rho_type)
        rho_peer_bid = self._scale_component("peer_bid", rho_peer_bid)

        rho_unclamped = rho_base + rho_type + rho_security + rho_scarcity + rho_peer_bid
        rho = torch.max(torch.min(rho_unclamped, cap), floor)

        components = {
            "rho_base":      rho_base,
            "rho_type":      rho_type,
            "rho_security":  rho_security,
            "rho_scarcity":  rho_scarcity,
            "rho_peer_bid":  rho_peer_bid,
            "rho_unclamped": rho_unclamped,
            "rho_total":     rho,
        }
        self._last_price_components = components
        self._last_price_components_detached = self._detach_components(components)
        return rho


# ===========================================================================
# Full multi-period mechanism
# ===========================================================================

class VPPMechanismMulti(nn.Module):
    """
    Full multi-period posted-price mechanism.

    forward(bids, context=None):
        bids    : [B, N, 2]  reported cost parameters used to form supply caps
        context : [B, T, N, F_ctx] or None (auto-built from net if None)
      Returns:
        x          [B, T, N]    dispatch (MW per hour)
        rho        [B, T, N]    own-bid-excluded posted price schedule
        p          [B, N]       payment: sum_t rho_{i,t} x_{i,t}
        P_VPP      [B, T]       upstream import per hour
    """

    def __init__(self, net_multi: dict,
                 dc3_opf: DC3OPFLayerMulti = None,
                 shadow_cfg: dict = None,
                 posted_price_cfg: dict = None):
        super().__init__()
        self.net_multi = net_multi
        self.T = net_multi["T"]
        self.N = net_multi["n_ders"]

        self.register_buffer("pi_DA_profile",
            torch.tensor(net_multi["pi_DA_profile"], dtype=torch.float32),
            persistent=False)

        # DC3 OPF
        self.dc3_opf = dc3_opf or DC3OPFLayerMulti(net_multi)

        cfg = {}
        if shadow_cfg:
            cfg.update(shadow_cfg)
        if posted_price_cfg:
            cfg.update(posted_price_cfg)
        self.context_builder = MultiPeriodContextBuilder(net_multi)
        self.posted_price_net = PostedPriceNetworkMulti(
            net_multi=net_multi,
            F_ctx=self.context_builder.N_CTX_FEATURES,
            hidden=cfg.get("price_hidden", cfg.get("d_model", 64)),
            price_floor_ratio=cfg.get("pi_buyback_ratio", 0.1),
            type_cap_ratio=cfg.get("type_cap_ratio", None),
            init_gate_by_type=cfg.get("init_gate_by_type", None),
            type_embed_dim=cfg.get("type_embed_dim", 4),
            use_peer_bid_context=cfg.get("use_peer_bid_context", False),
            peer_bid_scale=cfg.get("peer_bid_scale", 0.25),
        )
        self.register_buffer("x_bar_profile",
            torch.tensor(net_multi["x_bar_profile"], dtype=torch.float32),
            persistent=False)

    def set_scenario(self, net_multi: dict) -> None:
        """Hot-swap the time-varying profile data without resetting any
        learned parameters. Used by multi-scenario training to switch
        between the 24 ERCOT typical days each iteration.
        """
        self.net_multi = net_multi
        self.pi_DA_profile.data.copy_(
            torch.tensor(net_multi["pi_DA_profile"], dtype=torch.float32))
        self.x_bar_profile.data.copy_(
            torch.tensor(net_multi["x_bar_profile"], dtype=torch.float32))
        self.dc3_opf.set_scenario(net_multi)
        self.context_builder.set_scenario(net_multi)
        self.posted_price_net.set_scenario(net_multi)

    def _accepted_supply_cap(self, bids: torch.Tensor,
                             rho: torch.Tensor) -> torch.Tensor:
        """
        DER myopic supply at the posted price implied by its reported
        quadratic cost curve c(x)=a*x^2+b*x:
            q = argmax_y rho*y - a*y^2 - b*y.
        """
        a = bids[..., 0].unsqueeze(1).clamp(min=1e-6)          # [B, 1, N]
        b = bids[..., 1].unsqueeze(1)                          # [B, 1, N]
        x_bar = self.x_bar_profile.unsqueeze(0)                # [1, T, N]
        q = ((rho - b) / (2.0 * a)).clamp(min=0.0)
        return torch.min(q, x_bar)

    def forward(self, bids: torch.Tensor, context: torch.Tensor = None):
        B = bids.shape[0]
        if context is None:
            context = self.context_builder.build(B, bids.device)

        rho = self.posted_price_net(context, bids=bids)       # [B, T, N]
        supply_cap = self._accepted_supply_cap(bids, rho)     # [B, T, N]
        x, P_VPP = self.dc3_opf(rho, supply_cap=supply_cap)   # [B, T, N], [B, T]

        # Posted-price payment. A DER's own bid is excluded from its own price
        # schedule rho_i; bids only change accepted quantity.
        p = (rho * x).sum(dim=1)                              # [B, N]

        self._last_offer_cap = supply_cap
        self._last_price_components = getattr(
            self.posted_price_net, "_last_price_components", None)
        self._last_price_components_detached = getattr(
            self.posted_price_net, "_last_price_components_detached", None)

        return x, rho, p, P_VPP

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def true_cost_per_t(types: torch.Tensor,
                        x: torch.Tensor) -> torch.Tensor:
        """
        c_i(x_{i,t}) = a_i * x_{i,t}^2 + b_i * x_{i,t}    [B, T, N]
        types : [B, N, 2]  → broadcast to [B, 1, N, 2]
        x     : [B, T, N]
        """
        a = types[..., 0].unsqueeze(1)        # [B, 1, N]
        b = types[..., 1].unsqueeze(1)        # [B, 1, N]
        return a * x ** 2 + b * x             # [B, T, N]

    @classmethod
    def total_cost(cls, types: torch.Tensor,
                   x: torch.Tensor) -> torch.Tensor:
        """Σ_t c_i(x_{i,t})  →  [B, N]"""
        return cls.true_cost_per_t(types, x).sum(dim=1)

    @classmethod
    def utility(cls, types: torch.Tensor,
                x: torch.Tensor,
                p: torch.Tensor) -> torch.Tensor:
        """
        u_i = p_i - Σ_t c_i(x_{i,t})        [B, N]
        """
        return p - cls.total_cost(types, x)

    def system_cost(self, p: torch.Tensor,
                    P_VPP: torch.Tensor) -> torch.Tensor:
        """
        System cost = Σ_i p_i + Σ_t pi_DA_t * P_VPP_t, averaged over batch.
        """
        der_payment = p.sum(dim=-1)                               # [B]
        grid_cost   = (self.pi_DA_profile.unsqueeze(0) *
                       P_VPP).sum(dim=-1)                         # [B]
        return (der_payment + grid_cost).mean()

    def procurement_cost(self, p: torch.Tensor) -> torch.Tensor:
        return p.sum(dim=-1).mean()


# ===========================================================================
# Sanity check
# ===========================================================================
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(_THIS_DIR, "..", "network"))
    from vpp_network_multi import build_network_multi

    print("=== VPPMechanismMulti Sanity Check ===")
    net = build_network_multi(constant_price=True)
    mech = VPPMechanismMulti(net)

    n_params = sum(p.numel() for p in mech.parameters())
    print(f"  Total parameters: {n_params:,}")
    print(f"  T={net['T']}, N={net['n_ders']}")
    print()

    # Sample random bids
    B = 4
    N = net["n_ders"]
    torch.manual_seed(42)
    a_lo = torch.tensor(net["mc_a_lo"], dtype=torch.float32)
    a_hi = torch.tensor(net["mc_a_hi"], dtype=torch.float32)
    b_lo = torch.tensor(net["mc_b_lo"], dtype=torch.float32)
    b_hi = torch.tensor(net["mc_b_hi"], dtype=torch.float32)
    a = torch.rand(B, N) * (a_hi - a_lo) + a_lo
    b = torch.rand(B, N) * (b_hi - b_lo) + b_lo
    types = torch.stack([a, b], dim=-1)                           # [B, N, 2]
    bids = types.clone()                                          # truthful

    print(f"  bids shape: {tuple(bids.shape)}")

    x, rho, p, P_VPP = mech(bids)

    print(f"  x          shape: {tuple(x.shape)}")
    print(f"  rho        shape: {tuple(rho.shape)}")
    print(f"  p          shape: {tuple(p.shape)}")
    print(f"  P_VPP      shape: {tuple(P_VPP.shape)}")
    print()

    dc3 = mech.dc3_opf
    ess_net = dc3._last_P_d - dc3._last_P_c

    flow = (x.detach() @ dc3.A_flow.T
            + ess_net @ dc3.A_flow_ess.T)
    flow_viol = torch.maximum(
        torch.relu(flow - dc3.flow_margin_up_profile.unsqueeze(0)).amax(dim=(-2, -1)),
        torch.relu(-flow - dc3.flow_margin_dn_profile.unsqueeze(0)).amax(dim=(-2, -1)),
    )
    volt = (x.detach() @ dc3.A_volt.T
            + ess_net @ dc3.A_volt_ess.T)
    volt_viol = torch.maximum(
        torch.relu(volt - dc3.volt_margin_up_profile.unsqueeze(0)).amax(dim=(-2, -1)),
        torch.relu(-volt - dc3.volt_margin_dn_profile.unsqueeze(0)).amax(dim=(-2, -1)),
    )
    supply_cap = dc3._last_supply_cap
    cap_viol = torch.maximum(
        torch.relu(x.detach() - supply_cap).amax(dim=(-2, -1)),
        torch.relu(-x.detach()).amax(dim=(-2, -1)),
    )
    mt_floor = dc3.ctrl_min_ratio * dc3.load_profile.view(1, dc3.T)
    mt_total = x.detach().index_select(-1, dc3.mt_indices).sum(dim=-1)
    mt_viol = torch.relu(mt_floor - mt_total).amax(dim=-1)

    print(f"  line_violation  [B]: {flow_viol.detach().numpy().round(6)}")
    print(f"  volt_violation  [B]: {volt_viol.detach().numpy().round(6)}")
    print(f"  cap_violation   [B]: {cap_viol.detach().numpy().round(6)}")
    print(f"  mt_violation    [B]: {mt_viol.detach().numpy().round(6)}")
    print()

    load = torch.tensor(net["load_profile"], dtype=torch.float32)
    ess_net_sum = (dc3._last_P_d - dc3._last_P_c).sum(dim=-1)   # [B, T]
    res = (x.sum(dim=-1) + ess_net_sum + P_VPP - load.unsqueeze(0)).abs().max().item()
    print(f"  Power balance residual (max): {res:.6f}")
    print()

    print(f"  Per-DER posted price summary (sample 0):")
    print(f"    {'DER':<7}  {'a_true':>7} {'b_true':>7}  {'rho_avg':>8} "
          f"{'p':>8}  {'true_cost':>10}")
    true_cost = mech.total_cost(types, x.detach())               # [B, N]
    for i in range(N):
        print(f"    {net['der_labels'][i]:<7}  "
              f"{types[0, i, 0].item():7.3f} {types[0, i, 1].item():7.3f}  "
              f"{rho[0, :, i].mean().item():8.3f} "
              f"{p[0, i].item():8.3f}  {true_cost[0, i].item():10.3f}")

    u = mech.utility(types, x.detach(), p.detach())
    print(f"\n  utility [B, N] min = {u.min().item():.6f}  "
          f"(negative means IR violated; before training this may be negative)")
    print(f"  info_rent = {(p - true_cost).sum(dim=-1).mean().item():.4f}")
    print(f"  sys_cost  = {mech.system_cost(p, P_VPP).item():.4f}")
    print()

    loss = mech.system_cost(p, P_VPP)
    loss.backward()
    n_grad = sum(1 for p_ in mech.parameters() if p_.grad is not None
                 and p_.grad.abs().max() > 1e-12)
    n_total = sum(1 for _ in mech.parameters())
    print(f"  Params with non-zero grad: {n_grad}/{n_total}")

    smoke_ok = res < 1e-4 and cap_viol.max() < 1e-3 and n_grad > 0
    print("\n  Smoke check PASSED" if smoke_ok else "\n  Smoke check FAILED")
    if flow_viol.max() >= 1e-3 or volt_viol.max() >= 1e-3 or mt_viol.max() >= 1e-3:
        print("  Note: untrained preliminary dispatch has nonzero security violations; "
              "training penalties and postprocess handle these.")
