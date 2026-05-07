#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vpp_mechanism_multi.py
----------------------
Multi-period (24h) VPP posted-price mechanism.

Architecture (per 24h_extension_guide.md §6):

  Stage 1 — PostedPriceNetworkMulti
    Input:  public context [B, T, N, F_ctx] and DER technology class
    Output: rho [B, T, N], a bid-independent posted price schedule

  Stage 2 — DC3OPFLayerMulti
    Input:  rho [B, T, N] plus accepted supply caps from reported bids
    Output: x [B, T, N], P_VPP [B, T]

  Stage 3 — posted-price payment rule
    p_i = Σ_t rho_{i,t} x_{i,t}        [B, N]
    Prices are fixed before current bids, so bids affect quantity but not
    the price paid for that quantity.

Key differences from single-period vpp_mechanism.py:
  - Current bids no longer enter the price network
  - Reported costs only determine accepted supply caps at posted prices
  - utility() sums cost over T timesteps
  - system_cost() uses time-varying pi_DA_profile
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

        # Broadcasted context [T, N, F_ctx]
        ctx = np.zeros((T, N, self.N_CTX_FEATURES), dtype=np.float32)
        ctx[..., 0] = x_bar_prof / x_bar_nom[None, :]           # per-t capacity / nominal
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


# ===========================================================================
# Factored Attention Backbone
# ===========================================================================

class FactoredAttentionBlock(nn.Module):
    """
    One factored (spatial + temporal) attention block.

    Input:  x [B, T, N, d]
    Step A: Spatial attention   (attend across N for each (b, t))
    Step B: Temporal attention  (attend across T for each (b, n))
    Output: [B, T, N, d]

    Both sub-attentions use pre-LayerNorm + residual.
    Feed-forward at the end.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.0):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, T, N, d] → [B, T, N, d]"""
        B, T, N, d = x.shape

        # --- Spatial attention: attend across N within each (B, T) ---
        # Reshape to [(B*T), N, d]
        x_sp = x.reshape(B * T, N, d)
        h_sp = self.spatial_norm(x_sp)
        h_sp, _ = self.spatial_attn(h_sp, h_sp, h_sp, need_weights=False)
        x_sp = x_sp + h_sp
        x = x_sp.reshape(B, T, N, d)

        # --- Temporal attention: attend across T within each (B, N) ---
        # Permute to [B, N, T, d], then reshape to [(B*N), T, d]
        x_tp = x.permute(0, 2, 1, 3).reshape(B * N, T, d)
        h_tp = self.temporal_norm(x_tp)
        h_tp, _ = self.temporal_attn(h_tp, h_tp, h_tp, need_weights=False)
        x_tp = x_tp + h_tp
        # Back to [B, T, N, d]
        x = x_tp.reshape(B, N, T, d).permute(0, 2, 1, 3).contiguous()

        # --- Feed-forward ---
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h

        return x


# ===========================================================================
# ShadowPriceTransformerMulti
# ===========================================================================

class ShadowPriceTransformerMulti(nn.Module):
    """
    Multi-period shadow price + payment network.

    Inputs
    ------
    bids    : [B, N, 2]           per-DER reported (a, b)
    context : [B, T, N, F_ctx]    time-varying context features

    Outputs
    -------
    pi_tilde : [B, T, N]   dispatch shadow price (positive, via softplus)
    price    : [B, N]      daily unit price    (positive, via softplus)

    Architecture
    ------------
    Token embedding: concat(bids expanded over T, context) → Linear → [B, T, N, d_model]
    L stacked FactoredAttentionBlock layers
    Heads:
      Dispatch head: Linear(d_model, 1) → pi_tilde [B, T, N]
      Payment head : temporal attention-pool over T → [B, N, d_model]
                     → Linear(d_model, 1) → price [B, N]
    """

    def __init__(self, T: int, N: int, F_ctx: int = 5,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 128, dropout: float = 0.0,
                 cost_bias_init: tuple = (5.0, 5.0)):
        super().__init__()
        self.T = T
        self.N = N
        self.F_ctx = F_ctx
        self.d_model = d_model
        self.cost_bias_init = cost_bias_init   # (a'_init, b'_init)

        # Token embedding: bid (2) + context (F_ctx) = 2+F_ctx → d_model
        self.token_embed = nn.Linear(2 + F_ctx, d_model)

        # Backbone
        self.blocks = nn.ModuleList([
            FactoredAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.backbone_norm = nn.LayerNorm(d_model)

        # Dispatch head
        self.dispatch_head = nn.Linear(d_model, 1)

        # Payment head: outputs (a'_i, b'_i) per DER — the network's estimate
        # of each DER's true quadratic cost parameters.  Payment is then
        #   p_i = Σ_t (a'_i * x_{i,t}^2 + b'_i * x_{i,t})
        # which matches the quadratic cost structure exactly, enabling near-zero
        # info rent when a' ≈ a and b' ≈ b.
        self.pool_query = nn.Parameter(torch.zeros(1, N, d_model))
        self.pool_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.payment_norm = nn.LayerNorm(d_model)
        self.payment_head = nn.Linear(d_model, 2)   # outputs (a', b')

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pool_query, std=0.02)

        # Payment head: initialise softplus bias so a' ≈ cost_bias_init[0],
        # b' ≈ cost_bias_init[1].  These are typical mid-range values for the
        # DER cost parameters (a ranges ~0-50, b ranges ~3-10).
        nn.init.xavier_uniform_(self.payment_head.weight, gain=0.1)
        a_init, b_init = self.cost_bias_init
        inv_a = math.log(math.expm1(a_init)) if a_init < 20.0 else a_init
        inv_b = math.log(math.expm1(b_init)) if b_init < 20.0 else b_init
        self.payment_head.bias.data = torch.tensor([inv_a, inv_b],
                                                    dtype=torch.float32)

    def forward(self, bids: torch.Tensor, context: torch.Tensor):
        """
        bids    : [B, N, 2]
        context : [B, T, N, F_ctx]
        Returns:
          pi_tilde   [B, T, N]    dispatch shadow prices
          cost_pred  [B, N, 2]    predicted cost params (a', b'), non-negative
        """
        B, N, _ = bids.shape
        T = context.shape[1]
        assert N == self.N
        assert T == self.T

        # Broadcast bids over T: [B, N, 2] → [B, T, N, 2]
        bids_bcast = bids.unsqueeze(1).expand(-1, T, -1, -1)

        # Concat features: [B, T, N, 2+F_ctx]
        tokens = torch.cat([bids_bcast, context], dim=-1)
        tokens = self.token_embed(tokens)                               # [B, T, N, d]

        # Backbone
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.backbone_norm(tokens)                             # [B, T, N, d]

        # --- Dispatch head (per t, per i) ---
        pi_tilde = F.softplus(self.dispatch_head(tokens).squeeze(-1))   # [B, T, N]

        # --- Payment head: estimate (a', b') per DER ---
        tokens_bn = tokens.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        q = self.pool_query.expand(B, -1, -1).reshape(B * N, 1, self.d_model)
        pooled, _ = self.pool_attn(q, tokens_bn, tokens_bn, need_weights=False)
        pooled = pooled.squeeze(1)                                      # [(B*N), d]
        pooled = self.payment_norm(pooled)
        cost_raw = self.payment_head(pooled)                            # [(B*N), 2]
        cost_pred = F.softplus(cost_raw).view(B, N, 2)                 # [B, N, 2]

        return pi_tilde, cost_pred


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

        rho_total =
            rho_base + rho_type + rho_security + rho_scarcity + rho_peer_bid

    before the final floor/cap projection.  The legacy MLP is retained as the
    initial type/context adder so existing checkpoints still reproduce their
    old prices when the new heads are zero-initialised.
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
                 peer_bid_scale: float = 0.25,
                 price_arch: str = "mlp",
                 transformer_layers: int = 2,
                 transformer_heads: int = 4,
                 transformer_dropout: float = 0.0):
        super().__init__()
        self.T = net_multi["T"]
        self.N = net_multi["n_ders"]
        self.use_peer_bid_context = bool(use_peer_bid_context)
        self.peer_bid_scale = float(peer_bid_scale)
        self.price_arch = str(price_arch).lower()
        self._component_scale = {}
        if self.price_arch not in {"mlp", "transformer"}:
            raise ValueError("price_arch must be 'mlp' or 'transformer'")
        if hidden % int(transformer_heads) != 0:
            raise ValueError("price_hidden must be divisible by transformer_heads")

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
        floor = float(price_floor_ratio) * pi_DA
        cap_ratio = np.asarray([type_cap_ratio[t] for t in type_names],
                               dtype=np.float32)
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

        # Legacy type/context adder.  Keep the module name and shape stable so
        # old checkpoints load cleanly and initialise the decomposed network at
        # the previous posted-price policy.
        self.mlp = nn.Sequential(
            nn.Linear(F_ctx + type_embed_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

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
        self.security_residual_mlp = nn.Sequential(
            nn.Linear(F_ctx + type_embed_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.scarcity_residual_mlp = nn.Sequential(
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

        # Optional public-context Transformer backbone. It intentionally uses
        # only public context and DER class embeddings. Bid-aware information is
        # still added later through pointwise leave-one-out peer features, so
        # spatial attention cannot leak DER i's own bid into rho_i.
        self.public_token_proj = nn.Linear(F_ctx + type_embed_dim, hidden)
        self.public_blocks = nn.ModuleList([
            FactoredAttentionBlock(
                d_model=hidden,
                nhead=int(transformer_heads),
                dim_feedforward=2 * hidden,
                dropout=float(transformer_dropout),
            )
            for _ in range(int(transformer_layers))
        ])
        self.public_norm = nn.LayerNorm(hidden)
        self.tr_base_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.tr_type_head = nn.Linear(hidden, 1)
        self.tr_security_head = nn.Linear(hidden, 1)
        self.tr_scarcity_head = nn.Linear(hidden, 1)
        self.tr_security_residual_head = nn.Linear(hidden, 1)
        self.tr_scarcity_residual_head = nn.Linear(hidden, 1)
        self.tr_peer_bid_mlp = nn.Sequential(
            nn.Linear(hidden + self.peer_bid_feature_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        last = self.mlp[-1]
        nn.init.xavier_uniform_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)

        for head in (self.base_mlp, self.security_mlp, self.scarcity_mlp,
                     self.security_residual_mlp, self.scarcity_residual_mlp,
                     self.peer_bid_mlp):
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            last = head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        nn.init.xavier_uniform_(self.public_token_proj.weight)
        nn.init.zeros_(self.public_token_proj.bias)
        for head in (self.tr_base_head, self.tr_peer_bid_mlp):
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            last = head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        for head in (self.tr_type_head, self.tr_security_head,
                     self.tr_scarcity_head, self.tr_security_residual_head,
                     self.tr_scarcity_residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _detach_components(components: dict) -> dict:
        return {
            key: value.detach()
            for key, value in components.items()
            if torch.is_tensor(value)
        }

    def reset_component_scale(self):
        """Restore all price components to their default scale of one."""
        self._component_scale = {}

    def set_component_scale(self, **scale: float):
        """
        Set multiplicative component scales for evaluation ablations.

        Supported keys: base, type, security_main, scarcity_main,
        security_residual, scarcity_residual, peer_bid.
        Missing keys default to 1.0. This is intentionally not part of the
        state_dict so trained checkpoints remain unchanged.
        """
        valid = {
            "base", "type", "security_main", "scarcity_main",
            "security_residual", "scarcity_residual", "peer_bid",
        }
        unknown = set(scale) - valid
        if unknown:
            raise ValueError(f"Unknown price component scale keys: {sorted(unknown)}")
        self._component_scale = {key: float(value)
                                 for key, value in scale.items()}

    def _scale_component(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return value * float(self._component_scale.get(key, 1.0))

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

    def _finalize_price(self, floor: torch.Tensor, cap: torch.Tensor,
                        rho_base: torch.Tensor, rho_type: torch.Tensor,
                        rho_security_main: torch.Tensor,
                        rho_scarcity_main: torch.Tensor,
                        rho_security_residual: torch.Tensor,
                        rho_scarcity_residual: torch.Tensor,
                        rho_peer_bid: torch.Tensor,
                        type_raw: torch.Tensor) -> torch.Tensor:
        B, T, N = rho_type.shape
        rho_base = self._scale_component("base", rho_base)
        rho_type = self._scale_component("type", rho_type)
        rho_security_main = self._scale_component(
            "security_main", rho_security_main)
        rho_scarcity_main = self._scale_component(
            "scarcity_main", rho_scarcity_main)
        rho_security_residual = self._scale_component(
            "security_residual", rho_security_residual)
        rho_scarcity_residual = self._scale_component(
            "scarcity_residual", rho_scarcity_residual)
        rho_peer_bid = self._scale_component("peer_bid", rho_peer_bid)
        rho_security = rho_security_main + rho_security_residual
        rho_scarcity = rho_scarcity_main + rho_scarcity_residual
        rho_unclamped = (rho_base + rho_type + rho_security + rho_scarcity
                         + rho_peer_bid)
        rho = torch.max(torch.min(rho_unclamped, cap), floor)

        components = {
            "rho_base": rho_base,
            "rho_type": rho_type,
            "rho_security_main": rho_security_main,
            "rho_scarcity_main": rho_scarcity_main,
            "rho_security_residual": rho_security_residual,
            "rho_scarcity_residual": rho_scarcity_residual,
            "rho_security": rho_security,
            "rho_scarcity": rho_scarcity,
            "rho_peer_bid": rho_peer_bid,
            "rho_unclamped": rho_unclamped,
            "rho_total": rho,
            "rho_floor": floor.expand(B, T, N),
            "rho_cap": cap,
            "raw_type": type_raw,
        }
        self._last_price_components = components
        self._last_price_components_detached = self._detach_components(components)
        return rho

    def _forward_transformer(self, bids: torch.Tensor, features: torch.Tensor,
                             floor: torch.Tensor, cap: torch.Tensor,
                             span: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = features.shape

        h = self.public_token_proj(features)
        for block in self.public_blocks:
            h = block(h)
        h = self.public_norm(h)

        system_h = h.mean(dim=2)
        base_gate_delta = torch.tanh(self.tr_base_head(system_h).squeeze(-1))
        rho_base = floor.expand(B, T, N) + span * base_gate_delta.unsqueeze(-1)

        type_raw = self.tr_type_head(h).squeeze(-1)
        type_raw = type_raw + self.type_offset[self.type_ids].view(1, 1, N)
        rho_type = span * torch.sigmoid(type_raw)

        rho_security_main = span * torch.tanh(
            self.tr_security_head(h).squeeze(-1))
        rho_scarcity_main = span * torch.tanh(
            self.tr_scarcity_head(h).squeeze(-1))
        rho_security_residual = span * torch.tanh(
            self.tr_security_residual_head(h).squeeze(-1))
        rho_scarcity_residual = span * torch.tanh(
            self.tr_scarcity_residual_head(h).squeeze(-1))

        if self.use_peer_bid_context:
            if bids is None:
                raise ValueError("bids are required when use_peer_bid_context=True")
            peer_bid = self._peer_bid_features(bids).unsqueeze(1)
            peer_bid = peer_bid.expand(-1, T, -1, -1)
            peer_features = torch.cat([h, peer_bid], dim=-1)
            rho_peer_bid = (self.peer_bid_scale * span * torch.tanh(
                self.tr_peer_bid_mlp(peer_features).squeeze(-1)))
        else:
            rho_peer_bid = torch.zeros_like(rho_type)

        return self._finalize_price(
            floor, cap, rho_base, rho_type,
            rho_security_main, rho_scarcity_main,
            rho_security_residual, rho_scarcity_residual,
            rho_peer_bid, type_raw)

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

        if self.price_arch == "transformer":
            return self._forward_transformer(bids, features, floor, cap, span)

        # Base component: system-level public signal, shared across DERs.
        system_features = context[:, :, 0, list(self.system_feature_idx)]
        base_gate_delta = torch.tanh(self.base_mlp(system_features).squeeze(-1))
        rho_base = floor.expand(B, T, N) + span * base_gate_delta.unsqueeze(-1)

        # Type/context component: backward-compatible legacy posted-price adder.
        type_raw = self.mlp(features).squeeze(-1)
        type_raw = type_raw + self.type_offset[self.type_ids].view(1, 1, N)
        rho_type = span * torch.sigmoid(type_raw)

        # Security/scarcity adders have a main Stage-1 component plus a
        # zero-initialised residual component for postprocess-dual refinement.
        rho_security_main = span * torch.tanh(
            self.security_mlp(features).squeeze(-1))
        rho_scarcity_main = span * torch.tanh(
            self.scarcity_mlp(features).squeeze(-1))
        rho_security_residual = span * torch.tanh(
            self.security_residual_mlp(features).squeeze(-1))
        rho_scarcity_residual = span * torch.tanh(
            self.scarcity_residual_mlp(features).squeeze(-1))
        rho_security = rho_security_main + rho_security_residual
        rho_scarcity = rho_scarcity_main + rho_scarcity_residual

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

        return self._finalize_price(
            floor, cap, rho_base, rho_type,
            rho_security_main, rho_scarcity_main,
            rho_security_residual, rho_scarcity_residual,
            rho_peer_bid, type_raw)


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

        # Public posted-price network. Keep shadow_cfg accepted for backward
        # compatibility with old experiment code; its d_model value is reused
        # as the default hidden size if supplied.
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
            price_arch=cfg.get("price_arch", "mlp"),
            transformer_layers=cfg.get("transformer_layers", 2),
            transformer_heads=cfg.get("transformer_heads", 4),
            transformer_dropout=cfg.get("transformer_dropout", 0.0),
        )
        self.register_buffer("x_bar_profile",
            torch.tensor(net_multi["x_bar_profile"], dtype=torch.float32),
            persistent=False)

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

        # Posted-price payment. Current bids can change accepted quantity, but
        # a DER's own bid is excluded from its own price schedule rho_i.
        p = (rho * x).sum(dim=1)                              # [B, N]

        self._last_pi_tilde = rho
        self._last_posted_price = rho
        self._last_offer_cap = supply_cap
        self._last_cost_pred = None
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

    # Feasibility, including ESS bus injection.
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

    # Power balance per sample (including ESS net injection)
    load = torch.tensor(net["load_profile"], dtype=torch.float32)
    ess_net_sum = (dc3._last_P_d - dc3._last_P_c).sum(dim=-1)   # [B, T]
    res = (x.sum(dim=-1) + ess_net_sum + P_VPP - load.unsqueeze(0)).abs().max().item()
    print(f"  Power balance residual (max): {res:.6f}")
    print()

    # Per-DER summary: posted price and realized payment.
    print(f"  Per-DER posted price summary (sample 0):")
    print(f"    {'DER':<7}  {'a_true':>7} {'b_true':>7}  {'rho_avg':>8} "
          f"{'p':>8}  {'true_cost':>10}")
    true_cost = mech.total_cost(types, x.detach())               # [B, N]
    for i in range(N):
        print(f"    {net['der_labels'][i]:<7}  "
              f"{types[0, i, 0].item():7.3f} {types[0, i, 1].item():7.3f}  "
              f"{rho[0, :, i].mean().item():8.3f} "
              f"{p[0, i].item():8.3f}  {true_cost[0, i].item():10.3f}")

    # IR verification: u = p - Σ_t cost
    u = mech.utility(types, x.detach(), p.detach())
    print(f"\n  utility [B, N] min = {u.min().item():.6f}  "
          f"(negative means IR violated; before training this may be negative)")
    print(f"  info_rent = {(p - true_cost).sum(dim=-1).mean().item():.4f}")
    print(f"  sys_cost  = {mech.system_cost(p, P_VPP).item():.4f}")
    print()

    # Gradient flow
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
