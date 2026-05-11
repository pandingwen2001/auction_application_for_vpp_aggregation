#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainer_multi.py
----------------
Augmented Lagrangian training for the multi-period (24h) VPP mechanism.

Key differences from single-period trainer.py:
  * mech forward returns (x [B, T, N], rho [B, T, N], p [B, N], P_VPP [B, T])
  * utility: u_i = p_i - Σ_t c_i(x_{i, t})                     [B, N]
  * system_cost: Σ_i p_i + Σ_t pi_DA_t * P_VPP_t                scalar
  * DER type space unchanged: (a_i, b_i), 2D private

Training stability playbook (§8.5 of 24h_extension_guide.md):
  1. Curriculum on T          — external; this trainer accepts any T
  2. Warm start               — optional via `load_state_dict_partial`
  3. Gradient clipping        — `grad_clip_norm` in cfg (default 1.0)
  4. LayerNorm                — already in PostedPriceNetworkMulti heads
  5. Two-stage loss schedule  — `warmup_iters` iters without regret penalty,
                                then regret λ ramps as `λ_max * (1 - exp(-it/τ))`
  6. Checkpoint every 500 iters with best-regret tracking
"""

import os
import sys
import csv
import time
import datetime
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

wandb = None
_WANDB_AVAILABLE = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "network"))
sys.path.insert(0, _THIS_DIR)

from vpp_network_multi import build_network_multi, VPPNetworkMulti   # noqa: E402
from opf_layer_multi import DC3OPFLayerMulti                         # noqa: E402
from vpp_mechanism_multi import VPPMechanismMulti                    # noqa: E402


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# DER type prior — unchanged from single-period
# (type is still (a, b), time-invariant)
# ---------------------------------------------------------------------------

class DERTypePriorMulti:
    def __init__(self, net: dict):
        self.N = net["n_ders"]
        self.lo = torch.stack([
            torch.tensor(net["mc_a_lo"], dtype=torch.float32),
            torch.tensor(net["mc_b_lo"], dtype=torch.float32),
        ], dim=-1)                           # [N, 2]
        self.hi = torch.stack([
            torch.tensor(net["mc_a_hi"], dtype=torch.float32),
            torch.tensor(net["mc_b_hi"], dtype=torch.float32),
        ], dim=-1)                           # [N, 2]

    def sample(self, batch_size: int, device="cpu") -> torch.Tensor:
        lo = self.lo.to(device)
        hi = self.hi.to(device)
        rand = torch.rand(batch_size, self.N, 2, device=device)
        return rand * (hi - lo).unsqueeze(0) + lo.unsqueeze(0)

    def project(self, bids: torch.Tensor) -> torch.Tensor:
        lo = self.lo.to(bids.device)
        hi = self.hi.to(bids.device)
        return bids.clamp(lo.unsqueeze(0), hi.unsqueeze(0))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VPPTrainerMulti:
    """
    Augmented Lagrangian trainer for the multi-period mechanism.

    Loss:
        L = system_cost
          + Σ_i [ w_rgt_i * rgt_i + (γ/2) * rgt_i² ]
          + Σ_i [ w_ir_i  * ir_i  + (γ_ir/2) * ir_i² ]
          + α * DC3_constraint_penalty

    where rgt_i is the gradient-ascent-estimated misreport regret and
    ir_i = E[relu(-u_i(truthful))].

    Two-stage schedule:
      - Iters 0..warmup_iters-1:
          loss = system_cost + IR penalty + constraint_penalty
          (regret penalty is off; posted prices still learn from procurement)
      - Iters warmup_iters..max_iters-1:
          full augmented Lagrangian loss
          regret penalty weight scaled by (1 - exp(-((it-warmup)/tau_ramp)))
    """

    DEFAULT_CFG = dict(
        seed               = 42,
        max_iter           = 30000,
        lr                 = 1e-3,
        grad_clip_norm     = 1.0,      # §8.5 #3
        warmup_iters       = 500,      # §8.5 #5 stage 1
        tau_ramp           = 800,      # §8.5 #5 stage 2 ramp time
        # Dual variables
        gamma              = 1.0,
        gamma_ir           = 1.0,
        w_rgt_init         = 1.0,
        w_ir_init          = 1.0,
        update_rate        = 1.0,
        rgt_tol_mean       = 0.001,
        ir_tol_mean        = 0.001,
        ir_tol_max         = 0.005,
        # Misreport inner loop
        gd_iter            = 20,
        gd_lr              = 0.1,
        num_misreports     = 1,
        adv_reuse          = True,
        # Batch / data
        batch_size         = 64,
        num_batches        = 2000,
        # DC3 constraint penalty weight
        constr_penalty_w   = 100.0,
        # Directly push posted prices high enough for MTs to offer the
        # controllable-generation floor before the OPF dispatch step.
        mt_offer_penalty_w = 100.0,
        # Low grid buyback outside option:
        # u_i must exceed max_y sum_t [ratio*pi_DA_t*y_t - cost_i(y_t)].
        # Set <= 0 to recover the original u_i >= 0 IR constraint.
        pi_buyback_ratio   = 0.1,
        # Logging
        print_iter         = 100,
        save_iter          = 500,      # §8.5 #6
        log_every          = 50,
        # wandb
        use_wandb          = False,
        wandb_project      = "vpp-multi-period",
        wandb_run_name     = None,     # None → auto "phase1a_<timestamp>"
        wandb_tags         = None,
        wandb_mode         = "online", # "online" | "offline" | "disabled"
    )

    def __init__(self, mechanism: VPPMechanismMulti,
                 prior: DERTypePriorMulti,
                 cfg: dict = None, device="cpu", out_dir=None,
                 scenarios: list = None):
        """
        Parameters
        ----------
        scenarios : list of net_multi dicts, or None
            If provided, training cycles through these scenarios uniformly at
            random — each iteration calls `mechanism.set_scenario(...)` so all
            batches in that iteration share one scenario. If None, classical
            single-scenario training (Liu profiles).
        """
        self.cfg = {**self.DEFAULT_CFG, **(cfg or {})}
        self.device = torch.device(device)
        self.mech = mechanism.to(self.device)
        self.prior = prior
        self.N = prior.N
        self.T = mechanism.T

        self.scenarios = scenarios
        self._multi_scenario = scenarios is not None and len(scenarios) > 1
        if self._multi_scenario:
            print(f"  Multi-scenario training enabled: {len(scenarios)} scenarios")
            # Validate that all scenarios share the same N, T (sanity check)
            for i, sc in enumerate(scenarios):
                if sc["n_ders"] != self.N or sc["T"] != self.T:
                    raise ValueError(
                        f"Scenario {i}: shape mismatch "
                        f"(N={sc['n_ders']} T={sc['T']}; "
                        f"trainer expects N={self.N} T={self.T})")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if out_dir is None:
            out_dir = os.path.join(_THIS_DIR, "..", "runs", f"phase1a_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.run_timestamp = ts
        print(f"  Output dir: {self.out_dir}")

        # ---- wandb init ----
        self.wandb_run = None
        if self.cfg["use_wandb"]:
            global wandb, _WANDB_AVAILABLE
            if _WANDB_AVAILABLE is None:
                try:
                    import wandb as _wandb
                    wandb = _wandb
                    _WANDB_AVAILABLE = True
                except ImportError:
                    _WANDB_AVAILABLE = False
            if not _WANDB_AVAILABLE:
                print("  [wandb] 'wandb' package not installed — skipping logging")
            else:
                run_name = self.cfg["wandb_run_name"] or f"phase1a_{ts}"
                self.wandb_run = wandb.init(
                    project=self.cfg["wandb_project"],
                    name=run_name,
                    tags=self.cfg["wandb_tags"],
                    mode=self.cfg["wandb_mode"],
                    config=self.cfg,
                    dir=self.out_dir,
                )
                print(f"  [wandb] project={self.cfg['wandb_project']} "
                      f"run={run_name} mode={self.cfg['wandb_mode']}")

        # Optimisers
        self.opt_net = torch.optim.Adam(
            self.mech.parameters(), lr=self.cfg["lr"])

        # Dual variables
        self.w_rgt = nn.Parameter(
            torch.ones(self.N, device=self.device) * self.cfg["w_rgt_init"])
        self.w_ir = nn.Parameter(
            torch.ones(self.N, device=self.device) * self.cfg["w_ir_init"])
        self.opt_lag = torch.optim.SGD(
            [self.w_rgt, self.w_ir], lr=self.cfg["update_rate"])

        # Grid-buyback outside option. This is intentionally much lower
        # than the day-ahead purchase price because exporting DER power to
        # the grid is uncertain/discounted from the VPP's perspective.
        net_multi = mechanism.net_multi
        ratio = float(self.cfg["pi_buyback_ratio"])
        self._use_outside = ratio > 0.0

        def _build_scenario_tensors(net_dict):
            pi_DA_np = np.asarray(net_dict["pi_DA_profile"], dtype=np.float32)
            return dict(
                pi_buyback_T = torch.tensor(ratio * pi_DA_np,
                                            dtype=torch.float32,
                                            device=self.device),
                x_bar_T_N    = torch.tensor(net_dict["x_bar_profile"],
                                            dtype=torch.float32,
                                            device=self.device),
                load_T       = torch.tensor(net_dict["load_profile"],
                                            dtype=torch.float32,
                                            device=self.device),
                pi_DA_T      = torch.tensor(pi_DA_np,
                                            dtype=torch.float32,
                                            device=self.device),
            )

        if self._multi_scenario:
            self._scenario_tensors = [_build_scenario_tensors(sc)
                                      for sc in self.scenarios]
            self.current_scenario_idx = 0
            self._apply_scenario_state(0)
        else:
            t = _build_scenario_tensors(net_multi)
            self.pi_buyback_T = t["pi_buyback_T"]
            self.x_bar_T_N    = t["x_bar_T_N"]
            self.load_T       = t["load_T"]
            self.pi_DA_T      = t["pi_DA_T"]
            self.current_scenario_idx = None

        labels = net_multi.get("der_labels", [f"DER_{i}" for i in range(self.N)])
        der_types = net_multi.get("der_type", ["DER"] * self.N)
        self.source_types = [
            self._classify_source_type(lbl, dtype)
            for lbl, dtype in zip(labels, der_types)
        ]
        ordered = ["PV", "WT", "DG", "MT", "DR"]
        self.monitor_types = [t for t in ordered if t in self.source_types]
        self.type_masks = {
            t: torch.tensor([src == t for src in self.source_types],
                            dtype=torch.bool, device=self.device)
            for t in self.monitor_types
        }

        # Pre-sample types and initial misreports
        n_inst = self.cfg["num_batches"] * self.cfg["batch_size"]
        self.types_all = prior.sample(n_inst, device="cpu").numpy()

        num_mis = self.cfg["num_misreports"]
        lo_np = prior.lo.numpy()
        hi_np = prior.hi.numpy()
        rand = np.random.rand(num_mis, n_inst, self.N, 2).astype(np.float32)
        self.adv_all = rand * (hi_np - lo_np) + lo_np
        self.indices = np.arange(n_inst)

        base_history_keys = [
            "iter", "loss", "system_cost",
            "der_payment", "grid_cost",
            "regret_mean", "regret_max",
            "ir_violation_mean", "ir_violation_max",
            "w_rgt_mean", "w_ir_mean",
            "constr_penalty", "lambda_rgt_mult",
            "ess_discharge_total", "ess_charge_total",
            "scenario_idx",
        ]
        monitor_keys = self._monitor_history_keys()
        self.history = {k: [] for k in base_history_keys + monitor_keys}
        self.best_regret_mean = float("inf")
        self.best_constr_penalty = float("inf")
        self.best_loss = float("inf")

    # ------------------------------------------------------------------
    # Scenario switching (multi-scenario mode only)
    # ------------------------------------------------------------------

    def _apply_scenario_state(self, idx: int) -> None:
        """Point trainer's per-T tensors at the cached scenario tensors and
        ask the mechanism to swap its data buffers."""
        t = self._scenario_tensors[idx]
        self.pi_buyback_T = t["pi_buyback_T"]
        self.x_bar_T_N    = t["x_bar_T_N"]
        self.load_T       = t["load_T"]
        self.pi_DA_T      = t["pi_DA_T"]
        self.mech.set_scenario(self.scenarios[idx])
        self.current_scenario_idx = idx

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_source_type(label: str, der_type: str) -> str:
        if str(label).startswith("PV"):
            return "PV"
        if str(label).startswith("WT"):
            return "WT"
        if str(label).startswith("MT"):
            return "MT"
        if der_type == "DR":
            return "DR"
        return "DG"

    @staticmethod
    def _to_float(x) -> float:
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)

    def _monitor_history_keys(self):
        keys = [
            "true_der_cost", "info_rent",
            "total_der_energy_mwh", "offer_cap_mwh", "availability_mwh",
            "dispatch_offer_util", "dispatch_avail_util",
            "grid_import_mwh", "payment_per_mwh", "true_cost_per_mwh",
            "posted_price_mean", "posted_price_min", "posted_price_max",
            "price_to_grid_ratio_mean",
            "rho_base_mean", "rho_type_mean",
            "rho_security_mean", "rho_scarcity_mean", "rho_peer_bid_mean",
            "rho_projection_gap",
            "price_floor_bind_frac", "price_cap_bind_frac",
            "truthful_utility_mean", "outside_utility_mean",
            "mt_gen_mwh", "mt_floor_mwh", "mt_floor_gap_mwh",
            "mt_offer_gap_mwh", "mt_offer_penalty",
            "opf_flow_penalty", "opf_voltage_penalty", "opf_mt_floor_penalty",
            "power_balance_residual", "grad_norm",
        ]
        for typ in self.monitor_types:
            p = typ.lower()
            keys.extend([
                f"{p}_posted_price_mean",
                f"{p}_price_to_grid_ratio",
                f"{p}_price_floor_bind_frac",
                f"{p}_price_cap_bind_frac",
                f"{p}_energy_mwh",
                f"{p}_offer_cap_mwh",
                f"{p}_dispatch_offer_util",
                f"{p}_dispatch_avail_util",
                f"{p}_payment",
                f"{p}_true_cost",
                f"{p}_info_rent",
                f"{p}_payment_per_mwh",
                f"{p}_ir_violation",
                f"{p}_regret",
            ])
        return keys

    def _compute_monitoring(self, types: torch.Tensor, x: torch.Tensor,
                            rho: torch.Tensor, p: torch.Tensor,
                            P_VPP: torch.Tensor, rgt: torch.Tensor,
                            ir_viol: torch.Tensor, u_truthful: torch.Tensor,
                            u_outside: torch.Tensor,
                            mt_offer_penalty: torch.Tensor = None,
                            ess_d: torch.Tensor = None,
                            ess_c: torch.Tensor = None,
                            price_components: dict = None) -> dict:
        eps = 1e-8
        true_cost_i = self.mech.total_cost(types, x)              # [B, N]
        info_rent_i = p - true_cost_i                             # [B, N]
        energy_i = x.sum(dim=1)                                   # [B, N]
        offer_cap = getattr(self.mech, "_last_offer_cap", None)
        if offer_cap is None:
            offer_cap = torch.zeros_like(x)
        offer_i = offer_cap.sum(dim=1)                            # [B, N]

        pi_DA = self.pi_DA_T.view(1, self.T, 1)
        price_floor = getattr(self.mech.posted_price_net,
                              "price_floor_T", self.pi_buyback_T)
        price_floor = price_floor.to(rho.device).view(1, self.T, 1)
        price_cap = getattr(self.mech.posted_price_net,
                            "price_cap_TN", None)
        if price_cap is None:
            price_cap = torch.full_like(rho, float("inf"))
        else:
            price_cap = price_cap.to(rho.device).view(1, self.T, self.N)

        energy_total = energy_i.sum(dim=-1).mean()
        offer_total = offer_i.sum(dim=-1).mean()
        avail_total = self.x_bar_T_N.sum()
        payment_total = p.sum(dim=-1).mean()
        true_cost_total = true_cost_i.sum(dim=-1).mean()
        rent_total = info_rent_i.sum(dim=-1).mean()
        grid_import = P_VPP.sum(dim=-1).mean()

        mt_mask = self.type_masks.get("MT", None)
        if mt_mask is not None and mt_mask.any():
            mt_total_t = x[:, :, mt_mask].sum(dim=-1)             # [B, T]
            mt_offer_t = offer_cap[:, :, mt_mask].sum(dim=-1)     # [B, T]
            mt_floor_t = (float(self.mech.net_multi.get("ctrl_min_ratio", 0.0))
                          * self.load_T).view(1, self.T)
            mt_gap = torch.relu(mt_floor_t - mt_total_t).sum(dim=-1).mean()
            mt_offer_gap = torch.relu(mt_floor_t - mt_offer_t).sum(dim=-1).mean()
            mt_gen = mt_total_t.sum(dim=-1).mean()
            mt_floor = mt_floor_t.sum()
        else:
            mt_gap = torch.tensor(0.0, device=x.device)
            mt_offer_gap = torch.tensor(0.0, device=x.device)
            mt_gen = torch.tensor(0.0, device=x.device)
            mt_floor = torch.tensor(0.0, device=x.device)
        if mt_offer_penalty is None:
            mt_offer_penalty = torch.tensor(0.0, device=x.device)

        dc3 = self.mech.dc3_opf
        comps = getattr(dc3, "_last_constraint_components", {})
        flow_pen = comps.get("flow", torch.tensor(0.0, device=x.device))
        volt_pen = comps.get("voltage", torch.tensor(0.0, device=x.device))
        mt_pen = comps.get("mt_floor", torch.tensor(0.0, device=x.device))

        ess_net = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device)
        if ess_d is not None and ess_c is not None:
            ess_net = (ess_d - ess_c).sum(dim=-1, keepdim=True)
        balance_resid = (x.sum(dim=-1, keepdim=True) + ess_net
                         + P_VPP.unsqueeze(-1)
                         - self.load_T.view(1, self.T, 1)).abs().max()

        if price_components is None:
            price_components = getattr(
                self.mech, "_last_price_components_detached", None)
        if price_components is None:
            price_components = getattr(self.mech, "_last_price_components", {})
        rho_base = price_components.get("rho_base", torch.zeros_like(rho))
        rho_type = price_components.get("rho_type", torch.zeros_like(rho))
        rho_security = price_components.get("rho_security", torch.zeros_like(rho))
        rho_scarcity = price_components.get("rho_scarcity", torch.zeros_like(rho))
        rho_peer_bid = price_components.get("rho_peer_bid", torch.zeros_like(rho))
        rho_unclamped = price_components.get("rho_unclamped", rho)
        if rho_unclamped.shape != rho.shape:
            rho_base = torch.zeros_like(rho)
            rho_type = torch.zeros_like(rho)
            rho_security = torch.zeros_like(rho)
            rho_scarcity = torch.zeros_like(rho)
            rho_peer_bid = torch.zeros_like(rho)
            rho_unclamped = rho
        rho_projection_gap = (rho_unclamped - rho).abs().mean()

        diag = {
            "true_der_cost": self._to_float(true_cost_total),
            "info_rent": self._to_float(rent_total),
            "total_der_energy_mwh": self._to_float(energy_total),
            "offer_cap_mwh": self._to_float(offer_total),
            "availability_mwh": self._to_float(avail_total),
            "dispatch_offer_util": self._to_float(energy_total / (offer_total + eps)),
            "dispatch_avail_util": self._to_float(energy_total / (avail_total + eps)),
            "grid_import_mwh": self._to_float(grid_import),
            "payment_per_mwh": self._to_float(payment_total / (energy_total + eps)),
            "true_cost_per_mwh": self._to_float(true_cost_total / (energy_total + eps)),
            "posted_price_mean": self._to_float(rho.mean()),
            "posted_price_min": self._to_float(rho.min()),
            "posted_price_max": self._to_float(rho.max()),
            "price_to_grid_ratio_mean": self._to_float((rho / pi_DA).mean()),
            "rho_base_mean": self._to_float(rho_base.mean()),
            "rho_type_mean": self._to_float(rho_type.mean()),
            "rho_security_mean": self._to_float(rho_security.mean()),
            "rho_scarcity_mean": self._to_float(rho_scarcity.mean()),
            "rho_peer_bid_mean": self._to_float(rho_peer_bid.mean()),
            "rho_projection_gap": self._to_float(rho_projection_gap),
            "price_floor_bind_frac": self._to_float((rho <= price_floor + 1e-3).float().mean()),
            "price_cap_bind_frac": self._to_float((rho >= price_cap - 1e-3).float().mean()),
            "truthful_utility_mean": self._to_float(u_truthful.mean()),
            "outside_utility_mean": self._to_float(u_outside.mean()),
            "mt_gen_mwh": self._to_float(mt_gen),
            "mt_floor_mwh": self._to_float(mt_floor),
            "mt_floor_gap_mwh": self._to_float(mt_gap),
            "mt_offer_gap_mwh": self._to_float(mt_offer_gap),
            "mt_offer_penalty": self._to_float(mt_offer_penalty),
            "opf_flow_penalty": self._to_float(flow_pen),
            "opf_voltage_penalty": self._to_float(volt_pen),
            "opf_mt_floor_penalty": self._to_float(mt_pen),
            "power_balance_residual": self._to_float(balance_resid),
        }

        for typ, mask in self.type_masks.items():
            pfx = typ.lower()
            if not mask.any():
                continue
            energy = energy_i[:, mask].sum(dim=-1).mean()
            offer = offer_i[:, mask].sum(dim=-1).mean()
            avail = self.x_bar_T_N[:, mask].sum()
            payment = p[:, mask].sum(dim=-1).mean()
            true_cost = true_cost_i[:, mask].sum(dim=-1).mean()
            rent = info_rent_i[:, mask].sum(dim=-1).mean()
            rho_typ = rho[:, :, mask]
            cap_typ = price_cap[:, :, mask]
            diag.update({
                f"{pfx}_posted_price_mean": self._to_float(rho_typ.mean()),
                f"{pfx}_price_to_grid_ratio": self._to_float((rho_typ / pi_DA).mean()),
                f"{pfx}_price_floor_bind_frac": self._to_float(
                    (rho_typ <= price_floor + 1e-3).float().mean()),
                f"{pfx}_price_cap_bind_frac": self._to_float(
                    (rho_typ >= cap_typ - 1e-3).float().mean()),
                f"{pfx}_energy_mwh": self._to_float(energy),
                f"{pfx}_offer_cap_mwh": self._to_float(offer),
                f"{pfx}_dispatch_offer_util": self._to_float(energy / (offer + eps)),
                f"{pfx}_dispatch_avail_util": self._to_float(energy / (avail + eps)),
                f"{pfx}_payment": self._to_float(payment),
                f"{pfx}_true_cost": self._to_float(true_cost),
                f"{pfx}_info_rent": self._to_float(rent),
                f"{pfx}_payment_per_mwh": self._to_float(payment / (energy + eps)),
                f"{pfx}_ir_violation": self._to_float(ir_viol[mask].mean()),
                f"{pfx}_regret": self._to_float(rgt[mask].mean()),
            })

        return diag

    def _compute_mt_offer_penalty(self) -> torch.Tensor:
        offer_cap = getattr(self.mech, "_last_offer_cap", None)
        mt_mask = self.type_masks.get("MT", None)
        if offer_cap is None or mt_mask is None or not mt_mask.any():
            return torch.tensor(0.0, device=self.device)

        mt_offer = offer_cap[:, :, mt_mask].sum(dim=-1, keepdim=True)
        mt_floor = (float(self.mech.net_multi.get("ctrl_min_ratio", 0.0))
                    * self.load_T).view(1, self.T, 1)
        offer_gap = torch.relu(mt_floor - mt_offer)
        return offer_gap.pow(2).sum(dim=(1, 2)).mean() * self.cfg["mt_offer_penalty_w"]

    # ------------------------------------------------------------------
    # Grid-buyback outside option
    # ------------------------------------------------------------------

    def _compute_outside_utility(self, types: torch.Tensor) -> torch.Tensor:
        """
        Per-DER utility from bypassing the VPP and selling to the grid at
        pi_buyback_t = ratio * pi_DA_t, with output chosen independently
        subject to the per-hour availability cap.

        For cost c(y) = a*y^2 + b*y, the closed-form optimum is
            y* = clamp((pi_buyback - b) / (2a), 0, x_bar).

        Returns [B, N].
        """
        if not self._use_outside:
            return torch.zeros(types.shape[0], types.shape[1],
                               device=types.device)

        a = types[..., 0].unsqueeze(1)                   # [B, 1, N]
        b = types[..., 1].unsqueeze(1)                   # [B, 1, N]
        pi_buy = self.pi_buyback_T.view(1, -1, 1)        # [1, T, 1]
        x_bar = self.x_bar_T_N.unsqueeze(0)              # [1, T, N]

        y_opt = ((pi_buy - b) / (2.0 * a + 1e-6)).clamp(min=0.0)
        y_opt = torch.min(y_opt, x_bar)
        u_t = pi_buy * y_opt - a * y_opt.pow(2) - b * y_opt
        return u_t.sum(dim=1)

    # ------------------------------------------------------------------
    # Misreport helpers
    # ------------------------------------------------------------------

    def _get_misreports(self, types: torch.Tensor,
                        adv_var: torch.Tensor):
        """
        types   : [B, N, 2]
        adv_var : [num_mis, B, N, 2]
        Returns  types_rep [N*num_mis*B, N, 2], mis [N*num_mis*B, N, 2]
        """
        B, N, _ = types.shape
        num_mis = adv_var.shape[0]
        mis_list, types_list = [], []
        for i in range(N):
            for k in range(num_mis):
                bids_k = types.clone()
                bids_k[:, i, :] = adv_var[k, :, i, :]
                mis_list.append(bids_k)
                types_list.append(types)
        return torch.cat(types_list, 0), torch.cat(mis_list, 0)

    def _compute_regret(self, types: torch.Tensor,
                        adv_var: torch.Tensor):
        """
        Returns rgt [N] (per-DER expected regret) and rgt_max (scalar).
        """
        B, N, _ = types.shape
        num_mis = adv_var.shape[0]

        # Truthful utility
        x_t, _, p_t, _ = self.mech(types)
        u_true = self.mech.utility(types, x_t, p_t)             # [B, N]

        # Misreport utility
        types_rep, mis = self._get_misreports(types, adv_var)
        x_m, _, p_m, _ = self.mech(mis)
        u_mis = self.mech.utility(types_rep, x_m, p_m)          # [N*num_mis*B, N]

        u_mis_r = u_mis.view(N, num_mis, B, N)
        u_true_r = u_true.unsqueeze(0).unsqueeze(0).expand(N, num_mis, -1, -1)

        excess = F.relu(u_mis_r - u_true_r)                     # [N, num_mis, B, N]
        # For each adversarial DER i, we only care about its OWN utility gain
        # (excess[i, k, b, i]). Take the diagonal across the (outer-DER, inner-DER)
        # axes, giving [num_mis, B, N], then max over num_mis and mean over B.
        diag = excess.diagonal(dim1=0, dim2=3)                  # [num_mis, B, N]
        rgt = diag.max(dim=0).values.mean(dim=0)                # [N]
        return rgt, rgt.max()

    def _regret_lambda_multiplier(self, it: int) -> float:
        """
        Stage-2 regret penalty ramp.
        Iters < warmup_iters: returns 0.0 (regret penalty off)
        Iters >= warmup_iters: returns 1 - exp(-(it - warmup) / tau_ramp)
                               grows from 0 to 1.
        """
        warmup = self.cfg["warmup_iters"]
        tau = self.cfg["tau_ramp"]
        if it < warmup:
            return 0.0
        return 1.0 - math.exp(-(it - warmup) / max(1e-9, tau))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        set_seed(cfg["seed"])

        print(f"\n{'='*15} VPP Multi-Period Training (with ESS) {'='*15}")
        print(f"  N DERs         : {self.N}")
        print(f"  T              : {self.T}")
        print(f"  max_iter       : {cfg['max_iter']}")
        print(f"  batch_size     : {cfg['batch_size']}")
        print(f"  warmup_iters   : {cfg['warmup_iters']} (no regret penalty)")
        print(f"  tau_ramp       : {cfg['tau_ramp']}")
        print(f"  grad_clip_norm : {cfg['grad_clip_norm']}")
        print(f"  gd_iter        : {cfg['gd_iter']}")
        if self._use_outside:
            print(f"  IR floor       : outside-option at "
                  f"pi_buyback = {cfg['pi_buyback_ratio']:.2f} * pi_DA "
                  f"(mean {self.pi_buyback_T.mean().item():.2f} $/MWh)")
        else:
            print("  IR floor       : u >= 0  (outside option disabled)")
        print()

        it = 0
        total_start = time.time()
        iter_times = []
        BS = cfg["batch_size"]
        D = len(self.indices)

        while it < cfg["max_iter"]:
            np.random.shuffle(self.indices)

            for start in range(0, D, BS):
                if it >= cfg["max_iter"]:
                    break

                t0 = time.time()
                sel = self.indices[start:start + BS]

                if self._multi_scenario:
                    scen_idx = int(np.random.randint(len(self.scenarios)))
                    self._apply_scenario_state(scen_idx)

                types = torch.tensor(self.types_all[sel], device=self.device)
                adv_var = nn.Parameter(
                    torch.tensor(self.adv_all[:, sel], device=self.device))

                # ---- Inner: find best misreport ----
                for p in self.mech.parameters():
                    p.requires_grad_(False)

                opt_mis = torch.optim.Adam([adv_var], lr=cfg["gd_lr"])
                for _ in range(cfg["gd_iter"]):
                    opt_mis.zero_grad()
                    with torch.no_grad():
                        adv_var.data = self.prior.project(adv_var.data)

                    types_rep, mis = self._get_misreports(types, adv_var)
                    x_m, _, p_m, _ = self.mech(mis)
                    u_mis = self.mech.utility(types_rep, x_m, p_m)

                    B_, N_, num_mis = types.shape[0], types.shape[1], adv_var.shape[0]
                    u_mis_r = u_mis.view(N_, num_mis, B_, N_)
                    mask = torch.zeros_like(u_mis_r)
                    for i in range(N_):
                        mask[i, :, :, i] = 1.0
                    (-(u_mis_r * mask).sum()).backward()
                    opt_mis.step()

                for p in self.mech.parameters():
                    p.requires_grad_(True)
                with torch.no_grad():
                    adv_var.data = self.prior.project(adv_var.data)

                if cfg["adv_reuse"]:
                    self.adv_all[:, sel] = adv_var.detach().cpu().numpy()

                # ---- Outer: update mechanism ----
                self.opt_net.zero_grad()

                x_t, rho_t, p_t, P_VPP_t = self.mech(types)
                truth_price_components = {
                    k: v.detach().clone()
                    for k, v in getattr(
                        self.mech, "_last_price_components", {}).items()
                    if torch.is_tensor(v)
                }
                dc3_truth = self.mech.dc3_opf
                truth_P_d = (dc3_truth._last_P_d.detach().clone()
                             if hasattr(dc3_truth, "_last_P_d")
                             and dc3_truth._last_P_d is not None else None)
                truth_P_c = (dc3_truth._last_P_c.detach().clone()
                             if hasattr(dc3_truth, "_last_P_c")
                             and dc3_truth._last_P_c is not None else None)

                # Posted-price objective: payments are the actual procurement
                # cost from the first iteration. Warmup only disables regret.
                # Regret still uses true types to evaluate incentive violations.
                rgt_mult = self._regret_lambda_multiplier(it)
                sys_cost = self.mech.system_cost(p_t, P_VPP_t)

                # DC3 constraint violation penalty
                constr_penalty = torch.tensor(0.0, device=self.device)
                if hasattr(self.mech, 'dc3_opf') and self.mech.dc3_opf is not None:
                    constr_penalty = (self.mech.dc3_opf.constraint_violation(x_t)
                                      * cfg["constr_penalty_w"])
                mt_offer_penalty = self._compute_mt_offer_penalty()

                rgt, rgt_max = self._compute_regret(types, adv_var.detach())

                # IR violation per DER: truthful utility must clear the
                # low grid-buyback outside option.
                u_truthful = self.mech.utility(types, x_t, p_t)   # [B, N]
                u_outside = self._compute_outside_utility(types)  # [B, N]
                ir_gap = u_outside - u_truthful
                ir_viol = F.relu(ir_gap).mean(dim=0)              # [N]
                ir_viol_max = F.relu(ir_gap).max().item()

                # rgt_mult already computed above (warmup sys_cost switch)
                lag_rgt = rgt_mult * (self.w_rgt.detach() * rgt).sum()
                quad_rgt = rgt_mult * (cfg["gamma"] / 2.0) * rgt.pow(2).sum()

                # IR penalty is always active (Option B gives architectural IR
                # when cost(0) = 0, but the continuous domain still needs the
                # Lagrangian to push price_i ≥ average marginal cost)
                lag_ir = (self.w_ir.detach() * ir_viol).sum()
                quad_ir = (cfg["gamma_ir"] / 2.0) * ir_viol.pow(2).sum()

                loss = (sys_cost + lag_rgt + quad_rgt + lag_ir + quad_ir
                        + constr_penalty + mt_offer_penalty)

                loss.backward()

                # §8.5 #3 gradient clipping
                grad_norm_value = 0.0
                if cfg["grad_clip_norm"] is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.mech.parameters(), cfg["grad_clip_norm"])
                    grad_norm_value = self._to_float(grad_norm)

                self.opt_net.step()

                # ---- Dual update (tolerance-gated) ----
                rgt_mean = rgt.mean().item()
                ir_mean = ir_viol.mean().item()
                rgt_ok = rgt_mean < cfg["rgt_tol_mean"] or rgt_mult == 0.0
                ir_ok = (ir_mean < cfg["ir_tol_mean"]
                         and ir_viol_max < cfg["ir_tol_max"])

                self.opt_lag.zero_grad()
                rgt_for_dual = rgt.detach() if not rgt_ok else torch.zeros_like(rgt)
                ir_for_dual = ir_viol.detach() if not ir_ok else torch.zeros_like(ir_viol)
                dual_loss = -(self.w_rgt * rgt_for_dual).sum() \
                            - (self.w_ir * ir_for_dual).sum()
                dual_loss.backward()
                self.opt_lag.step()
                with torch.no_grad():
                    self.w_rgt.clamp_(min=0.0)
                    self.w_ir.clamp_(min=0.0)

                it += 1
                iter_times.append(time.time() - t0)

                # Logging
                if it % cfg["log_every"] == 0:
                    with torch.no_grad():
                        der_pay = p_t.sum(dim=-1).mean().item()
                        pi_DA = self.mech.pi_DA_profile
                        grid_pay = (pi_DA.unsqueeze(0) *
                                    P_VPP_t).sum(dim=-1).mean().item()
                        # ESS monitoring
                        ess_d_tot = truth_P_d.sum(dim=(1, 2)).mean().item() \
                            if truth_P_d is not None else 0.0
                        ess_c_tot = truth_P_c.sum(dim=(1, 2)).mean().item() \
                            if truth_P_c is not None else 0.0
                        diag = self._compute_monitoring(
                            types, x_t, rho_t, p_t, P_VPP_t,
                            rgt, ir_viol, u_truthful, u_outside,
                            mt_offer_penalty,
                            truth_P_d, truth_P_c,
                            truth_price_components)
                        diag["grad_norm"] = grad_norm_value
                    self.history["iter"].append(it)
                    self.history["loss"].append(loss.item())
                    self.history["system_cost"].append(sys_cost.item())
                    self.history["der_payment"].append(der_pay)
                    self.history["grid_cost"].append(grid_pay)
                    self.history["regret_mean"].append(rgt.mean().item())
                    self.history["regret_max"].append(rgt_max.item())
                    self.history["ir_violation_mean"].append(ir_viol.mean().item())
                    self.history["ir_violation_max"].append(ir_viol_max)
                    self.history["w_rgt_mean"].append(self.w_rgt.mean().item())
                    self.history["w_ir_mean"].append(self.w_ir.mean().item())
                    self.history["constr_penalty"].append(constr_penalty.item())
                    self.history["lambda_rgt_mult"].append(rgt_mult)
                    self.history["ess_discharge_total"].append(ess_d_tot)
                    self.history["ess_charge_total"].append(ess_c_tot)
                    self.history["scenario_idx"].append(
                        -1 if self.current_scenario_idx is None
                        else self.current_scenario_idx)
                    for k, v in diag.items():
                        self.history[k].append(v)

                    if self.wandb_run is not None:
                        wandb_log = {
                            "train/loss":              loss.item(),
                            "train/system_cost":       sys_cost.item(),
                            "train/der_payment":       der_pay,
                            "train/grid_cost":         grid_pay,
                            "train/regret_mean":       rgt.mean().item(),
                            "train/regret_max":        rgt_max.item(),
                            "train/ir_violation_mean": ir_viol.mean().item(),
                            "train/ir_violation_max":  ir_viol_max,
                            "train/w_rgt_mean":        self.w_rgt.mean().item(),
                            "train/w_ir_mean":         self.w_ir.mean().item(),
                            "train/constr_penalty":    constr_penalty.item(),
                            "train/mt_offer_penalty":  mt_offer_penalty.item(),
                            "train/lambda_rgt_mult":   rgt_mult,
                            "train/ess_discharge_MWh": ess_d_tot,
                            "train/ess_charge_MWh":    ess_c_tot,
                            "train/scenario_idx":      (
                                -1 if self.current_scenario_idx is None
                                else self.current_scenario_idx),
                        }
                        wandb_log.update({f"monitor/{k}": v
                                          for k, v in diag.items()})
                        self.wandb_run.log(wandb_log, step=it)

                if it % cfg["print_iter"] == 0:
                    with torch.no_grad():
                        rent_print = (p_t - self.mech.total_cost(types, x_t)
                                      ).sum(dim=-1).mean().item()
                        rho_print = rho_t.mean().item()
                        mt_mask = self.type_masks.get("MT", None)
                        if mt_mask is not None and mt_mask.any():
                            mt_total_t = x_t[:, :, mt_mask].sum(dim=-1)
                            mt_floor_t = (float(self.mech.net_multi.get("ctrl_min_ratio", 0.0))
                                          * self.load_T).view(1, self.T)
                            mt_gap_print = torch.relu(mt_floor_t - mt_total_t
                                                       ).sum(dim=-1).mean().item()
                        else:
                            mt_gap_print = 0.0
                    elapsed = time.time() - total_start
                    h, rem = divmod(int(elapsed), 3600)
                    m, s = divmod(rem, 60)
                    tstr = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                    phase = "warmup " if it <= cfg["warmup_iters"] else "regret "
                    print(
                        f"[{it:5d}/{cfg['max_iter']}] {tstr} {phase}| "
                        f"SysCost={sys_cost.item():.3f} | "
                        f"rho={rho_print:.2f} rent={rent_print:.2f} | "
                        f"MTgap={mt_gap_print:.3f} | "
                        f"OfferPen={mt_offer_penalty.item():.2f} | "
                        f"Rgt={rgt.mean().item():.4f} (mult={rgt_mult:.2f}) | "
                        f"IR={ir_viol.mean().item():.4f} | "
                        f"Constr={constr_penalty.item():.3f}"
                    )

                # §8.5 #6 checkpoint every 500 iters + best-regret tracking
                if it % cfg["save_iter"] == 0:
                    path = os.path.join(self.out_dir, f"model_{it}.pth")
                    torch.save(self.mech.state_dict(), path)
                    # Best-regret model
                    if rgt_mean < self.best_regret_mean and rgt_mult > 0.5:
                        self.best_regret_mean = rgt_mean
                        best_path = os.path.join(self.out_dir, "model_best.pth")
                        torch.save(self.mech.state_dict(), best_path)
                        print(f"   New best regret: {rgt_mean:.5f}  → model_best.pth")
                        if self.wandb_run is not None:
                            self.wandb_run.summary["best_regret_mean"] = rgt_mean
                            self.wandb_run.summary["best_regret_iter"] = it
                    constr_scalar = constr_penalty.item() + mt_offer_penalty.item()
                    if constr_scalar < self.best_constr_penalty:
                        self.best_constr_penalty = constr_scalar
                        path_constr = os.path.join(self.out_dir,
                                                   "model_best_constr.pth")
                        torch.save(self.mech.state_dict(), path_constr)
                        print(f"   New best constraints: {constr_scalar:.3f} "
                              f"→ model_best_constr.pth")
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        path_loss = os.path.join(self.out_dir,
                                                 "model_best_loss.pth")
                        torch.save(self.mech.state_dict(), path_loss)
                        print(f"   New best loss: {loss.item():.3f} "
                              f"→ model_best_loss.pth")

        # Final model
        torch.save(self.mech.state_dict(),
                   os.path.join(self.out_dir, "final_model.pth"))
        total_time = time.time() - total_start
        print(f"\nDone. Total: {total_time:.1f}s | "
              f"Avg iter: {np.mean(iter_times):.4f}s")
        self._export()

        if self.wandb_run is not None:
            fig_path = os.path.join(self.out_dir, "training_curves.png")
            if os.path.exists(fig_path):
                self.wandb_run.log({
                    "final/training_curves": wandb.Image(fig_path)})
            self.wandb_run.summary["total_time_sec"] = total_time
            self.wandb_run.summary["avg_iter_sec"]   = float(np.mean(iter_times))
            self.wandb_run.finish()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self):
        csv_path = os.path.join(self.out_dir, "history.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history.keys()))
            w.writeheader()
            L = len(self.history["iter"])
            for i in range(L):
                w.writerow({k: self.history[k][i] for k in self.history})
        print(f"CSV: {csv_path}")

        iters = self.history["iter"]
        if len(iters) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        ax = axes.flatten()

        ax[0].plot(iters, self.history["system_cost"])
        ax[0].set_title("System Cost"); ax[0].set_xlabel("Iter")

        ax[1].plot(iters, self.history["regret_mean"], label="mean")
        ax[1].plot(iters, self.history["regret_max"],  label="max", alpha=0.6)
        ax[1].set_title("Regret"); ax[1].set_xlabel("Iter"); ax[1].legend()
        ax[1].set_yscale("symlog", linthresh=1e-3)

        ax[2].plot(iters, self.history["ir_violation_mean"], label="mean")
        ax[2].plot(iters, self.history["ir_violation_max"],  label="max", alpha=0.6)
        ax[2].set_title("IR Violation"); ax[2].set_xlabel("Iter"); ax[2].legend()

        ax[3].plot(iters, self.history["der_payment"], label="DER")
        ax[3].plot(iters, self.history["grid_cost"], label="Grid")
        ax[3].set_title("Payment Breakdown"); ax[3].set_xlabel("Iter"); ax[3].legend()

        ax[4].plot(iters, self.history["w_rgt_mean"], label="w_rgt")
        ax[4].plot(iters, self.history["w_ir_mean"],  label="w_ir", alpha=0.6)
        ax[4].plot(iters, self.history["lambda_rgt_mult"], label="λ_mult", linestyle="--")
        ax[4].set_title("Lagrange / Ramp"); ax[4].set_xlabel("Iter"); ax[4].legend()

        ax[5].plot(iters, self.history["constr_penalty"])
        ax[5].set_title("DC3 Constraint Penalty"); ax[5].set_xlabel("Iter")

        plt.tight_layout()
        fig_path = os.path.join(self.out_dir, "training_curves.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Plot: {fig_path}")

        if "posted_price_mean" not in self.history:
            return

        fig, axes = plt.subplots(3, 3, figsize=(16, 11))
        ax = axes.flatten()

        ax[0].plot(iters, self.history["posted_price_mean"], label="all")
        for typ in self.monitor_types:
            key = f"{typ.lower()}_posted_price_mean"
            if key in self.history:
                ax[0].plot(iters, self.history[key], label=typ)
        ax[0].set_title("Posted Price"); ax[0].set_xlabel("Iter"); ax[0].legend()

        ax[1].plot(iters, self.history["price_floor_bind_frac"], label="floor")
        ax[1].plot(iters, self.history["price_cap_bind_frac"], label="cap")
        ax[1].set_title("Price Boundary Binding"); ax[1].set_xlabel("Iter"); ax[1].legend()

        ax[2].plot(iters, self.history["total_der_energy_mwh"], label="dispatch")
        ax[2].plot(iters, self.history["offer_cap_mwh"], label="offer cap")
        ax[2].plot(iters, self.history["grid_import_mwh"], label="grid import")
        ax[2].set_title("Energy / Offers"); ax[2].set_xlabel("Iter"); ax[2].legend()

        for typ in self.monitor_types:
            key = f"{typ.lower()}_energy_mwh"
            if key in self.history:
                ax[3].plot(iters, self.history[key], label=typ)
        ax[3].set_title("Dispatch by Type"); ax[3].set_xlabel("Iter"); ax[3].legend()

        for typ in self.monitor_types:
            key = f"{typ.lower()}_info_rent"
            if key in self.history:
                ax[4].plot(iters, self.history[key], label=typ)
        ax[4].plot(iters, self.history["info_rent"], label="all", linestyle="--")
        ax[4].set_title("Info Rent"); ax[4].set_xlabel("Iter"); ax[4].legend()

        for typ in self.monitor_types:
            key = f"{typ.lower()}_ir_violation"
            if key in self.history:
                ax[5].plot(iters, self.history[key], label=typ)
        ax[5].set_title("IR Violation by Type"); ax[5].set_xlabel("Iter"); ax[5].legend()
        ax[5].set_yscale("symlog", linthresh=1e-3)

        for typ in self.monitor_types:
            key = f"{typ.lower()}_regret"
            if key in self.history:
                ax[6].plot(iters, self.history[key], label=typ)
        ax[6].set_title("Regret by Type"); ax[6].set_xlabel("Iter"); ax[6].legend()
        ax[6].set_yscale("symlog", linthresh=1e-3)

        ax[7].plot(iters, self.history["opf_flow_penalty"], label="flow")
        ax[7].plot(iters, self.history["opf_voltage_penalty"], label="voltage")
        ax[7].plot(iters, self.history["opf_mt_floor_penalty"], label="MT floor")
        ax[7].set_title("Constraint Components"); ax[7].set_xlabel("Iter"); ax[7].legend()
        ax[7].set_yscale("symlog", linthresh=1e-6)

        ax[8].plot(iters, self.history["dispatch_offer_util"], label="dispatch / offer")
        ax[8].plot(iters, self.history["dispatch_avail_util"], label="dispatch / availability")
        ax[8].plot(iters, self.history["mt_floor_gap_mwh"], label="MT floor gap")
        ax[8].plot(iters, self.history["mt_offer_gap_mwh"], label="MT offer gap")
        ax[8].set_title("Utilization / MT Gap"); ax[8].set_xlabel("Iter"); ax[8].legend()

        plt.tight_layout()
        fig_path = os.path.join(self.out_dir, "posted_price_diagnostics.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Plot: {fig_path}")


# Need math module (used by _regret_lambda_multiplier)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building multi-period network...")
    net = build_network_multi(constant_price=True)

    print("Constructing mechanism...")
    mech = VPPMechanismMulti(net)
    prior = DERTypePriorMulti(net)

    # Full training: 20000 iters
    trainer = VPPTrainerMulti(
        mechanism=mech,
        prior=prior,
        cfg=dict(
            max_iter=20000,
            batch_size=32,
            num_batches=500,
            warmup_iters=1000,
            tau_ramp=1500,
            print_iter=500,
            save_iter=2000,
            log_every=50,
            gd_iter=15,
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.train()
