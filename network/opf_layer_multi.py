#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opf_layer_multi.py
------------------
Multi-period (24h) DC3-style approximate OPF for the VPP mechanism.

Self-contained multi-period dispatch layer for the 33-bus VPP posted-price
project. It handles the T=24 dimension with ESS (Energy Storage Systems).

Forward signature:
  pi_tilde : [B, T, N]  shadow offering prices (Transformer dispatch head)
  Returns:
    x     : [B, T, N]   per-hour DER dispatch (MW)
    P_VPP : [B, T]      per-hour upstream import (MW)

ESS integration:
  - 2 VPP-owned ESS units participate in the bisection with fixed
    pseudo-marginal costs (not learnable, since ESS is non-strategic).
  - After bisection, a forward-pass SOC projection clips charge/discharge
    to maintain SOC bounds [soc_min, soc_max].
  - ESS net injection (P_d - P_c) enters power balance and network
    constraint calculations (line flow / voltage).
"""

import torch
import torch.nn as nn
import numpy as np


# ===========================================================================
# DC3OPFLayerMulti
# ===========================================================================

class DC3OPFLayerMulti(nn.Module):
    """
    Multi-period DC3 approximate OPF with ESS.

    Per timestep t, the economic dispatch includes 16 DERs + 2 ESS units.
    ESS units are VPP-owned (non-strategic) with fixed marginal costs.

    Parameters
    ----------
    net_multi : dict  from vpp_network_multi.build_network_multi()
    tau       : float  sigmoid temperature (smaller = sharper merit order)
    n_bisect  : int    bisection iterations
    n_correction_steps : int  projection iterations for line/voltage
    correction_lr : float      step size for correction
    ess_degrad_cost : float    ESS degradation cost per MWh throughput ($/MWh).
                               The ess_cost in Liu Table IV ($50-54/MWh) is the
                               LCOS including amortized capex. For the OPF
                               dispatch decision we use only the marginal
                               degradation component (~5 $/MWh for Li-ion).
    """

    def __init__(self, net_multi: dict,
                 tau: float = 0.5,
                 n_bisect: int = 30,
                 n_correction_steps: int = 10,
                 correction_lr: float = 0.1,
                 ess_degrad_cost: float = 5.0):
        super().__init__()
        self.net = net_multi
        self.T   = net_multi["T"]
        self.N   = net_multi["n_ders"]

        self.tau                = tau
        self.n_bisect           = n_bisect
        self.n_correction_steps = n_correction_steps
        self.correction_lr      = correction_lr

        # Static tensors (not time-varying).
        # All buffers below use persistent=False: they are deterministic
        # functions of net_multi and must be rebuilt from the (possibly new)
        # net at construction, not restored from state_dict.
        self.register_buffer("A_flow",
            torch.tensor(net_multi["A_flow"], dtype=torch.float32), persistent=False)        # [n_lines, N]
        self.register_buffer("A_volt",
            torch.tensor(net_multi["A_volt"], dtype=torch.float32), persistent=False)        # [n_buses, N]

        # Time-varying tensors (shape [T, ...])
        self.register_buffer("load_profile",
            torch.tensor(net_multi["load_profile"], dtype=torch.float32), persistent=False)  # [T]
        self.register_buffer("pi_DA_profile",
            torch.tensor(net_multi["pi_DA_profile"], dtype=torch.float32), persistent=False) # [T]
        self.register_buffer("x_bar_profile",
            torch.tensor(net_multi["x_bar_profile"], dtype=torch.float32), persistent=False) # [T, N]
        self.register_buffer("flow_margin_up_profile",
            torch.tensor(net_multi["flow_margin_up_profile"], dtype=torch.float32), persistent=False)   # [T, n_lines]
        self.register_buffer("flow_margin_dn_profile",
            torch.tensor(net_multi["flow_margin_dn_profile"], dtype=torch.float32), persistent=False)   # [T, n_lines]
        self.register_buffer("volt_margin_up_profile",
            torch.tensor(net_multi["volt_margin_up_profile"], dtype=torch.float32), persistent=False)   # [T, n_buses]
        self.register_buffer("volt_margin_dn_profile",
            torch.tensor(net_multi["volt_margin_dn_profile"], dtype=torch.float32), persistent=False)   # [T, n_buses]

        # Controllable generation constraint: local MT must supply >= ctrl_min_ratio × load
        self.ctrl_min_ratio = net_multi.get("ctrl_min_ratio", 0.15)
        self.register_buffer("mt_indices",
            torch.tensor(net_multi["mt_indices"], dtype=torch.long), persistent=False)              # MT DER indices

        # ESS parameters
        ess = net_multi["ess_params"]
        self.n_ess = int(ess["n_ess"])
        self.register_buffer("ess_power_max",
            torch.tensor(ess["ess_power_max"], dtype=torch.float32), persistent=False)                  # [n_ess]
        self.register_buffer("ess_capacity",
            torch.tensor(ess["ess_capacity"], dtype=torch.float32), persistent=False)                   # [n_ess]
        self.register_buffer("ess_soc_max",
            torch.tensor(ess["ess_soc_max_pu"], dtype=torch.float32)
            * torch.tensor(ess["ess_capacity"], dtype=torch.float32), persistent=False)                 # [n_ess] MWh
        self.register_buffer("ess_soc_min",
            torch.tensor(ess["ess_soc_min_pu"], dtype=torch.float32)
            * torch.tensor(ess["ess_capacity"], dtype=torch.float32), persistent=False)                 # [n_ess] MWh
        self.register_buffer("ess_eta_c",
            torch.tensor(ess["ess_eta_c"], dtype=torch.float32), persistent=False)                      # [n_ess]
        self.register_buffer("ess_eta_d",
            torch.tensor(ess["ess_eta_d"], dtype=torch.float32), persistent=False)                      # [n_ess]
        self.register_buffer("ess_soc_init",
            torch.tensor(ess["ess_soc_init_pu"], dtype=torch.float32)
            * torch.tensor(ess["ess_capacity"], dtype=torch.float32), persistent=False)                 # [n_ess] MWh
        self.register_buffer("ess_mc",
            torch.tensor([ess_degrad_cost] * self.n_ess, dtype=torch.float32), persistent=False)        # [n_ess]

        # ESS network sensitivity (for line flow / voltage impact)
        self.register_buffer("A_flow_ess",
            torch.tensor(net_multi["A_flow_ess"], dtype=torch.float32), persistent=False)               # [n_lines, n_ess]
        self.register_buffer("A_volt_ess",
            torch.tensor(net_multi["A_volt_ess"], dtype=torch.float32), persistent=False)               # [n_buses, n_ess]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, pi_tilde: torch.Tensor,
                supply_cap: torch.Tensor = None):
        """
        Parameters
        ----------
        pi_tilde : [B, T, N]  shadow offering prices (non-negative)
        supply_cap : [B, T, N] optional accepted supply cap from DER offers.
                     If None, physical x_bar_profile is used.

        Returns
        -------
        x     : [B, T, N]   DER dispatch (MW)
        P_VPP : [B, T]      upstream import (MW)

        Side effects (stored as attributes for monitoring):
            _last_P_d  : [B, T, n_ess]  ESS discharge (MW)
            _last_P_c  : [B, T, n_ess]  ESS charge (MW)
            _last_SOC  : [B, T+1, n_ess] SOC trajectory (MWh)
        """
        B, T, N = pi_tilde.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"
        assert N == self.N, f"Expected N={self.N}, got {N}"
        dev = pi_tilde.device
        n_ess = self.n_ess

        x_bar_phys = self.x_bar_profile.unsqueeze(0)        # [1, T, N]
        if supply_cap is None:
            x_bar = x_bar_phys
        else:
            x_bar = torch.min(supply_cap.clamp(min=0.0), x_bar_phys)
        load  = self.load_profile.view(1, T, 1)             # [1, T, 1]
        pi_DA = self.pi_DA_profile.view(1, T, 1)            # [1, T, 1]

        # ESS parameters broadcast for bisection
        ess_pmax = self.ess_power_max.view(1, 1, n_ess)     # [1, 1, n_ess]
        ess_mc   = self.ess_mc.view(1, 1, n_ess)            # [1, 1, n_ess]

        # --------------------------------------------------------------
        # Step 1: Bisection with ESS virtual channels
        # --------------------------------------------------------------
        # DER:  x_{i,t}(λ) = x_bar * σ((pi_DA - λ - pi_tilde) / τ)
        # ESS discharge: P_d_j(λ) = P_max * σ((pi_DA - λ - mc) / τ)
        # ESS charge:    P_c_j(λ) = P_max * σ((λ - pi_DA + mc) / τ)
        # Balance: Σx + ΣP_d - ΣP_c = load
        # --------------------------------------------------------------

        lam_lo = torch.full((B, T, 1), -200.0, device=dev)
        lam_hi = torch.full((B, T, 1),
                            self.pi_DA_profile.max().item() + 50.0,
                            device=dev)

        pi_tilde_det = pi_tilde.detach()
        x_bar_det = x_bar.detach()

        for _ in range(self.n_bisect):
            lam_mid = (lam_lo + lam_hi) * 0.5                               # [B, T, 1]
            x_trial = x_bar_det * torch.sigmoid(
                (pi_DA - lam_mid - pi_tilde_det) / self.tau)                # [B, T, N]
            P_d_trial = ess_pmax * torch.sigmoid(
                (pi_DA - lam_mid - ess_mc) / self.tau)                      # [B, T, n_ess]
            P_c_trial = ess_pmax * torch.sigmoid(
                (lam_mid - pi_DA + ess_mc) / self.tau)                      # [B, T, n_ess]
            net_supply = (x_trial.sum(dim=-1, keepdim=True)
                          + P_d_trial.sum(dim=-1, keepdim=True)
                          - P_c_trial.sum(dim=-1, keepdim=True))            # [B, T, 1]
            excess = net_supply - load
            lam_lo = torch.where(excess > 0, lam_mid, lam_lo)
            lam_hi = torch.where(excess > 0, lam_hi, lam_mid)

        lam_star = ((lam_lo + lam_hi) * 0.5).detach()                       # [B, T, 1]

        # DER dispatch (gradients flow through pi_tilde)
        x = x_bar * torch.sigmoid(
            (pi_DA - lam_star - pi_tilde) / self.tau)                        # [B, T, N]

        # ESS dispatch (no gradient — deterministic given λ*)
        P_d = (ess_pmax * torch.sigmoid(
            (pi_DA - lam_star - ess_mc) / self.tau)).detach()                # [B, T, n_ess]
        P_c = (ess_pmax * torch.sigmoid(
            (lam_star - pi_DA + ess_mc) / self.tau)).detach()                # [B, T, n_ess]

        # --------------------------------------------------------------
        # Step 2a: SOC projection (forward + backward for terminal SOC)
        # --------------------------------------------------------------
        SOC = torch.zeros(B, T + 1, n_ess, device=dev)
        soc_init = self.ess_soc_init.unsqueeze(0)                           # [1, n_ess]
        SOC[:, 0, :] = soc_init
        eta_c = self.ess_eta_c.view(1, n_ess)
        eta_d = self.ess_eta_d.view(1, n_ess)
        soc_max = self.ess_soc_max.view(1, n_ess)
        soc_min = self.ess_soc_min.view(1, n_ess)

        # Forward pass: clip P_c/P_d to maintain SOC bounds
        for t in range(T):
            soc_next = SOC[:, t] + eta_c * P_c[:, t] - P_d[:, t] / eta_d

            over = torch.relu(soc_next - soc_max)
            P_c[:, t] = (P_c[:, t] - over / eta_c).clamp(min=0.0)

            soc_next = SOC[:, t] + eta_c * P_c[:, t] - P_d[:, t] / eta_d
            under = torch.relu(soc_min - soc_next)
            P_d[:, t] = (P_d[:, t] - under * eta_d).clamp(min=0.0)

            SOC[:, t + 1] = SOC[:, t] + eta_c * P_c[:, t] - P_d[:, t] / eta_d

        # Backward pass: enforce SOC[T] >= SOC[0] (sustainable cycling)
        # If terminal SOC is below initial, walk backward and reduce
        # discharge (or boost charge) until the deficit is covered.
        deficit = torch.relu(soc_init - SOC[:, T])                           # [B, n_ess]
        for t in range(T - 1, -1, -1):
            if (deficit <= 1e-8).all():
                break
            # Option A: reduce discharge at this timestep
            reduce_d = torch.min(P_d[:, t], deficit * eta_d)
            P_d[:, t] = P_d[:, t] - reduce_d
            deficit = deficit - reduce_d / eta_d

            # Option B: increase charge (if there's room in SOC)
            room = soc_max - SOC[:, t] - eta_c * P_c[:, t] + P_d[:, t] / eta_d
            # room here is approximate; clamp to available power headroom
            boost_c = torch.min(
                torch.min(deficit / eta_c, self.ess_power_max.unsqueeze(0) - P_c[:, t]),
                (room / eta_c).clamp(min=0.0),
            ).clamp(min=0.0)
            P_c[:, t] = P_c[:, t] + boost_c
            deficit = deficit - boost_c * eta_c

        # Recompute SOC with adjusted P_c/P_d
        for t in range(T):
            SOC[:, t + 1] = SOC[:, t] + eta_c * P_c[:, t] - P_d[:, t] / eta_d

        # ESS net injection per timestep: [B, T, n_ess]
        ess_net = P_d - P_c                                                  # [B, T, n_ess]

        # --------------------------------------------------------------
        # Step 2b: Correction — line flow + voltage violations
        # ESS injection included in flow/voltage calculation, but only
        # DER dispatch x is corrected (ESS handled by SOC projection).
        # --------------------------------------------------------------
        for _ in range(self.n_correction_steps):
            flow = (x @ self.A_flow.T
                    + ess_net @ self.A_flow_ess.T)                            # [B, T, n_lines]
            flow_viol_up = torch.relu(
                flow - self.flow_margin_up_profile.unsqueeze(0))
            flow_viol_dn = torch.relu(
                -flow - self.flow_margin_dn_profile.unsqueeze(0))
            grad_flow = flow_viol_up @ self.A_flow - flow_viol_dn @ self.A_flow

            volt = (x @ self.A_volt.T
                    + ess_net @ self.A_volt_ess.T)                            # [B, T, n_buses]
            volt_viol_up = torch.relu(
                volt - self.volt_margin_up_profile.unsqueeze(0))
            volt_viol_dn = torch.relu(
                -volt - self.volt_margin_dn_profile.unsqueeze(0))
            grad_volt = volt_viol_up @ self.A_volt - volt_viol_dn @ self.A_volt

            x = x - self.correction_lr * (grad_flow + grad_volt)
            x = torch.min(x, x_bar).clamp(min=0.0)

        # --------------------------------------------------------------
        # Step 2c: Local controllable generation constraint
        # MT dispatch >= ctrl_min_ratio × load  per timestep
        # If MT total is below the floor, scale UP MT proportionally
        # (capped at each MT's x_bar to stay feasible)
        # --------------------------------------------------------------
        mt_min = self.ctrl_min_ratio * load                                       # [1, T, 1]
        x_mt = x.index_select(-1, self.mt_indices)                               # [B, T, N_mt]
        mt_total = x_mt.sum(dim=-1, keepdim=True)                                # [B, T, 1]
        x_bar_mt = x_bar.index_select(-1, self.mt_indices)                       # [1, T, N_mt]

        # Where MT is below floor, scale up proportionally toward x_bar
        need_boost = (mt_total < mt_min).float()                                 # [B, T, 1]
        deficit = (mt_min - mt_total).clamp(min=0.0)                             # [B, T, 1]
        headroom = (x_bar_mt - x_mt).clamp(min=0.0)                             # [B, T, N_mt]
        headroom_total = headroom.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # Distribute deficit proportionally to available headroom
        boost = headroom * (deficit / headroom_total).clamp(max=1.0)
        x_mt_boosted = x_mt + boost * need_boost
        x_mt_boosted = torch.min(x_mt_boosted, x_bar_mt)
        x = x.index_copy(-1, self.mt_indices, x_mt_boosted)

        # --------------------------------------------------------------
        # Step 3: Power balance completion (including ESS)
        # --------------------------------------------------------------
        net_der = x.sum(dim=-1, keepdim=True)                                 # [B, T, 1]
        net_ess = ess_net.sum(dim=-1, keepdim=True)                           # [B, T, 1]
        total_supply = net_der + net_ess

        # If total supply exceeds load, reduce DER output while preserving
        # the MT floor whenever it is physically/offer feasible. The previous
        # proportional scaling could undo the MT boost above.
        net_ess = torch.min(net_ess, load)                                    # ESS alone can't exceed load
        target_der = (load - net_ess).clamp(min=0.0)                           # [B, T, 1]
        excess_der = (x.sum(dim=-1, keepdim=True) - target_der).clamp(min=0.0)

        x_mt = x.index_select(-1, self.mt_indices)
        mt_total = x_mt.sum(dim=-1, keepdim=True)
        mt_protect_total = torch.min(mt_total, mt_min)
        mt_protect = x_mt * (mt_protect_total / mt_total.clamp(min=1e-8))
        x_floor = torch.zeros_like(x)
        x_floor = x_floor.index_copy(-1, self.mt_indices, mt_protect)

        reducible = (x - x_floor).clamp(min=0.0)
        reducible_total = reducible.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        reduction = reducible * (excess_der / reducible_total).clamp(max=1.0)
        x = (x - reduction).clamp(min=0.0)

        P_VPP = (load - x.sum(dim=-1, keepdim=True) - net_ess
                 ).clamp(min=0.0).squeeze(-1)                                 # [B, T]

        # Store ESS state for monitoring
        self._last_P_d = P_d
        self._last_P_c = P_c
        self._last_SOC = SOC
        self._last_supply_cap = x_bar

        return x, P_VPP

    # ------------------------------------------------------------------
    # Constraint violation (for training loss penalty)
    # ------------------------------------------------------------------

    def constraint_violation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Total squared constraint violation across B x T, averaged.
        Used as penalty in trainer loss.  Includes ESS bus injection
        in flow/voltage calculations if ESS state is available.
        """
        # ESS net injection (use stored state from last forward pass)
        ess_net = getattr(self, "_last_P_d", None)
        if ess_net is not None:
            ess_net = self._last_P_d - self._last_P_c                         # [B, T, n_ess]
        else:
            ess_net = torch.zeros(x.shape[0], x.shape[1], self.n_ess,
                                  device=x.device)

        # Line flow violation
        flow = (x @ self.A_flow.T
                + ess_net @ self.A_flow_ess.T)                                # [B, T, n_lines]
        flow_viol = (
            torch.relu(flow - self.flow_margin_up_profile.unsqueeze(0)).pow(2).sum(dim=(-2, -1)) +
            torch.relu(-flow - self.flow_margin_dn_profile.unsqueeze(0)).pow(2).sum(dim=(-2, -1))
        )

        # Voltage violation
        volt = (x @ self.A_volt.T
                + ess_net @ self.A_volt_ess.T)                                # [B, T, n_buses]
        volt_viol = (
            torch.relu(volt - self.volt_margin_up_profile.unsqueeze(0)).pow(2).sum(dim=(-2, -1)) +
            torch.relu(-volt - self.volt_margin_dn_profile.unsqueeze(0)).pow(2).sum(dim=(-2, -1))
        )

        # Local controllable generation floor. With posted-price offers, MTs
        # may decline to offer enough capacity unless the learned price is high
        # enough, so expose that violation to the training objective.
        load = self.load_profile.view(1, self.T, 1)
        mt_total = x.index_select(-1, self.mt_indices).sum(dim=-1, keepdim=True)
        mt_floor = self.ctrl_min_ratio * load
        mt_floor_viol = torch.relu(mt_floor - mt_total).pow(2).sum(dim=(-2, -1))

        flow_mean = flow_viol.mean()
        volt_mean = volt_viol.mean()
        mt_floor_mean = mt_floor_viol.mean()
        self._last_constraint_components = {
            "flow": flow_mean.detach(),
            "voltage": volt_mean.detach(),
            "mt_floor": mt_floor_mean.detach(),
        }

        return flow_mean + volt_mean + mt_floor_mean


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    _THIS = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _THIS)
    from vpp_network_multi import build_network_multi, VPPNetworkMulti

    print("=== DC3OPFLayerMulti Sanity Check ===")
    net = build_network_multi(constant_price=True)
    dc3 = DC3OPFLayerMulti(net, tau=0.5, n_bisect=30,
                           n_correction_steps=10, correction_lr=0.1)
    vpp = VPPNetworkMulti(net)

    T = net["T"]
    N = net["n_ders"]
    B = 4

    torch.manual_seed(42)
    # Generate random shadow prices: pi_tilde in range [0, 2*pi_DA]
    pi_tilde = torch.rand(B, T, N) * 2 * float(net["pi_DA"])
    pi_tilde.requires_grad_(True)

    x, P_VPP = dc3(pi_tilde)
    print(f"  pi_tilde shape : {tuple(pi_tilde.shape)}")
    print(f"  x        shape : {tuple(x.shape)}")
    print(f"  P_VPP    shape : {tuple(P_VPP.shape)}")
    print()

    # Power balance: Σ x_{i,t} + Σ(P_d - P_c)_t + P_VPP_t == load[t]
    load = torch.tensor(net["load_profile"], dtype=torch.float32)
    ess_net_sum = (dc3._last_P_d - dc3._last_P_c).sum(dim=-1)   # [B, T]
    balance = x.sum(dim=-1) + ess_net_sum + P_VPP - load.unsqueeze(0)
    print(f"  Power balance max residual: {balance.abs().max().item():.6f}")

    # ESS summary
    P_d = dc3._last_P_d[0]   # [T, n_ess]
    P_c = dc3._last_P_c[0]
    SOC = dc3._last_SOC[0]   # [T+1, n_ess]
    print(f"\n  ESS summary (sample 0):")
    print(f"    n_ess          : {dc3.n_ess}")
    print(f"    P_d total (MWh): {P_d.sum(dim=0).numpy().round(4)}")
    print(f"    P_c total (MWh): {P_c.sum(dim=0).numpy().round(4)}")
    print(f"    SOC range      : [{SOC.min(dim=0).values.numpy().round(4)}, "
          f"{SOC.max(dim=0).values.numpy().round(4)}]")
    print(f"    SOC init       : {SOC[0].numpy().round(4)}")
    print(f"    SOC final      : {SOC[-1].numpy().round(4)}")

    # Feasibility via network checker
    ll  = vpp.line_limit_violation(x.detach()).numpy()
    vv  = vpp.voltage_violation(x.detach()).numpy()
    cv  = vpp.capacity_violation(x.detach()).numpy()
    print(f"\n  Line violation  (max per sample): {ll.round(6)}")
    print(f"  Voltage violation               : {vv.round(6)}")
    print(f"  Capacity violation              : {cv.round(6)}")
    print()

    # Per-t summary for sample 0
    print("  Sample 0 per-hour dispatch summary:")
    for t in [0, 6, 12, 18]:
        xt_sum = x[0, t].sum().item()
        ess_t  = ess_net_sum[0, t].item()
        pvpp_t = P_VPP[0, t].item()
        load_t = net['load_profile'][t]
        print(f"    t={t:2d}  Σx={xt_sum:.4f}  ESS_net={ess_t:+.4f}  "
              f"P_VPP={pvpp_t:.4f}  load={load_t:.4f}  "
              f"bal={xt_sum + ess_t + pvpp_t - load_t:+.6f}")
    print()

    # Gradient check
    loss = (x * pi_tilde).sum()
    loss.backward()
    grad_nonzero = (pi_tilde.grad.abs() > 1e-12).sum().item()
    print(f"  Gradient on pi_tilde: "
          f"{grad_nonzero}/{pi_tilde.numel()} entries non-zero")
    print(f"  Max gradient magnitude: {pi_tilde.grad.abs().max().item():.6f}")
    print()

    # SOC feasibility
    soc_ok = (SOC >= dc3.ess_soc_min.unsqueeze(0) - 1e-4).all() and \
             (SOC <= dc3.ess_soc_max.unsqueeze(0) + 1e-4).all()
    print(f"  SOC feasible: {bool(soc_ok)}")

    print("  Sanity check PASSED" if
          balance.abs().max().item() < 1e-4 and ll.max() < 1e-3 and
          vv.max() < 1e-3 and cv.max() < 1e-3 and grad_nonzero > 0
          and soc_ok
          else "  Sanity check FAILED")
