#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cooperative_disaggregation_multi.py
-----------------------------------
Cooperative-game disaggregation baselines for the multi-period VPP setting.

These baselines separate the dispatch problem from the settlement problem:

  1) use an efficient network-constrained dispatch supplied by the caller;
  2) compute a cooperative surplus game from ERCOT/Liu public load and price
     profiles plus DER true costs;
  3) distribute the grand-coalition surplus by VCG marginal contribution,
     Shapley value, or nucleolus;
  4) pay each DER its realized true cost plus the allocated surplus.

The characteristic value v(S) is the economic avoided grid cost from coalition
S in a simplified load-capped dispatch game:

    v(S) = max_x sum_t sum_{i in S} [pi_t x_{i,t} - c_i(x_{i,t})]
           s.t. 0 <= x_{i,t} <= xbar_{i,t}, sum_i x_{i,t} <= load_t.

This keeps the cooperative allocation fast enough for experiment tables while
the actual physical feasibility metrics still come from the caller's dispatch.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable

import numpy as np

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - scipy is normally pulled by cvxpy
    linprog = None
    SCIPY_AVAILABLE = False


def _coalition_membership(n: int) -> np.ndarray:
    """Boolean matrix M[mask, i] indicating whether DER i is in coalition mask."""
    masks = np.arange(1 << n, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1)
    return bits.astype(bool)


@lru_cache(maxsize=8)
def _cached_membership(n: int) -> np.ndarray:
    return _coalition_membership(n)


def _economic_hour_surplus(active: np.ndarray,
                           a: np.ndarray,
                           b: np.ndarray,
                           xbar_t: np.ndarray,
                           load_t: float,
                           pi_t: float,
                           n_bisect: int = 40) -> float:
    """Optimal one-hour coalition surplus against a grid price pi_t."""
    if not active.any() or load_t <= 1e-12 or pi_t <= 1e-12:
        return 0.0

    idx = np.flatnonzero(active)
    a_s = np.maximum(a[idx], 1e-9)
    b_s = b[idx]
    cap_s = np.maximum(xbar_t[idx], 0.0)

    q_at_pi = np.clip((pi_t - b_s) / (2.0 * a_s), 0.0, cap_s)
    if q_at_pi.sum() <= load_t + 1e-10:
        q = q_at_pi
    else:
        lo = float(np.min(b_s - 2.0 * a_s * cap_s))
        hi = float(pi_t)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            q_mid = np.clip((mid - b_s) / (2.0 * a_s), 0.0, cap_s)
            if q_mid.sum() > load_t:
                hi = mid
            else:
                lo = mid
        lam = 0.5 * (lo + hi)
        q = np.clip((lam - b_s) / (2.0 * a_s), 0.0, cap_s)

    surplus = pi_t * q - a_s * q ** 2 - b_s * q
    return float(np.maximum(surplus, 0.0).sum())


def coalition_values_for_sample(net: dict,
                                type_sample: np.ndarray) -> np.ndarray:
    """
    Compute v(S) for every coalition S for one DER type sample.

    Parameters
    ----------
    net : dict
        Multi-period network dictionary.
    type_sample : ndarray [N, 2]
        True cost parameters (a_i, b_i).

    Returns
    -------
    values : ndarray [2**N]
        Characteristic values, with values[0] = 0.
    """
    type_sample = np.asarray(type_sample, dtype=np.float64)
    a = type_sample[:, 0]
    b = type_sample[:, 1]
    n = int(net["n_ders"])
    T = int(net["T"])
    load = np.asarray(net["load_profile"], dtype=np.float64)
    pi = np.asarray(net["pi_DA_profile"], dtype=np.float64)
    xbar = np.asarray(net["x_bar_profile"], dtype=np.float64)
    members = _cached_membership(n)

    values = np.zeros(1 << n, dtype=np.float64)
    for mask in range(1, 1 << n):
        active = members[mask]
        val = 0.0
        for t in range(T):
            val += _economic_hour_surplus(
                active, a, b, xbar[t], float(load[t]), float(pi[t]))
        values[mask] = val
    return values


def shapley_value(values: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley value for a characteristic-value game."""
    values = np.asarray(values, dtype=np.float64)
    phi = np.zeros(n, dtype=np.float64)
    full = (1 << n) - 1
    weights = {
        k: 1.0 / (n * math.comb(n - 1, k))
        for k in range(n)
    }
    sizes = np.array([int(mask).bit_count() for mask in range(1 << n)])
    for i in range(n):
        bit = 1 << i
        acc = 0.0
        for mask in range(1 << n):
            if mask & bit:
                continue
            with_i = mask | bit
            if with_i > full:
                continue
            acc += weights[int(sizes[mask])] * (values[with_i] - values[mask])
        phi[i] = acc
    return phi


def _rank_from_masks(masks: Iterable[int], n: int) -> int:
    rows = []
    for mask in masks:
        rows.append([(mask >> i) & 1 for i in range(n)])
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.asarray(rows, dtype=np.float64), tol=1e-8))


def nucleolus_value(values: np.ndarray, n: int,
                    tol: float = 1e-7,
                    max_iter: int = None) -> np.ndarray:
    """
    Compute the nucleolus of a transferable-utility value game by sequential LPs.

    If scipy/HiGHS is unavailable or an LP fails, this falls back to the
    Shapley value rather than stopping the experiment.
    """
    if not SCIPY_AVAILABLE:
        return shapley_value(values, n)

    values = np.asarray(values, dtype=np.float64)
    full = (1 << n) - 1
    coalitions = [m for m in range(1, full)]
    fixed = []  # list of (mask, epsilon_at_tightness)
    x_best = shapley_value(values, n)
    max_iter = max_iter or n

    for _ in range(max_iter):
        # Variables are [x_0, ..., x_{n-1}, eps]. Max eps -> min -eps.
        c = np.zeros(n + 1, dtype=np.float64)
        c[-1] = -1.0

        A_ub = []
        b_ub = []
        fixed_masks = {m for m, _eps in fixed}
        for mask in coalitions:
            if mask in fixed_masks:
                continue
            row = np.zeros(n + 1, dtype=np.float64)
            for i in range(n):
                if mask & (1 << i):
                    row[i] = -1.0
            row[-1] = 1.0
            A_ub.append(row)
            b_ub.append(-values[mask])

        A_eq = []
        b_eq = []
        row_full = np.zeros(n + 1, dtype=np.float64)
        row_full[:n] = 1.0
        A_eq.append(row_full)
        b_eq.append(values[full])
        for mask, eps_value in fixed:
            row = np.zeros(n + 1, dtype=np.float64)
            for i in range(n):
                if mask & (1 << i):
                    row[i] = 1.0
            A_eq.append(row)
            b_eq.append(values[mask] + eps_value)

        res = linprog(
            c,
            A_ub=np.asarray(A_ub, dtype=np.float64) if A_ub else None,
            b_ub=np.asarray(b_ub, dtype=np.float64) if b_ub else None,
            A_eq=np.asarray(A_eq, dtype=np.float64),
            b_eq=np.asarray(b_eq, dtype=np.float64),
            bounds=[(None, None)] * (n + 1),
            method="highs",
        )
        if not res.success:
            return x_best

        x_best = np.asarray(res.x[:n], dtype=np.float64)
        eps = float(res.x[-1])
        slacks: Dict[int, float] = {}
        for mask in coalitions:
            if mask in fixed_masks:
                continue
            x_sum = sum(x_best[i] for i in range(n) if mask & (1 << i))
            slacks[mask] = x_sum - values[mask] - eps

        tight = [m for m, s in slacks.items() if abs(s) <= tol]
        if not tight and slacks:
            min_slack = min(slacks.values())
            tight = [m for m, s in slacks.items() if s <= min_slack + 10.0 * tol]

        for mask in tight:
            fixed.append((mask, eps))

        rank_masks = [full] + [m for m, _eps in fixed]
        if _rank_from_masks(rank_masks, n) >= n:
            break

    return x_best


def vcg_marginal_contribution(values: np.ndarray, n: int) -> np.ndarray:
    """VCG-style marginal-contribution utility for each DER."""
    values = np.asarray(values, dtype=np.float64)
    full = (1 << n) - 1
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        without_i = full ^ (1 << i)
        out[i] = values[full] - values[without_i]
    return np.maximum(out, 0.0)


def cooperative_payoffs(types: np.ndarray,
                        net: dict,
                        methods: Iterable[str] = ("shapley", "nucleolus"),
                        nucleolus_tol: float = 1e-7) -> Dict[str, np.ndarray]:
    """
    Batch helper returning allocated surplus payoffs for requested methods.

    Returns a dict method -> payoff array [B, N].
    """
    types = np.asarray(types, dtype=np.float64)
    B, N, _ = types.shape
    requested = tuple(methods)
    out = {method: np.zeros((B, N), dtype=np.float32) for method in requested}
    for b in range(B):
        values = coalition_values_for_sample(net, types[b])
        if "shapley" in out:
            out["shapley"][b] = shapley_value(values, N).astype(np.float32)
        if "nucleolus" in out:
            out["nucleolus"][b] = nucleolus_value(
                values, N, tol=nucleolus_tol).astype(np.float32)
        if "vcg" in out:
            out["vcg"][b] = vcg_marginal_contribution(
                values, N).astype(np.float32)
    return out
