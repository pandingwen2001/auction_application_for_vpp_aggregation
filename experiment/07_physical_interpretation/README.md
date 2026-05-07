# Experiment 7: Physical Sensitivity and Learned Topology Analysis

This experiment checks whether the learned decomposed posted-price components
carry physically meaningful structure.

It compares learned price adders with feeder topology, line/voltage sensitivity
matrices, postprocess OPF duals, and MT-floor correction pressure.

## Run

```powershell
cd C:\tcxp\2026_spring\posted_price_VPP
python experiment\07_physical_interpretation\run_physical_interpretation.py --samples 12
```

Fast smoke test:

```powershell
python experiment\07_physical_interpretation\run_physical_interpretation.py --samples 2 --skip-plots
```

Paper run with public-context comparison:

```powershell
python experiment\07_physical_interpretation\run_physical_interpretation.py --samples 24 --include-public-context-baseline --top-k 3
```

## Analyses

DER/bus-level component map:

- `rho_security`, `rho_scarcity`, `rho_base`, `rho_type`, and residual components.
- DER bus, feeder depth, path resistance/reactance.
- Flow and voltage sensitivity norms.
- Dual-implied marginal security values.
- Postprocess correction and MT security uplift.

Time-level alignment:

- `rho_base` vs load and day-ahead price.
- `rho_scarcity` for MTs vs MT-floor dual and MT correction.
- `rho_security` vs line/voltage dual pressure.

Topology overlap:

- Top-k overlap between high learned security price and high physical sensitivity.
- Top-k overlap between high scarcity price and high correction/uplift burden.

## Outputs

- `results/physical_component_by_der.csv`: DER/bus-level learned component and physical sensitivity table.
- `results/physical_alignment_by_time.csv`: hourly price-component and dual/correction signals.
- `results/physical_alignment_summary.csv`: Pearson/Spearman correlations.
- `results/physical_alignment_summary.md`: Markdown correlation table.
- `results/topology_overlap_summary.csv`: top-k overlap summary.
- `results/physical_run_summary.csv`: one-row run diagnostics per method/checkpoint.
- `results/physical_interpretation_config.json`: run settings and plot paths.

If plotting is enabled and matplotlib is installed:

- `results/fig_security_component_by_der.png`
- `results/fig_time_scarcity_alignment.png`
- `results/fig_security_physical_scatter.png`

## Key Diagnostics

- `rho_security_abs_mean` vs `voltage_sensitivity_l1`.
- `rho_security_abs_mean` vs `dual_security_abs_value_mean`.
- `MT_rho_scarcity_abs_mean` vs `mt_floor_dual_abs_mean`.
- `MT_rho_scarcity_abs_mean` vs `MT_positive_adjustment_mwh`.
- Top-k overlap between learned component magnitude and physical sensitivity.

Important wording for the paper: this is evidence that the decomposed price
contains physically interpretable structure. It is not a proof that the neural
network has learned the exact OPF dual map.
