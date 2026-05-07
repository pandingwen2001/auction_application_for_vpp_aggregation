# Experiment Plan

This document records the experiments needed for the paper. The current
main objective is:

```text
After security postprocess, satisfy physical constraints, especially MT floor,
while keeping information rent and procurement cost low.
```

The secondary objective is:

```text
Reduce postprocess correction burden, such as MT offer gap, positive adjustment,
and additional correction payment.
```

## 1. Metric Definitions

| Metric | Meaning | Primary or secondary | Notes |
|---|---|---|---|
| `mt_floor_gap_mwh` | Gap between required MT minimum output and actual MT dispatch | Primary feasibility | For `*:post`, this should be near zero after postprocess |
| `postprocess_mt_slack_mwh` | Slack used by postprocess for MT floor feasibility | Primary feasibility | Should be near zero |
| `info_rent` | DER payment minus true DER cost | Primary economic metric | Lower is better after feasibility is satisfied |
| `procurement_cost` | DER payment plus grid import cost | Primary economic metric | Lower is better |
| `social_cost_true` | True DER cost plus grid import cost | Primary efficiency metric | Compare against constrained social optimum |
| `utility_min` | Minimum DER utility | IR check | Should be nonnegative or near nonnegative |
| `mt_offer_gap_mwh` | Required MT amount minus accepted MT offer cap | Secondary correction-burden metric | Not the same as physical MT floor violation |
| `positive_adjustment_mwh` | Total upward dispatch correction by postprocess | Secondary correction-burden metric | Lower means price-induced dispatch is closer to feasible |
| `positive_adjustment_payment` | Extra payment induced by upward correction | Secondary correction-burden metric | Lower is better |
| `dispatch_l1_gap_mwh` | Dispatch distance to constrained social optimum | Efficiency diagnostic | Lower is better |
| `post_mt_floor_dual_abs_mean` | Mean absolute MT floor dual in postprocess | Scarcity/security diagnostic | Lower usually means less pressure after postprocess |
| `post_voltage_dual_abs_max` | Max voltage dual after postprocess | Security diagnostic | Use to detect voltage stress |

## 2. Current Pipeline

| Stage | Script or module | Output | Status | Purpose |
|---|---|---|---|---|
| Stage 1 | `our_method/run_phase1a.py` | Base trained checkpoint | Done | Train bid-independent posted-price mechanism |
| Stage 2 | `our_method/vpp_mechanism_multi.py` | Decomposed price components | Done | Split price into base, type, security, scarcity, residual parts |
| Stage 3 | `our_method/postprocess_security.py` | Feasible postprocess dispatch and duals | Done | Enforce physical constraints and collect correction signals |
| Stage 4 | `our_method/generate_correction_feedback.py` | Correction feedback dataset | Done | Build primal/dual feedback for fine-tuning |
| Stage 5 | `our_method/run_dual_guided.py` | Residual dual-guided checkpoints | Done | Reduce correction burden if desired |
| Stage 6 | `our_method/evaluate_posted_price.py` | `eval_postprocess.csv` | Done | Evaluate pre/post dispatch, rent, cost, and feasibility |
| Stage 7 | `run_dual_guided.py` selection | `model_best_feasible_rent.pth` | Done | Select feasible low-rent checkpoint |
| Stage 8 | `run_dual_guided.py` selection | `model_best_correction.pth` | Legacy/secondary | Correction-focused checkpoint, not primary objective |

## 3. Checkpoint Naming

| Checkpoint | Meaning | Recommended use |
|---|---|---|
| `model_best_constr.pth` | Best Stage 1 constraint checkpoint | Main input for postprocess and dual-guided training |
| `model_initial.pth` | Stage 1 checkpoint copied into a dual-guided run | Baseline inside each fine-tuning run |
| `model_best.pth` | Best dual-guided training loss checkpoint | Training diagnostic |
| `model_best_feasible_rent.pth` | Best postprocess-feasible low-rent checkpoint | Main paper result checkpoint |
| `model_best_correction.pth` | Best correction-burden checkpoint | Secondary tradeoff result |
| `final_model.pth` | Last iteration checkpoint | Diagnostic only unless selected |

## 4. Completed Reference Results

These are current reference results after security postprocess.

| Method | Checkpoint | `mt_floor_gap_mwh` | `mt_offer_gap_mwh` | `positive_adjustment_mwh` | `positive_adjustment_payment` | `info_rent` | `procurement_cost` | Role |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Old baseline | `phase1a_20260426_192136/model_best_constr.pth:post` | 0.000009 | 1.6435 | 2.1047 | 107.89 | 222.82 | 2589.81 | Original reference |
| Price decomposition | `phase1a_20260427_111518/model_best_constr.pth:post` | 0.000007 | 1.1769 | 1.7019 | 86.86 | 251.67 | 2610.38 | Current feasible low-rent candidate |
| Dual-guided balanced | `dual_guided_20260427_134702/model_best.pth:post` | 0.000005 | 0.9044 | 1.5083 | 76.65 | 261.40 | 2616.43 | Balanced correction tradeoff |
| Dual-guided aggressive | `dual_guided_20260427_140437/model_best_correction.pth:post` | 0.000005 | 0.8419 | 1.4686 | 74.51 | 263.44 | 2617.93 | Correction-focused tradeoff |
| Feasible-rent selector | `dual_guided_20260427_140437/model_best_feasible_rent.pth:post` | 0.000007 | 1.1769 | 1.7019 | 86.86 | 251.67 | 2610.38 | Primary objective selector |

Interpretation:

```text
All postprocess results satisfy MT floor up to numerical tolerance.
Dual-guided learning reduces correction burden, but increases info rent.
For the primary objective, model_best_feasible_rent is the correct selector.
For the correction-burden analysis, model_best_correction is useful as a tradeoff point.
```

## 5. Main Comparison Experiments

| ID | Experiment | Method/checkpoint | Comparison target | Primary metrics | Status | Priority |
|---|---|---|---|---|---|---|
| M1 | Original posted-price baseline | Old `model_best_constr.pth` | Current method | Feasibility, rent, cost, correction | Partially done | High |
| M2 | Price decomposition only | New `model_best_constr.pth` | Old baseline | Feasibility, rent, correction | Done once | High |
| M3 | Feasible-rent selected method | `model_best_feasible_rent.pth` | M1, M2 | `mt_floor_gap`, `info_rent`, `procurement_cost` | Done once | High |
| M4 | Correction-focused dual-guided | `model_best_correction.pth` | M2, M3 | `mt_offer_gap`, positive adjustment, correction payment | Done once | Medium |
| M5 | Constrained social optimum | `baseline_social_opt_multi.py` | All methods | `social_cost_true`, dispatch gap | Done through evaluator | High |
| M6 | Pre vs post comparison | Each checkpoint `:pre` and `:post` | Same checkpoint | Constraint gap and correction burden | Available in CSV | High |

Expected paper table:

| Table | Rows | Columns |
|---|---|---|
| Main results | Old baseline, decomposition, feasible-rent, correction-focused, social optimum | Cost, social cost, rent, IR, MT floor gap, correction, dispatch gap |
| Pre/post feasibility | Selected methods, pre and post | MT floor gap, voltage/line status, slack, positive adjustment |
| Tradeoff result | Decomposition, balanced dual, aggressive dual | Rent vs correction burden |

## 6. Ablation Study

| ID | Ablation | Implementation idea | What it tests | Main metrics | Priority |
|---|---|---|---|---|---|
| A1 | No price decomposition | Use old price network style | Whether decomposition itself helps | Rent, correction, dispatch gap | High |
| A2 | Base plus type only | Disable security/scarcity components | Value of structured security/scarcity terms | Correction burden, dual pressure | High |
| A3 | Remove security component | Keep base/type/scarcity | Whether voltage/line dual guidance matters | Voltage dual, line dual, correction | Medium |
| A4 | Remove scarcity component | Keep base/type/security | Whether MT floor scarcity term matters | MT offer gap, MT correction | High |
| A5 | No residual dual-guided heads | Use decomposition only | Whether residual fine-tuning is needed | Rent and correction tradeoff | High |
| A6 | Residual-only vs train-all-heads | `--train-all-price-heads` on/off | Whether freezing base policy improves stability | Rent, cost, regret, correction | Medium |
| A7 | No dual targets | Remove dual alignment, keep regularization | Whether duals add useful information | Correction burden, dual metrics | Medium |
| A8 | No correction selection | Use `model_best.pth` only | Whether postprocess-aware selection matters | Feasible-rent score | High |
| A9 | Feasible-rent vs correction selector | Compare `model_best_feasible_rent.pth` and `model_best_correction.pth` | Economic objective tradeoff | Rent, cost, correction | High |

## 7. Dual-Guided Parameter Study

| ID | Parameters | Purpose | Expected effect | Priority |
|---|---|---|---|---|
| P1 | `scarcity_scale=1.0`, `lambda_mt_offer_gap=2.0`, `lambda_anchor=0.1` | Conservative dual-guided baseline | Small correction reduction, low drift | Medium |
| P2 | `scarcity_scale=1.5`, `lambda_mt_offer_gap=3.0`, `lambda_anchor=0.1` | Balanced correction reduction | Good correction reduction, moderate rent increase | High |
| P3 | `scarcity_scale=1.5`, `lambda_mt_offer_gap=5.0`, `lambda_anchor=0.2` | Strong guard with stronger anchor | More conservative than P2 | Medium |
| P4 | `scarcity_scale=2.0`, `lambda_mt_offer_gap=5.0`, `lambda_anchor=0.1` | Aggressive correction reduction | Lowest correction, highest rent | High for tradeoff figure |
| P5 | Vary `selection_info_rent_weight` | Sensitivity of feasible-rent selection | Changes selected checkpoint if rent/cost conflict | Medium |
| P6 | Vary `selection_offer_gap_weight` | Sensitivity of correction tie-breaker | Controls how much correction matters after feasibility | Medium |

Use these studies to produce a tradeoff plot:

```text
x-axis: information rent or procurement cost
y-axis: positive adjustment / MT offer gap / correction payment
point labels: parameter settings
```

## 8. Robustness Experiments

| ID | Experiment | What to vary | Output | Priority |
|---|---|---|---|---|
| R1 | Multi-seed evaluation | Evaluation seeds, e.g. 3 to 5 seeds | Mean and std of main metrics | High |
| R2 | Multi-seed training | Training seeds for Stage 1 and dual-guided | Stability of learned mechanism | Medium |
| R3 | More type samples | `--samples` in evaluation | Robustness to DER type draws | High |
| R4 | Load profile stress | Higher/lower load or peak stress | Feasibility and correction under stress | Medium |
| R5 | MT floor ratio stress | `ctrl_min_ratio` values | How hard MT floor affects rent/correction | High |
| R6 | Price buyback ratio | `pi_buyback_ratio` values | Sensitivity to outside option/floor price | Medium |
| R7 | Solver robustness | Different postprocess settings or tolerances | Verify postprocess feasibility is not numerical artifact | Medium |

Recommended reporting:

| Metric group | Report format |
|---|---|
| Main metrics | Mean plus std over seeds |
| Feasibility metrics | Mean, max, and number of violations |
| Solver status | Count of `optimal`, `optimal_inaccurate`, failures |
| Tradeoff metrics | Scatter plot or small table |

## 9. Strategic Behavior Analysis

| ID | Experiment | Design | Main metrics | Claim supported | Priority |
|---|---|---|---|---|---|
| S1 | Regret evaluation | Run best-response/misreport evaluation for selected checkpoints | Mean regret, max regret, IR violation | Individual strategic robustness | High |
| S2 | Pre vs post regret check | Compare regret before and after postprocess settlement | Regret, utility, payment | Postprocess does not create large strategic loophole | High |
| S3 | Single DER withholding | One DER reports less availability or higher cost | Cost, rent, feasibility, correction | Robustness to unilateral manipulation | Medium |
| S4 | Collusive withholding | Group of MT or controllable DERs reduce offers together | Procurement cost, slack, correction payment | Collusion stress test | High if claiming collusion resistance |
| S5 | Collusive price pressure | Group reports types to induce higher prices | Info rent, payment, regret | Limits of bid-independent pricing | Medium |
| S6 | Group size sweep | Coalition size 1, 2, 3, ... | Worst-case cost/rent increase | Collusion sensitivity curve | Medium |

Important wording for the paper:

```text
The bid-independent posted-price design reduces individual price manipulation channels.
Collusion robustness should be presented as stress-test evidence unless a formal
collusion-proof theorem is added.
```

## 10. Baseline Comparison Plan

| Baseline | Source | Purpose | Notes |
|---|---|---|---|
| Constrained social optimum | `baseline/baseline_social_opt_multi.py` | Efficiency lower bound | Not incentive compatible, no info rent comparison as mechanism |
| Old posted-price mechanism | Old run `phase1a_20260426_192136` | Direct prior-method baseline | Use same evaluation seeds |
| Current decomposition only | New run `phase1a_20260427_111518` | Main low-rent feasible method | Strong candidate for paper main method |
| Correction-focused dual-guided | Dual-guided run `140437` | Shows optional correction-burden reduction | Present as tradeoff rather than main objective |
| Feasible-rent selected method | `model_best_feasible_rent.pth` | Final automatic selector | Main checkpoint under current objective |

## 11. Paper Figures and Tables

| Figure/Table | Content | Required experiments |
|---|---|---|
| Table 1: Main results | Compare baseline, decomposition, feasible-rent, correction-focused, social optimum | M1 to M5 |
| Table 2: Feasibility | Pre/post MT floor gap, slack, status | M6, R1 |
| Table 3: Ablation | Remove decomposition/security/scarcity/residual/selection | A1 to A9 |
| Table 4: Strategic robustness | Regret, IR, withholding/collusion stress | S1 to S6 |
| Figure 1: Pipeline | Posted price, DER response, postprocess, dual feedback, selection | Existing framework |
| Figure 2: Price decomposition | Components over time/type | A2 to A5 |
| Figure 3: Rent-correction tradeoff | Info rent vs positive adjustment or MT offer gap | P1 to P4 |
| Figure 4: Robustness | Mean/std over seeds or stress levels | R1 to R6 |

## 12. Recommended Next Steps

| Step | Task | Output |
|---|---|---|
| 1 | Write `summarize_runs.py` | One CSV/Markdown table from multiple `eval_postprocess.csv` files |
| 2 | Re-evaluate main checkpoints with common seeds | Clean main comparison table |
| 3 | Run ablations A1, A4, A5, A8, A9 first | Minimal ablation set |
| 4 | Run robustness R1 and R3 | Mean/std result table |
| 5 | Run strategic tests S1 and S4 | Mechanism robustness evidence |
| 6 | Create paper-ready figures | Tradeoff and pipeline plots |
| 7 | Write experiment section | Method comparison, ablation, robustness, limitations |

## 13. Current Interpretation

The current framework is ready for systematic analysis. The most important
experimental distinction is:

```text
Feasibility objective:
  mt_floor_gap_mwh and postprocess_mt_slack_mwh should be near zero.

Economic objective:
  among feasible postprocess results, choose low info_rent and low procurement_cost.

Correction objective:
  mt_offer_gap_mwh and positive_adjustment_mwh measure how much postprocess
  had to compensate for the posted-price dispatch.
```

Therefore:

```text
Use model_best_feasible_rent.pth for the primary mechanism result.
Use model_best_correction.pth only to show the optional correction-burden tradeoff.
```

## 14. Paper Experiment Suite Draft

This section records the current paper-level experiment design. Compared with
the earlier checklist, this section is organized closer to the final experiment
section of the manuscript.

### 14.1 Experiment 1: Overall Performance Against Baselines

Goal:

```text
Show that the proposed mechanism achieves feasible dispatch with competitive
cost, low information rent, and reasonable efficiency compared with baseline
models.
```

| Item | Design |
|---|---|
| Main comparison | Proposed feasible-rent mechanism vs baseline models |
| Proposed method | `model_best_feasible_rent.pth` after security postprocess |
| Secondary proposed method | `model_best_correction.pth` to show correction-cost tradeoff |
| Baselines | Old posted-price baseline, constrained social optimum, possibly no-postprocess model, heuristic OPF or fixed-price method if available |
| Main table | One table comparing cost, rent, feasibility, correction, and dispatch efficiency |
| Main metrics | `procurement_cost`, `social_cost_true`, `info_rent`, `utility_min`, `mt_floor_gap_mwh`, `postprocess_mt_slack_mwh`, `positive_adjustment_mwh`, `dispatch_l1_gap_mwh` |
| Key claim | Our mechanism satisfies physical constraints after postprocess while preserving a reasonable economic objective |

Suggested output:

```text
Table: Overall Performance Comparison
Rows: Baseline mechanisms + proposed variants + social optimum
Columns: cost, rent, IR, feasibility, correction, dispatch gap
```

Important note:

```text
The constrained social optimum is not a strategic mechanism. It should be used
as an efficiency lower bound, not as a fair incentive-compatible baseline.
```

### 14.2 Experiment 2: In-Day Allocation Trajectory Comparison

Goal:

```text
Visualize how the proposed allocation changes over the 24-hour horizon and how
it differs from other methods.
```

| Item | Design |
|---|---|
| Figure type | Time-series allocation plot |
| X-axis | Hour `t = 1,...,24` |
| Y-axis | Dispatch or energy allocation |
| Curves | Proposed method, old baseline, constrained social optimum, possibly pre/post allocation |
| DER grouping | Plot by DER type: PV, WT, MT, maybe total DER and grid import |
| Main metrics shown | `x_pre`, `x_post`, `x_social`, `P_VPP`, MT dispatch, MT floor |
| Key claim | The proposed mechanism produces an in-day allocation closer to physically feasible and economically meaningful dispatch |

Suggested figures:

| Figure | Content |
|---|---|
| Fig. 2a | Total DER dispatch and grid import over 24h |
| Fig. 2b | MT dispatch vs MT floor over 24h |
| Fig. 2c | PV/WT/MT type-level allocation comparison |
| Fig. 2d | Pre-post allocation correction over time |

Implementation note:

```text
Use the same sampled DER type scenario across all methods.
For each checkpoint, save per-hour allocation arrays from evaluate_posted_price.py
or add a small plotting script that exports x_pre/x_post/x_social by time.
```

### 14.3 Experiment 3: Postprocess and Price Decomposition Analysis

Goal:

```text
Explain why the security postprocess is needed and what physical meaning the
price decomposition components learn.
```

This experiment combines ablation study and interpretability analysis.

| Part | Question | Design | Main metrics |
|---|---|---|---|
| Postprocess ablation | What happens without postprocess? | Compare `:pre` vs `:post` for each method | `mt_floor_gap_mwh`, voltage/line violation, slack, correction |
| No-uplift ablation | Is MT security uplift necessary? | Run evaluation with `--no-mt-security-uplift` | MT gap, infeasibility, correction failure |
| Price decomposition ablation | Which price component matters? | Remove or freeze security/scarcity/type components | rent, cost, correction, dual metrics |
| Residual dual-guided ablation | Does dual-guided residual help? | Compare decomposition-only vs residual dual-guided | correction burden and rent tradeoff |
| Component interpretation | Do components match physical signals? | Plot `rho_base`, `rho_type`, `rho_security`, `rho_scarcity` over time/type | correlation with MT floor dual, voltage dual, load |

Suggested output:

```text
Table: Ablation Results
Figure: Price Components Over Time
Figure: Pre/Post Correction Magnitude by Time and DER Type
```

Key interpretation:

```text
postprocess guarantees physical feasibility.
price decomposition makes the learned posted price more interpretable.
dual-guided residual learning can reduce correction burden but may increase rent.
```

### 14.4 Experiment 4: Strategic Behavior Stress Test

Goal:

```text
Analyze how different individual strategic behaviors affect cost, rent,
feasibility, and correction burden.
```

| Strategic behavior | Design | Main output |
|---|---|---|
| Cost over-reporting | One DER reports higher cost parameters | Cost/rent increase and regret |
| Capacity withholding | One DER reports lower available capacity | Feasibility, correction, procurement cost |
| Type misreport | One DER changes type-related report if applicable | Payment, utility, regret |
| Price pressure behavior | DER reports to induce higher payment | Info rent and procurement cost |
| Worst-response search | Use existing regret/best-response loop | Mean and max regret |

Main metrics:

```text
procurement_cost
info_rent
DER utility
regret_mean / regret_max
mt_floor_gap_mwh
postprocess_mt_slack_mwh
positive_adjustment_mwh
```

Suggested output:

| Output | Content |
|---|---|
| Table | Cost/rent/regret under each strategic behavior |
| Figure | Cost increase under different manipulation strengths |
| Figure | Utility or regret distribution across DER types |

Key claim:

```text
Because the price is bid-independent, individual DERs have limited ability to
manipulate the posted price. Stress tests should show how much cost or rent can
increase under different unilateral strategic behaviors.
```

### 14.5 Experiment 5: Coalition and Collusion Stress Test

Goal:

```text
Analyze how coalition behavior affects cost and feasibility, especially when
multiple controllable DERs coordinate.
```

| Coalition scenario | Design | Main metrics |
|---|---|---|
| MT coalition withholding | Multiple MT units reduce reported availability | cost, rent, MT slack, correction payment |
| Controllable DER coalition | MT/DG/DR group manipulates together | procurement cost and feasibility |
| Renewable withholding | PV/WT group withholds availability | grid import and correction |
| Coalition size sweep | Coalition size from 1 to K | worst-case cost/rent curve |
| Coalition location sweep | Coalition at different buses/locations | physical sensitivity and security impact |

Suggested output:

```text
Figure: Procurement cost increase vs coalition size
Figure: Info rent increase vs coalition size
Table: Worst-case coalition results by DER type
```

Important paper wording:

```text
Unless a formal collusion-proof theorem is added, this should be framed as a
collusion stress test rather than a full collusion-proof guarantee.
```

### 14.6 Experiment 6: Robustness Under Load Uncertainty

Goal:

```text
Evaluate whether the proposed mechanism remains feasible and economically stable
under uncertain load conditions.
```

| Load uncertainty setting | Design |
|---|---|
| Mild uncertainty | Load scale sampled from a narrow interval, e.g. `[0.95, 1.05]` |
| Medium uncertainty | Load scale sampled from `[0.90, 1.10]` |
| High uncertainty | Load scale sampled from `[0.80, 1.20]` |
| Peak stress | Increase selected peak-hour load only |
| Forecast error | Train/evaluate with mismatch between expected and realized load |

Main metrics:

```text
feasibility violation rate
mt_floor_gap_mwh
postprocess_mt_slack_mwh
procurement_cost
info_rent
positive_adjustment_mwh
grid_import_mwh
solver status counts
```

Suggested output:

| Output | Content |
|---|---|
| Figure | Cost and rent vs load uncertainty level |
| Figure | Feasibility violation/slack vs load uncertainty level |
| Table | Mean/std metrics across random load scenarios |

Key claim:

```text
The security postprocess provides a feasibility backstop under load uncertainty,
while the learned posted-price mechanism remains economically stable.
```

### 14.7 Experiment 7: Physical Sensitivity and Learned Topology Analysis

Goal:

```text
Analyze whether the learned price decomposition captures abstract physical
structure, such as bus/location sensitivity and OPF shadow-price patterns.
```

Main question:

```text
Do learned location-dependent price weights resemble physical OPF sensitivity
patterns across buses?
```

Potential analyses:

| Analysis | Design | Evidence |
|---|---|---|
| Bus-level price component map | Plot `rho_security` or security residual by bus/location | Spatial pattern of learned security price |
| Correlation with OPF duals | Correlate learned security/scarcity components with voltage/line/MT floor duals | Physical alignment |
| Sensitivity to injection | Compare learned bus weights with PTDF-like or voltage-sensitivity matrices | Topology-aware learning |
| Location perturbation | Add/remove injection at different buses and observe price/correction response | Physical sensitivity |
| Component clustering | Cluster DER/bus embeddings or price weights | Whether physically similar locations group together |
| Topology visualization | Overlay learned weights on feeder/bus graph | Interpretability figure |

Suggested output:

```text
Figure: Learned bus/location weight heatmap
Figure: OPF physical sensitivity vs learned security price component
Figure: Correlation scatter plot between dual-implied marginal values and learned price adders
```

Important interpretation:

```text
This experiment can support the claim that price decomposition does not merely
fit numerical targets, but learns an abstract representation of physical
network constraints.
```

Potential metrics:

```text
Pearson/Spearman correlation between learned price component and dual signal
rank correlation of bus importance
top-k overlap between high-sensitivity buses and high learned security weights
change in correction burden after perturbing high-weight vs low-weight buses
```

### 14.8 Proposed Experiment Order

| Order | Experiment | Reason |
|---|---|---|
| 1 | Overall baseline comparison | Establish headline performance |
| 2 | In-day allocation plots | Provide intuitive mechanism behavior |
| 3 | Postprocess/decomposition ablation | Explain where improvement comes from |
| 4 | Strategic behavior stress test | Support mechanism-design claim |
| 5 | Coalition stress test | Support collusion robustness discussion |
| 6 | Load uncertainty robustness | Support reliability under uncertainty |
| 7 | Physical sensitivity/topology analysis | Provide deeper interpretability and novelty |

Minimum experiment set for first paper draft:

```text
Experiment 1 + Experiment 2 + Experiment 3 + Experiment 4
```

Full experiment set for stronger submission:

```text
Experiment 1 through Experiment 7
```
