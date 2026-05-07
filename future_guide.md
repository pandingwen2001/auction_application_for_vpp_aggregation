# Future Guide: Security-Correction-Aware Neural Posted-Price Mechanism

Date: 2026-04-27

This note records the next algorithmic direction for the `original/` codebase. The goal is to make the current posted-price procurement framework stronger than a simple learned price rule, while avoiding a menu-based design that would move the paper too close to contract-style mechanisms.

---

## 1. Current concern

The current framework has become much more interpretable than the earlier end-to-end neural auction + differentiable OPF design:

```text
public state
  -> neural posted price rho
  -> DER price response / offer cap
  -> preliminary dispatch
  -> OPF-based post process
  -> final settlement at rho
```

However, if we describe it only as "a neural network learns a price and then an OPF fixes the dispatch", the paper may look algorithmically thin for journals such as TSG/TPS.

The next algorithmic upgrade should therefore turn the current framework into:

```text
a security-correction-aware neural posted-price procurement mechanism
```

The key idea is not to add a more complicated contract/menu. Instead, we should make the learned posted price physically meaningful and explicitly aware of the security-correction burden.

---

## 2. Main algorithmic direction

The recommended direction is to combine two ideas:

```text
1. correction-aware price learning
2. shadow-price-guided posted-price formation
```

Individually:

```text
correction-aware learning
  - directly addresses the current MT-floor correction issue
  - gives rho_MT a clear learning direction
  - relatively easy to implement
  - may look like an engineering refinement if used alone

shadow-price-guided price formation
  - more power-system-aware
  - more attractive for TSG/TPS
  - gives the neural price physical interpretation
  - needs correction-aware learning to become a complete algorithmic loop
```

Together, they create a stronger algorithmic story:

```text
price -> DER response -> security correction -> primal/dual feedback -> better price
```

---

## 3. Important design rule: keep price bid-independent

The posted price used in the mechanism should remain bid-independent.

Do not do this:

```text
DER reports bids
  -> preliminary dispatch
  -> postprocess OPF duals
  -> current rho is modified using current bid-dependent duals
```

This would make the price indirectly depend on DER reports and would weaken the incentive-compatibility story.

Instead, use OPF correction signals only as training feedback or teacher signals:

```text
training data / public state
  -> preliminary price and dispatch
  -> postprocess correction
  -> collect correction residuals and duals
  -> train the price network to predict better future prices
```

At deployment time, the mechanism remains:

```text
public state -> rho_theta(public state)
DER response -> dispatch -> security correction
```

This preserves the key structural advantage:

```text
rho does not depend on an individual DER's own bid.
```

---

## 4. Proposed structured price formation

Instead of treating the posted price as a fully black-box score:

```text
rho_i,t = NN_theta(s)_i,t
```

use a semi-structured decomposition:

```text
rho_i,t =
    rho_base_t
  + rho_type_i,t
  + rho_security_i,t
  + rho_scarcity_i,t
```

Interpretation:

```text
rho_base_t
  - base energy signal
  - should track DA price, grid import cost, and system load level

rho_type_i,t
  - type-specific adjustment
  - captures PV/WT/DG/MT cost and availability differences

rho_security_i,t
  - network/security value
  - should respond to line/voltage congestion and location-dependent sensitivities

rho_scarcity_i,t
  - controllable-resource scarcity signal
  - should respond to MT floor pressure and shortage of controllable generation
```

This makes the method more than "NN predicts price"; it becomes a learned price-formation rule with power-system structure.

---

## 5. Correction-aware learning

Current issue:

```text
rho_MT too low
  -> MT offer is insufficient
  -> preliminary dispatch violates MT floor
  -> postprocess must add MT output

rho_MT too high
  -> MT payment and information rent increase
```

The algorithm should learn the balance between these two forces.

The learning objective should therefore include postprocess-aware terms:

```text
final procurement cost after postprocess
information rent
regret / IR penalties
MT floor gap before postprocess
postprocess positive adjustment
MT security uplift
additional payment caused by correction
dispatch deviation from constrained social optimum
security slack
```

The most important conceptual change is:

```text
The price policy should be trained with feedback from the security-correction layer,
not only from preliminary dispatch.
```

---

## 6. Shadow-price guidance

The postprocess OPF/QP contains useful physical signals. In addition to primal correction quantities such as `x_post - x_pre`, we should collect dual variables from active constraints.

Useful dual signals:

```text
power balance dual
MT floor dual
line upper/lower duals
voltage upper/lower duals
possibly ESS/SOC duals later
```

These duals can serve as teacher signals for the security/scarcity components of price:

```text
MT floor dual high
  -> controllable generation is scarce
  -> rho_scarcity for MT should increase

line/voltage dual high
  -> certain locations/resources have higher security value
  -> rho_security should increase for DERs with helpful network sensitivities

power balance dual high
  -> energy is scarce in that time period
  -> rho_base should increase
```

A possible alignment loss:

```text
security/scarcity adder predicted by NN
  should align with
dual-implied marginal security value
```

This gives the learned price a power-system interpretation and makes the contribution more suitable for TSG/TPS.

---

## 7. Lightweight implementation plan

Avoid implementing a fully differentiable postprocess QP immediately. A safer first version is a two-stage or alternating training pipeline.

### Stage 1: Train current posted-price network

Use the current pipeline:

```text
sample types/load
rho_theta(public state)
DER response
preliminary OPF dispatch
loss and monitors
```

Output:

```text
trained checkpoint
history.csv
preliminary dispatch statistics
```

### Stage 2: Run postprocess and collect correction data

For training/evaluation samples, run:

```text
x_pre, P_VPP_pre, rho, offer_cap
  -> SecurityPostProcessor
  -> x_post, P_VPP_post, slack, status
```

Collect:

```text
x_post - x_pre
positive_adjustment
MT uplift
additional payment
postprocess slack
solver status
dual variables from QP constraints
```

### Stage 3: Fine-tune price network with correction feedback

Use the collected correction data to train or fine-tune the price network:

```text
minimize
  postprocessed feasible procurement cost
  + correction magnitude
  + additional correction payment
  + information rent
  + regret/IR penalties
  + dual-alignment loss
```

The fine-tuned price should reduce future postprocess burden while avoiding excessive information rent.

---

## 8. Possible loss terms

A possible future training objective:

```text
L =
  L_procurement_post
+ lambda_rent * L_info_rent
+ lambda_regret * L_regret
+ lambda_ir * L_IR
+ lambda_corr * ||x_post - x_pre||_1
+ lambda_uplift * MT_uplift
+ lambda_addpay * additional_payment
+ lambda_slack * security_slack
+ lambda_dual * L_dual_align
```

Where:

```text
L_procurement_post
  = final payment to DERs + grid import cost after postprocess

L_dual_align
  = alignment between predicted security/scarcity adders
    and OPF dual-implied marginal values
```

Do not overcomplicate this at first. Start with:

```text
correction magnitude
MT uplift
additional payment
MT floor dual alignment
```

Then add line/voltage duals only if the current cases show meaningful congestion.

---

## 9. What to implement first

Recommended near-term order:

1. Modify `postprocess_security.py` to expose dual variables.
2. Add postprocess correction summary by time and DER type.
3. Add a script to generate a correction-feedback dataset from a trained checkpoint.
4. Modify `PostedPriceNetworkMulti` to output decomposed price components:

```text
rho_base
rho_type
rho_security
rho_scarcity
rho_total
```

5. Add correction-aware fine-tuning:

```text
use correction residuals and dual targets
to train security/scarcity adders
```

6. Evaluate whether the new price reduces:

```text
pre MT floor gap
postprocess positive adjustment
additional payment
dispatch gap to constrained social optimum
```

without causing a large increase in:

```text
information rent
procurement cost
regret
IR violation
```

---

## 10. How to frame this in the paper

Do not frame the method as:

```text
machine learning predicts a price and OPF fixes it
```

Frame it as:

```text
a security-correction-aware neural posted-price procurement mechanism
```

Suggested description:

```text
We propose a bid-independent neural posted-price procurement mechanism whose price formation is guided by the security-correction layer. The mechanism preserves a clean incentive structure through bid-independent prices, while OPF correction residuals and shadow prices provide physical feedback for learning security-aware price adders.
```

The key contribution is the closed loop:

```text
posted price
  -> DER self-selection
  -> preliminary dispatch
  -> OPF security correction
  -> primal/dual correction feedback
  -> improved posted-price formation
```

This is more algorithmically meaningful than directly tuning an MT penalty, and it avoids moving toward menu-based contract mechanisms.

---

## 11. Working hypothesis

The current 1.5 MWh MT-floor gap should not be treated as the central problem by itself. It should be treated as evidence that a pure price-response stage does not automatically internalize all security requirements.

The algorithmic goal is:

```text
learn prices that reduce the required security correction
while controlling procurement cost and information rent.
```

If successful, the improved method should show:

```text
smaller pre-dispatch MT gap
smaller postprocess uplift
lower additional payment
similar or lower information rent
similar regret/IR performance
better dispatch proximity to constrained social optimum
```

This would make the framework much stronger for a TSG/TPS-style paper.
