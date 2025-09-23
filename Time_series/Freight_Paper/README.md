# Spot Freight Market Dynamics — USA

## Overview
This project explores the **dynamics of the U.S. spot freight market at the Market Area level**.  
We investigate the interplay between **freight rates**, **market capacity**, and **fuel costs**, focusing on how endogenous relationships drive short-run and long-run adjustments.

The analysis uses multiple econometric strategies to test robustness and address endogeneity, ultimately relying on a **Panel VAR(2)** specification to disentangle the feedback loop between **rates** and **capacity**.

---

## Goals
- Understand **how market capacity and fuel shocks affect spot rates**.
- Test for **endogeneity** between rate and capacity.
- Compare model performance across multiple specifications.
- Provide **impulse response functions (IRFs)** to interpret market dynamics.

---

## Data Pipeline
1. **Raw Inputs**  
   - Weekly panel dataset at the *Market Area* level.  
   - Variables: `rate (r)`, `capacity (c)`, `fuel (f)`.

2. **Preprocessing**  
   - Panel alignment (`id`, `week`).
   - Handling missing data (drop / impute strategies).
   - Stationarity testing (ADF per id).
   - Differencing for unit root correction: `r_d`, `c_d`, `f_d`.

3. **Panel Structuring**  
   - Balanced panel checks.
   - Group-wise time ordering.
   - Optional within-demeaning to control for unit fixed effects.

---

## Model Pipeline
We run several model families in sequence to test assumptions:

1. **Baseline OLS**
   - Simple linear regression of rate on capacity and fuel.
   - Diagnostic tests: residual plots, QQ-plots, Durbin–Watson statistic.
   - Identified potential **endogeneity bias**.

2. **System GMM**
   - Dynamic panel estimator with lagged instruments.
   - Hansen and Arellano–Bond tests for instrument validity and serial correlation.
   - Short time dimension limited power.

3. **Panel VAR(2)**
   - Addressed endogeneity by jointly modeling `rate`, `capacity`, and `fuel`.
   - Lag order fixed at **p=2** for comparability with System GMM.
   - Stability checks via unit circle plots.
   - **Impulse Response Functions (IRFs)**  to interpret market dynamics.

---

## Tests & Validation
- **Stationarity**: Augmented Dickey–Fuller (ADF) tests by KMA market area.  
- **Causality**: Pairwise Granger causality tests.  
- **Model stability**: Inverse roots of characteristic polynomial (unit circle test).  
- **Residual diagnostics**: Serial correlation and distribution checks.  
- **Robustness**: Comparison of OLS, GMM, and VAR outcomes.

---

## Key Insight
- **VAR(2) framework successfully handles the endogeneity** between **rates** and **capacity**, enabling meaningful interpretation of shocks.  
- IRFs show how a capacity shock propagates into rate adjustments (and vice versa), while fuel shocks act as exogenous cost-push drivers.
- While VAR(2) models allow for **Impulse Response Functions (IRFs)** to trace structural dynamics, tree-based ML models such as **LightGBM** do not naturally provide IRFs.  


## Why IRFs Don’t Apply to LightGBM
- **IRFs require linear dynamics**: In VAR, the system is defined as  
$$
  y_t = A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t
$$
where shocks $u_t$ can be propagated forward through the lag matrices $A_i$.
- **LightGBM is non-linear**: It uses gradient-boosted decision trees, with no explicit \(A_i\) matrices or linear innovations.
- Therefore, **structural IRFs (orthogonalized or generalized) are not defined for LightGBM**.

## When to Use Econometrics and ML techniques

- **Use VAR(2) IRFs** when the goal is *causal inference* and *structural dynamics*.  
- **Use LightGBM pseudo-IRFs** when the goal is *forecasting performance* and *scenario testing*.  

Together, they provide complementary insights:

- **Econometric models** → identification & theory  
- **ML models** → predictive power & nonlinear scenario analysis  

---

## Next Steps (if needed:)
- Expand model to allow **heterogeneous dynamics across market areas**.  
- Test alternative identification strategies (e.g., structural VAR with economic restrictions).  
- Automate pipeline for weekly updates.

---

