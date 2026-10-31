# Kaggle Competition Submissions & Empirical Progression

This directory archives the progressive submissions evaluated on the official **NeurIPS 2026 Smart Buildings Competition** hidden evaluation benchmark (30 hidden test scenarios: 6 unseen buildings $\times$ 5 unseen weather stations, 8,640 steps each).

These empirical iterations directly correspond to **Table 1** and **Section 5** of the conference paper:
> **"Physics-Aware Hierarchical HVAC Control with Permutation-Invariant Residuals for Zero-Shot Building Transfer"**  
> *Author:* Tushar Bhardwaj ([ORCID: 0009-0007-6664-3235](https://orcid.org/0009-0007-6664-3235))  
> *Track:* Smart Buildings Competition Track at 40th Conference on Neural Information Processing Systems (NeurIPS 2026).

---

## Leaderboard Progression Summary

| Iteration | Submission Notebook / Script | Kaggle LB Score (MAE) | Advantage vs. Best Baseline | Key Architectural Levers |
| :---: | :--- | :---: | :---: | :--- |
| **Baseline** | Official `pid_water` | 0.05414 | 0.0% | Classical spatial mean water loop + coupled fan |
| **Iteration 1** | [`iteration_1_standby_decoupling_lb0.04165.py`](policies/iteration_1_standby_decoupling_lb0.04165.py) | **0.04165** | **-23.1%** | Per-zone PI band-centre tracking, unoccupied deep standby, decoupled fan pressure during heating. |
| **Iteration 2** | [`iteration_2_modulated_dp_coasting_lb0.04188.py`](policies/iteration_2_modulated_dp_coasting_lb0.04188.py) | **0.04188** | **-22.6%** | Soft pump differential pressure modulation ($1.5 F_{\text{hw}} - 1.0$) and initial end-of-day thermal coasting. |
| **Iteration 3** | [`iteration_3_preconditioning_hysteresis_lb0.04091.py`](policies/iteration_3_preconditioning_hysteresis_lb0.04091.py) | **0.04091** | **-24.4%** | Dynamic preconditioning window and heating hysteresis to eliminate setpoint chatter. |
| **Iteration 4** | [`iteration_4_multiregime_cooling_hold_lb0.01540.py`](policies/iteration_4_multiregime_cooling_hold_lb0.01540.py) | **0.01540** | **-71.6%** | 24-hour nocturnal cooling-run holding (288 steps) in hot climates, preventing envelope collapse. |
| **Run 5 (Master)** | [`iteration_5_climate_adaptive_master_lb0.01507.py`](policies/iteration_5_climate_adaptive_master_lb0.01507.py) | **0.01507** | **-72.2%** | Continuous cyclic tracking ($\theta_{\text{hod}} = \text{atan2}$), demand Floating SAT ($+0.1 \to -0.7$), subzero hydronic reset floor ($F_{\text{hw}} \ge (283.15 - T_{\text{OAT}})/32.0$), and climate-adaptive damper anti-windup clamping. |

---

## Directory Structure

```
kaggle/
├── README.md                      # This documentation and benchmark mapping
├── notebooks/                     # Exact Jupyter Notebooks submitted to Kaggle
│   ├── iteration_1_standby_decoupling_lb0.04165.ipynb
│   ├── iteration_2_modulated_dp_coasting_lb0.04188.ipynb
│   ├── iteration_3_preconditioning_hysteresis_lb0.04091.ipynb
│   ├── iteration_4_multiregime_cooling_hold_lb0.01540.ipynb
│   └── iteration_5_climate_adaptive_master_lb0.01507.ipynb
└── policies/                      # Standalone, importable Python policies for peer review
    ├── iteration_1_standby_decoupling_lb0.04165.py
    ├── iteration_2_modulated_dp_coasting_lb0.04188.py
    ├── iteration_3_preconditioning_hysteresis_lb0.04091.py
    ├── iteration_4_multiregime_cooling_hold_lb0.01540.py
    └── iteration_5_climate_adaptive_master_lb0.01507.py
```

---

## How to Inspect and Run Any Policy

Each policy file in `policies/` is self-contained with its embedded baseline dependencies and inference server. 

To run or evaluate any specific policy using the benchmark evaluation server:
```python
from kaggle.policies.iteration_5_climate_adaptive_master_lb0.01507 import (
    TransferPolicy,
    SmartBuildingsInferenceServer,
)

# Launch inference server with the Master policy
SmartBuildingsInferenceServer(TransferPolicy).run()
```
