# Monte Carlo Orbital Conjunction

A Python numerical simulation package for two-body orbital propagation, bracket-cached Time of Closest Approach (TCA) root-finding, and parallelized Monte Carlo satellite collision probability ($P_c$) estimation.

## Example Output: TCA Root-Finding

To pinpoint the Time of Closest Approach (TCA), the solver tracks the range-rate indicator $D(t) = \mathbf{r}_{\text{rel}}(t) \cdot \mathbf{v}_{\text{rel}}(t)$ across the conjunction window, brackets sign changes where $D(t) = 0$, and refines the root via bisection to locate the minimum separation distance $d_{\min}$:

![Approach Indicator D(t) Across Conjunction Window](assets/D_of_t.png)
*Range-rate indicator $D(t) = \mathbf{r}_{\text{rel}}(t) \cdot \mathbf{v}_{\text{rel}}(t)$ evaluated over a $3,000\text{ s}$ two-body propagation window; zero-crossings ($D(t)=0$) bracket candidate local extrema in relative separation.*

## Overview

Evaluating collision risk between orbiting spacecraft and debris requires propagating uncertain state vectors forward in time, pinpointing the exact moment of closest approach, and estimating the probability that minimum separation falls below a combined hard-body radius ($R_{\text{HBR}}$).

This repository implements a modular numerical pipeline built from first principles:
* **Two-Body Orbital Dynamics (`orbital.dynamics`):** Cartesian gravitational equations of motion in Earth-Centered Inertial (ECI) coordinates.
* **4th-Order Runge-Kutta Integrator (`orbital.integrators`):** Fixed-step RK4 numerical solver with boundary-clamped time stepping to prevent final-step overshoot.
* **Cached TCA Bisection Solver (`orbital.conjunction`):** Detects sign changes in the range-rate indicator $D(t) = \mathbf{r}_{\text{rel}}(t) \cdot \mathbf{v}_{\text{rel}}(t) = 0$. Includes both a baseline solver (`score`) and a bracket-cached multi-root solver (`score_optimized`) that propagates from saved bracket endpoints rather than restarting from $t_0$ on every bisection iteration.
* **Parallel Monte Carlo & Importance Sampling (`orbital.probability`):** Evaluates rare-event collision probabilities across $N$ uncertain $12\text{D}$ joint initial states using `multiprocessing.Pool`, computing empirical $P_c$, sample variance, $95\%$ confidence intervals, and statistical relative error.

## Repository Structure

```text
monte-carlo-orbital-conjunction/
├── pyproject.toml
├── LICENSE
├── assets/
│   └── D_of_t.png
├── src/
│   └── orbital/
│       ├── dynamics.py       # Cartesian two-body equations of motion
│       ├── integrators.py    # Classical RK4 integrator and propagation loop
│       ├── conjunction.py    # Relative geometry, D(t) indicator, score(), and score_optimized()
│       └── probability.py    # Parallelized importance sampling & crude Monte Carlo estimator
└── examples/
    ├── runtime_score_comparison.py   # Speedup benchmark: baseline score() vs. score_optimized()
    └── mc_conjunction_symmetric.py   # 1-day conjunction Monte Carlo sample-size convergence study
```

## Quickstart

Clone the repository and install the package locally in editable mode:

```bash
git clone [https://github.com/Rayan36l/monte-carlo-orbital-conjunction.git](https://github.com/Rayan36l/monte-carlo-orbital-conjunction.git)
cd monte-carlo-orbital-conjunction
pip install -e .
```

Run the bracket-caching runtime benchmark:

```bash
python examples/runtime_score_comparison.py
```

Run the parallelized Monte Carlo collision probability study:

```bash
python examples/mc_conjunction_symmetric.py
```

## Research Context

These numerical solvers were developed during undergraduate astrodynamics research at the **Astrodynamics, Space, and Robotics Laboratory (ASRL)** at the University of Central Florida, focusing on rare-event satellite collision probability estimation and high-performance parallel Monte Carlo simulation on the UCF Stokes HPC cluster.
