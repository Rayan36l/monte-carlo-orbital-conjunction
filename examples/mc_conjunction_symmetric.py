import os
import time
import numpy as np
from scipy.stats import multivariate_normal

from orbital.conjunction import score_optimized
from orbital.dynamics import two_body
from orbital.integrators import rk4
from orbital.probability import importance_sampling


# Nominal initial states at t0 = 0 s (km and km/s)
state1_0 = np.array([
    3125.707941035216,
    6263.380868543334,
    0.0,
    -6.751973369989614,
    3.369537385195492,
    0.0,
])

state2_0 = np.array([
    3137.049638644246,
    6266.996798520096,
    0.002234817599464,
    -6.737865492326000,
    3.377756281886220,
    -0.000004815958344,
])

# Propagation and conjunction window parameters (symmetric [-100, +100] s around 86,400 s)
t0 = 0.0
t_tca_nominal = 86400.0
t_start = 86300.0
t_end = 86500.0
h = 25.0
tol = 0.001
R_HBR = 0.01  # 10 m combined hard-body radius [km]


def conjunction_score(sample: np.ndarray) -> float:
    """
    Evaluate minimum miss distance d_min [km] for one 12D joint initial state sample.
    Defined at module level so multiprocessing worker processes can pickle it.
    """
    state1 = sample[:6]
    state2 = sample[6:]
    _, d_min, _ = score_optimized(
        rk4,
        two_body,
        t0,
        state1,
        state2,
        t_start,
        t_end,
        h,
        tol,
    )
    return d_min


def main():
    # Verify nominal deterministic conjunction first
    t_tca, d_min, _ = score_optimized(
        rk4,
        two_body,
        t0,
        state1_0,
        state2_0,
        t_start,
        t_end,
        h,
        tol,
    )

    print("Nominal Conjunction Check")
    print("-" * 40)
    print(f"Nominal TCA input : {t_tca_nominal:.6f} s")
    print(f"Refined TCA       : {t_tca:.6f} s")
    print(f"Minimum distance  : {d_min:.9f} km")
    print(f"Window            : [{t_start:.3f}, {t_end:.3f}] s")

    if not (t_start <= t_tca <= t_end):
        raise ValueError("Refined nominal TCA is outside the conjunction window.")

    # Construct 12x12 joint initial-state covariance matrix
    sigma_r = 5.0e-5  # km
    sigma_v = 1.0e-8  # km/s

    P1 = np.diag([sigma_r**2] * 3 + [sigma_v**2] * 3)
    P2 = np.diag([sigma_r**2] * 3 + [sigma_v**2] * 3)

    P = np.zeros((12, 12))
    P[:6, :6] = P1
    P[6:, 6:] = P2

    mean = np.concatenate((state1_0, state2_0))

    # Crude Monte Carlo setup: proposal distribution q(x) equals target p(x)
    p = multivariate_normal(mean=mean, cov=P)
    q = p

    sample_sizes = [100, 1_000, 10_000]
    seed = 42
    rows = []

    for num_samples in sample_sizes:
        print(f"\nRunning N = {num_samples:,}...")
        rng = np.random.default_rng(seed)

        start = time.perf_counter()
        result = importance_sampling(
            target_distribution=p,
            proposal_distribution=q,
            score_function=conjunction_score,
            threshold=R_HBR,
            num_samples=num_samples,
            rng=rng,
            return_details=False,
        )
        runtime = time.perf_counter() - start
        ci_width = result["ci_upper"] - result["ci_lower"]

        rows.append({
            "num_samples": num_samples,
            "probability": result["probability"],
            "variance": result["variance"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "ci_width": ci_width,
            "relative_error": result["relative_error"],
            "num_workers": result["num_workers"],
            "runtime": runtime,
        })
        print(f"Finished N = {num_samples:,} in {runtime:.3f} s ({result['num_workers']} cores)")

    # Print formatted results table
    print("\nMonte Carlo Sample-Size Study")
    print("=" * 124)
    header = (
        f"{'Samples N':>12}"
        f"{'Pc':>16}"
        f"{'Variance':>16}"
        f"{'CI Lower':>16}"
        f"{'CI Upper':>16}"
        f"{'CI Width':>16}"
        f"{'Rel. Error':>16}"
        f"{'Runtime (s)':>16}"
    )
    print(header)
    print("-" * 124)
    for row in rows:
        print(
            f"{row['num_samples']:>12,}"
            f"{row['probability']:>16.6e}"
            f"{row['variance']:>16.6e}"
            f"{row['ci_lower']:>16.6e}"
            f"{row['ci_upper']:>16.6e}"
            f"{row['ci_width']:>16.6e}"
            f"{row['relative_error']:>16.6e}"
            f"{row['runtime']:>16.3f}"
        )
    print("=" * 124)


if __name__ == "__main__":
    main()
