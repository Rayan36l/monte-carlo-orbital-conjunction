import time
import numpy as np
from orbital.integrators import rk4
from orbital.dynamics import two_body
from orbital.conjunction import score, score_optimized


def format_float(value, digits=6):
    """Format values for clean table output."""
    if value is None:
        return "failed"
    return f"{value:.{digits}g}"


def main():
    mu = 398600.4415
    r0 = 7000.0
    vc = np.sqrt(mu / r0)

    state1_0 = np.array([7000.0, 0.0, 0.0, 0.0, vc, 0.0])
    state2_0 = np.array([7005.0, 0.0, 20.0, 0.05, vc + 0.01, -0.03])

    t0 = 0.0
    h = 10.0
    tol = 1e-6

    # Propagation times chosen so a conjunction occurs within +/- 500 s
    propagation_times = [103.0, 17613.0, 41036.0, 82027.0]
    window_half_width = 500.0
    rows = []

    for propagation_time in propagation_times:
        t_start = max(t0, propagation_time - window_half_width)
        t_end = propagation_time + window_half_width

        # Benchmark original score()
        start = time.perf_counter()
        try:
            old_t_tca, old_d_min, old_iters = score(
                rk4, two_body, t0, state1_0, state2_0, t_start, t_end, h, tol
            )
            old_runtime = time.perf_counter() - start
            old_success = True
        except ValueError as error:
            old_runtime = time.perf_counter() - start
            old_t_tca = old_d_min = old_iters = None
            old_success = False
            print(f"Old score failed at t={propagation_time}: {error}")

        # Benchmark cached score_optimized()
        start = time.perf_counter()
        try:
            new_result = score_optimized(
                rk4,
                two_body,
                t0,
                state1_0,
                state2_0,
                t_start,
                t_end,
                h,
                tol,
                return_details=True,
            )
            new_runtime = time.perf_counter() - start
            new_t_tca = new_result["t_tca"]
            new_d_min = new_result["d_min"]
            new_success = True
        except ValueError as error:
            new_runtime = time.perf_counter() - start
            new_t_tca = new_d_min = None
            new_success = False
            print(f"Optimized score failed at t={propagation_time}: {error}")

        speedup = (old_runtime / new_runtime) if (old_success and new_success) else None

        rows.append({
            "t_start": t_start,
            "old_runtime": old_runtime if old_success else None,
            "new_runtime": new_runtime if new_success else None,
            "speedup": speedup,
            "old_d_min": old_d_min,
            "new_d_min": new_d_min,
        })

    # Print Markdown comparison table
    print("\n| t_start (s) | Old Runtime (s) | Optimized Runtime (s) | Speedup | Old d_min (km) | Optimized d_min (km) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['t_start']:.0f} "
            f"| {format_float(row['old_runtime'], 4)} "
            f"| {format_float(row['new_runtime'], 4)} "
            f"| {format_float(row['speedup'], 3)}x "
            f"| {format_float(row['old_d_min'], 6)} "
            f"| {format_float(row['new_d_min'], 6)} |"
        )


if __name__ == "__main__":
    main()
