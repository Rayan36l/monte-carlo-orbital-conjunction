import os
from multiprocessing import Pool
from typing import Callable
import numpy as np


def importance_sampling(
    target_distribution,
    proposal_distribution,
    score_function: Callable,
    threshold: float,
    num_samples: int,
    rng: np.random.Generator,
    return_details: bool = False,
    num_workers: int = None,
) -> dict:
    """
    Estimate an event probability using parallelized importance sampling.
    When proposal_distribution == target_distribution, all weights equal 1
    and the estimator reduces to crude Monte Carlo.
    """
    samples = proposal_distribution.rvs(size=num_samples, random_state=rng)

    if num_workers is None:
        num_workers = os.cpu_count()

    with Pool(num_workers) as pool:
        scores = np.array(pool.map(score_function, samples))

    log_weights = target_distribution.logpdf(samples) - proposal_distribution.logpdf(samples)
    weights = np.exp(log_weights)

    indicators = scores < threshold
    weighted_indicators = weights * indicators

    probability = float(np.mean(weighted_indicators))
    variance = float(np.var(weighted_indicators, ddof=1))

    standard_error = np.sqrt(variance / num_samples)
    ci_half_width = 1.96 * standard_error
    ci_lower = probability - ci_half_width
    ci_upper = probability + ci_half_width

    if probability > 0.0:
        relative_error = float(ci_half_width / probability)
    else:
        relative_error = np.inf

    result = {
        "probability": probability,
        "variance": variance,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "relative_error": relative_error,
        "num_workers": num_workers,
    }

    if return_details:
        result["num_events"] = int(np.count_nonzero(indicators))
        result["samples"] = samples
        result["scores"] = scores

    return result
