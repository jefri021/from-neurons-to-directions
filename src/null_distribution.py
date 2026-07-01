"""
null_distribution.py
---------------------
Addendum to safety_neurons.py / metrics.py: builds theoretical and
empirical null distributions for cosine similarity, so that raw values
reported in Sections 5.2 and 5.7.1 can be judged against chance rather
than read as absolute magnitudes.

Two nulls are provided:

  1. THEORETICAL — closed-form SD for cosine similarity between two
     independent random unit vectors in R^d. Fast, no extra compute,
     but assumes isotropy (uniform spread on the hypersphere), which
     real weight/activation spaces may not satisfy.

  2. EMPIRICAL — ablate K randomly selected (non-safety) neurons,
     recompute the refusal direction, and measure its cosine similarity
     with the original. Repeated across seeds. This is model-specific
     and does not assume isotropy — it is the rigorous version of the
     "random neurons" baseline already used for variance_explained()
     in Experiment 1, extended to Experiment 2's ablation setting.

Typical usage — add to notebook 04 after computing the original
direction-survival results:

    from null_distribution import (
        theoretical_null_sd, z_score, sample_random_neurons,
        empirical_ablation_null
    )

    # Quick theoretical check
    d = get_hidden_size(instruct_model)
    sd = theoretical_null_sd(d)
    print(f"Theoretical null SD: {sd:.4f}")
    print(f"Layer 16 z-score: {z_score(0.034, sd):.2f}")
    print(f"Mean z-score: {z_score(0.082, sd):.2f}")

    # Rigorous empirical check (reuses your own pipeline)
    null_results = empirical_ablation_null(
        model=instruct_model,
        tokenizer=tokenizer,
        prompts=harmful_prompts + harmless_prompts,
        safety_neurons=top_neurons,
        original_directions=original_refusal_directions,  # dict[layer -> r]
        recompute_direction_fn=compute_refusal_direction,  # your own function
        n_seeds=10,
    )
"""

import random
import numpy as np
import torch
from typing import Callable, Optional

from model_utils import get_num_layers, get_intermediate_size
from safety_neurons import collect_activations_with_neuron_ablation
from metrics import cosine_similarity_1d


# ═════════════════════════════════════════════════════════════════════════════
# 1. Theoretical null
# ═════════════════════════════════════════════════════════════════════════════

def theoretical_null_sd(dim: int) -> float:
    """
    Exact standard deviation of cosine similarity between two independent
    vectors drawn uniformly at random from the unit hypersphere in R^dim.
    Mean is exactly 0 by symmetry. SD = 1/sqrt(dim) for any dim >= 2.

    For Qwen2.5-7B, dim = hidden_size = 3584 -> SD ~= 0.0167.
    """
    return 1.0 / np.sqrt(dim)


def z_score(cosine_sim: float, null_sd: float) -> float:
    """
    Number of standard deviations `cosine_sim` lies from the null mean (0).
    Use this instead of reading raw cosine similarity in isolation —
    it is what makes 0.066 (Section 5.2) and 0.082 (Section 5.7.1)
    directly comparable.
    """
    return cosine_sim / null_sd


# ═════════════════════════════════════════════════════════════════════════════
# 2. Empirical null — random-neuron ablation baseline
# ═════════════════════════════════════════════════════════════════════════════

def sample_random_neurons(
    model,
    k: int,
    exclude: Optional[list[tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> list[tuple[int, int]]:
    """
    Sample k random (layer_idx, neuron_idx) pairs uniformly across the
    whole model, matched in count to a safety neuron set, for use as a
    baseline. Same output format as get_top_safety_neurons().

    Args:
        model:   the transformer model (for layer count / intermediate size)
        k:       number of neurons to sample (match len(safety_neurons))
        exclude: pairs to exclude — pass your safety neuron set itself,
                 so the baseline never accidentally re-samples a real one
        seed:    RNG seed; vary this across repeated calls to build a
                 distribution rather than a single point estimate

    Returns:
        list of (layer_idx, neuron_idx) tuples
    """
    rng = random.Random(seed)
    n_layers = get_num_layers(model)
    intermediate_size = get_intermediate_size(model)
    exclude_set = set(exclude) if exclude else set()

    # Sampling without materializing the full (layer x neuron) list when
    # the model is large: rejection-sample instead of building all pairs.
    chosen: set[tuple[int, int]] = set()
    while len(chosen) < k:
        l = rng.randrange(n_layers)
        n = rng.randrange(intermediate_size)
        if (l, n) not in exclude_set and (l, n) not in chosen:
            chosen.add((l, n))

    return list(chosen)


def empirical_ablation_null(
    model,
    tokenizer,
    prompts: list[str],
    safety_neurons: list[tuple[int, int]],
    original_directions: dict[int, torch.Tensor],
    recompute_direction_fn: Callable,
    layers: Optional[list[int]] = None,
    n_seeds: int = 10,
) -> dict[int, list[float]]:
    """
    Build an empirical null distribution for "direction survival after
    ablation" by repeatedly ablating K random (non-safety) neurons and
    recomputing the refusal direction, exactly as done for the real
    safety-neuron ablation in Section 5.7.1.

    This directly answers: does ablating safety neurons destroy the
    refusal direction MORE than ablating any K neurons would? It is the
    ablation-experiment analogue of the random-neuron baseline already
    used for variance_explained() in Experiment 1 — same logic, applied
    to the recompute-then-compare pipeline instead.

    Args:
        safety_neurons:        your real top-k safety neurons (excluded
                                from random sampling, see sample_random_neurons)
        original_directions:   dict[layer_idx -> unit vector r], the
                                ORIGINAL (pre-ablation) refusal direction
                                per layer, as already computed in 5.7.1
        recompute_direction_fn: your own function that takes cached
                                activations (harmful/harmless) and returns
                                a per-layer direction dict — i.e. whatever
                                you call in refusal_direction.py to go
                                from activations to r. Plug it in here;
                                signature will depend on your own code.
        n_seeds:                number of random neuron sets to test
                                (10 is a reasonable default; more = tighter
                                estimate of the null, at proportional cost)

    Returns:
        dict mapping layer_idx -> list of cosine similarities (one per
        seed), i.e. the empirical null distribution at each layer.
        Compare these against your real safety-neuron-ablation values
        with a simple percentile check (see summarize_null below).
    """
    if layers is None:
        layers = list(original_directions.keys())

    k = len(safety_neurons)
    null_per_layer: dict[int, list[float]] = {l: [] for l in layers}

    for seed in range(n_seeds):
        random_neurons = sample_random_neurons(
            model, k=k, exclude=safety_neurons, seed=seed
        )

        # Reuses your existing ablation + activation collection exactly
        # as in Section 5.7.1, just with random_neurons instead of
        # top_neurons.
        ablated_acts = collect_activations_with_neuron_ablation(
            model, tokenizer, prompts, random_neurons, layers=layers
        )

        # >>> Plug in your own activations -> refusal direction step here.
        # This should mirror exactly what you did to get the *ablated*
        # direction in 5.7.1, just fed `ablated_acts` from this random
        # ablation instead of the safety-neuron ablation.
        ablated_directions = recompute_direction_fn(ablated_acts)

        for l in layers:
            sim = cosine_similarity_1d(
                original_directions[l], ablated_directions[l]
            )
            null_per_layer[l].append(sim)

        print(f"  seed {seed+1}/{n_seeds} done "
              f"(mean sim so far: "
              f"{np.mean([s for vals in null_per_layer.values() for s in vals]):.4f})")

    return null_per_layer


def summarize_null(
    null_per_layer: dict[int, list[float]],
    observed_per_layer: dict[int, float],
) -> None:
    """
    Compare the real safety-neuron-ablation survival values against the
    empirical random-neuron-ablation null, layer by layer.

    Prints, for each layer: the null mean +/- SD, the observed value,
    and an empirical z-score / percentile — i.e. "is safety-neuron
    ablation more destructive than random ablation of the same size?"
    """
    print(f"{'Layer':>6} | {'Null mean':>10} | {'Null SD':>8} | "
          f"{'Observed':>9} | {'z-score':>8} | {'Below null min?':>16}")
    print("-" * 70)

    for l in sorted(null_per_layer):
        null_vals = np.array(null_per_layer[l])
        null_mean, null_sd = null_vals.mean(), null_vals.std()
        observed = observed_per_layer[l]
        z = (observed - null_mean) / null_sd if null_sd > 0 else float("nan")
        below_min = observed < null_vals.min()

        print(f"{l:>6} | {null_mean:>10.4f} | {null_sd:>8.4f} | "
              f"{observed:>9.4f} | {z:>8.2f} | {str(below_min):>16}")