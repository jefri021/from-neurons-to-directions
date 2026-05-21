"""
refusal_direction.py
--------------------
Implements the core methods from Paper 1:
  "Refusal in Language Models Is Mediated by a Single Direction"

What lives here:
  1. compute_refusal_direction()   — difference-in-means across layers
  2. select_best_layer()           — pick the layer where r is most causal
  3. ablate_direction()            — remove r from a residual stream vector
  4. add_direction()               — add r to a residual stream vector
  5. Hook-based runtime intervention — patch activations during generation
  6. weight_orthogonalize()        — permanent weight-space ablation

Typical usage flow (matches notebook 01 + experiments 2 & 3):

  # Step 1: compute r at every layer
  directions = compute_refusal_direction(harmful_acts, harmless_acts)
  # directions["layer_15"] → unit vector in R^hidden_size

  # Step 2: pick best layer
  best_layer, best_r = select_best_layer(directions, model, tokenizer, harmful_prompts)

  # Step 3: ablate at runtime
  responses = generate_with_ablation(model, tokenizer, prompts, best_r, best_layer)

  # Step 4: add at runtime
  responses = generate_with_addition(model, tokenizer, prompts, best_r, best_layer)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional

from model_utils import HookManager, tokenize, generate, get_num_layers


# ── 1. Compute refusal direction ──────────────────────────────────────────────

def compute_refusal_direction(
    harmful_acts:  dict[str, torch.Tensor],
    harmless_acts: dict[str, torch.Tensor],
    normalize: bool = True,
) -> dict[int, torch.Tensor]:
    """
    Compute the refusal direction r at each layer using difference-in-means.

    r_l = mean(harmful_acts_l) - mean(harmless_acts_l)

    This is the core of Paper 1 Section 3: the direction that separates
    the internal representation of harmful vs harmless inputs.

    Args:
        harmful_acts:  dict from ActivationStore, key "residual_{l}"
                       each tensor shape [n_prompts, hidden_size]
        harmless_acts: same format, harmless prompts
        normalize:     if True, return unit vectors (recommended —
                       makes cosine similarity comparisons meaningful)

    Returns:
        dict mapping layer_idx (int) → direction vector [hidden_size]

    Example:
        harmful  = ActivationStore.load("results/activations/instruct/harmful_residual.pt")
        harmless = ActivationStore.load("results/activations/instruct/harmless_residual.pt")
        directions = compute_refusal_direction(harmful, harmless)
        r_15 = directions[15]   # unit vector, shape [hidden_size]
    """
    directions = {}

    # Keys are like "residual_0", "residual_1", ..., "residual_31"
    layer_keys = [k for k in harmful_acts if k.startswith("residual_")]

    for key in layer_keys:
        layer_idx = int(key.split("_")[1])

        mu_harmful  = harmful_acts[key].float().mean(dim=0)   # [hidden_size]
        mu_harmless = harmless_acts[key].float().mean(dim=0)  # [hidden_size]

        r = mu_harmful - mu_harmless                          # [hidden_size]

        if normalize:
            r = F.normalize(r, dim=0)

        directions[layer_idx] = r

    print(f"Computed refusal directions for {len(directions)} layers.")
    return directions


# ── 2. Select best layer ───────────────────────────────────────────────────────

def score_direction_at_layer(
    model,
    tokenizer,
    direction: torch.Tensor,
    layer_idx: int,
    harmful_prompts: list[str],
    n_eval: int = 20,
) -> float:
    """
    Score a candidate direction by measuring how much ablating it
    reduces refusal on harmful prompts.

    Score = fraction of harmful prompts where the model no longer refuses
            after ablating the direction at this layer.

    Higher score → direction is more causally responsible for refusal.

    Args:
        direction:       candidate unit vector [hidden_size]
        layer_idx:       layer to apply ablation at
        harmful_prompts: prompts that normally trigger refusal
        n_eval:          number of prompts to evaluate (keep small, it's slow)

    Returns:
        float in [0, 1]
    """
    prompts = harmful_prompts[:n_eval]
    responses = generate_with_ablation(
        model, tokenizer, prompts, direction, layer_idx
    )
    # Rough refusal check: count responses that do NOT contain refusal phrases
    refusal_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i won't", "i will not", "as an ai", "i'm not able",
    ]
    non_refusals = sum(
        1 for r in responses
        if not any(phrase in r.lower() for phrase in refusal_phrases)
    )
    return non_refusals / len(responses)


def select_best_layer(
    directions: dict[int, torch.Tensor],
    model,
    tokenizer,
    harmful_prompts: list[str],
    candidate_layers: Optional[list[int]] = None,
    n_eval: int = 20,
) -> tuple[int, torch.Tensor]:
    """
    Evaluate candidate layers and return the one where ablating r
    most reduces refusal (i.e. the causally strongest layer).

    This is Paper 1 Section 3's layer-selection procedure.

    Args:
        directions:       output of compute_refusal_direction()
        candidate_layers: subset of layers to evaluate.
                          Default: middle 50% of layers (where refusal
                          representations tend to form in 8B models).
        n_eval:           prompts to use for scoring (20 is usually enough)

    Returns:
        (best_layer_idx, best_direction_vector)

    Example:
        best_layer, best_r = select_best_layer(directions, model, tokenizer, harmful_prompts)
        print(f"Best layer: {best_layer}")
    """
    n_layers = get_num_layers(model)

    if candidate_layers is None:
        # Middle 50% heuristic — refusal tends to form in mid-to-late layers
        lo = n_layers // 4
        hi = 3 * n_layers // 4
        candidate_layers = list(range(lo, hi))

    print(f"Evaluating {len(candidate_layers)} candidate layers with {n_eval} prompts each ...")
    scores = {}
    for layer_idx in tqdm(candidate_layers):
        if layer_idx not in directions:
            continue
        score = score_direction_at_layer(
            model, tokenizer, directions[layer_idx], layer_idx,
            harmful_prompts, n_eval=n_eval
        )
        scores[layer_idx] = score

    best_layer = max(scores, key=scores.get)
    print(f"\nLayer scores: { {l: f'{s:.2f}' for l, s in sorted(scores.items())} }")
    print(f"Best layer: {best_layer} (score={scores[best_layer]:.2f})")

    return best_layer, directions[best_layer]


# ── 3. Direction arithmetic ───────────────────────────────────────────────────

def ablate_direction(
    x: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Remove the component of x along the refusal direction.

    x' = x - (x · r̂) r̂

    This is Paper 1's directional ablation. Applied to the residual stream,
    it prevents the model from "writing" refusal into that dimension.

    Args:
        x:         residual stream tensor [..., hidden_size]
        direction: unit vector [hidden_size] (will be normalized if not already)

    Returns:
        tensor of same shape as x
    """
    r = F.normalize(direction.to(x.device, dtype=x.dtype), dim=0)  # ← add dtype=x.dtype
    return x - (x @ r).unsqueeze(-1) * r


def add_direction(
    x: torch.Tensor,
    direction: torch.Tensor,
    alpha: float = 20.0,
) -> torch.Tensor:
    """
    Add a scaled refusal direction to x.

    x' = x + alpha * r̂

    This is Paper 1's activation addition. Applied to the residual stream,
    it pushes the model toward refusal even on harmless prompts.

    Args:
        x:         residual stream tensor [..., hidden_size]
        direction: unit vector [hidden_size]
        alpha:     scale factor. Paper 1 uses values around 15–30.
                   Too large → incoherent output. Start with 20 and tune.

    Returns:
        tensor of same shape as x
    """
    r = F.normalize(direction.to(x.device, dtype=x.dtype), dim=0)  # ← add dtype=x.dtype
    return x + alpha * r


# ── 4. Runtime interventions (hook-based) ─────────────────────────────────────

def generate_with_ablation(
    model,
    tokenizer,
    prompts: list[str],
    direction: torch.Tensor,
    layer_idx: int,
    max_new_tokens: int = 200,
) -> list[str]:
    """
    Generate responses with the refusal direction ablated at runtime.

    Registers a forward hook that removes the component along `direction`
    from the residual stream at `layer_idx` on every forward pass.

    Expected result: model stops refusing harmful prompts.
    This is Paper 1's core causal intervention.

    Args:
        prompts:    list of input prompts (use harmful prompts to test)
        direction:  refusal direction unit vector [hidden_size]
        layer_idx:  layer at which to apply the ablation

    Returns:
        list of generated response strings
    """
    hook_handles = []

    def ablation_hook(module, input, output):
        # In newer transformers, layer output may be a plain tensor or a tuple
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = ablate_direction(hidden, direction)
            return (hidden,) + output[1:]
        else:
            # output is a plain tensor
            return ablate_direction(output, direction)

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(ablation_hook)
    hook_handles.append(handle)

    try:
        responses = generate(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
    finally:
        for h in hook_handles:
            h.remove()

    return responses


def generate_with_addition(
    model,
    tokenizer,
    prompts: list[str],
    direction: torch.Tensor,
    layer_idx: int,
    alpha: float = 20.0,
    max_new_tokens: int = 200,
) -> list[str]:
    """
    Generate responses with the refusal direction added at runtime.

    Expected result: model refuses even harmless prompts.
    Use harmless prompts as input to test direction sufficiency.

    Args:
        prompts:  list of input prompts (use harmless prompts to test)
        alpha:    injection strength (default 20.0, tune if output is incoherent)
    """
    hook_handles = []

    def addition_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = add_direction(hidden, direction, alpha=alpha)
            return (hidden,) + output[1:]
        else:
            return add_direction(output, direction, alpha=alpha)

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(addition_hook)
    hook_handles.append(handle)

    try:
        responses = generate(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
    finally:
        for h in hook_handles:
            h.remove()

    return responses


# ── 5. Weight-space ablation (permanent jailbreak) ────────────────────────────

def weight_orthogonalize(
    model,
    direction: torch.Tensor,
    layers: Optional[list[int]] = None,
    inplace: bool = False,
):
    """
    Permanently remove the refusal direction from model weights.

    For each weight matrix W in the specified layers:
        W' = W - r̂ r̂ᵀ W

    This removes the model's ability to write into direction r at all,
    across all inputs — no hooks needed at inference time.

    This is Paper 1's weight orthogonalization (Section 4).
    It is mathematically equivalent to always running ablate_direction()
    on every forward pass, but baked into the weights permanently.

    Args:
        model:     the model to modify
        direction: refusal direction unit vector [hidden_size]
        layers:    which layers to modify. Default: all layers.
                   Modifying all layers is strongest; you can ablate
                   specific layers to study where the effect comes from.
        inplace:   if False (default), modifies a deepcopy of the model
                   and returns it, leaving the original untouched.
                   if True, modifies the model in place (saves memory,
                   but you lose the original — be careful).

    Returns:
        Modified model (new object if inplace=False, same if inplace=True)

    WARNING:
        This is irreversible on the returned model.
        Always keep your original model loaded separately if you need it.

    Example:
        jailbroken_model = weight_orthogonalize(model, best_r, layers=range(32))
        responses = generate(jailbroken_model, tokenizer, harmful_prompts)
    """
    import copy

    if not inplace:
        model = copy.deepcopy(model)

    if layers is None:
        layers = range(get_num_layers(model))

    r = F.normalize(direction.float(), dim=0)   # [hidden_size]
    # Outer product: r̂ r̂ᵀ — shape [hidden_size, hidden_size]
    rrt = torch.outer(r, r)

    n_modified = 0
    with torch.no_grad():
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]

            # Modify all linear weight matrices in this layer
            # (attention projections + MLP projections)
            for name, module in layer.named_modules():
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    W = module.weight.float()           # [out, in] or [in, out]
                    # Project out r̂ from the input dimension of W
                    # W' = W - r̂ r̂ᵀ W  (removes ability to read r from input)
                    module.weight.data = (W - rrt.to(W.device) @ W).to(module.weight.dtype)
                    n_modified += 1

    print(f"Weight orthogonalization complete. Modified {n_modified} weight matrices "
          f"across {len(list(layers))} layers.")
    return model
