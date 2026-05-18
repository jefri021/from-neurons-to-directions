"""
model_utils.py
--------------
Handles model loading and forward hook registration.

Kaggle input paths (add these as inputs to your notebook once via GUI):
  Base model:     /kaggle/input/llama-3.1-8b/transformers/default/1
  Instruct model: /kaggle/input/llama-3.1-8b-instruct/transformers/default/1

All other src/ modules import from here.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager


# ── Kaggle input paths ────────────────────────────────────────────────────────

PATHS = {
    "base":     "/kaggle/input/models/metaresearch/llama-3.2/transformers/3b/1",
    "instruct": "/kaggle/input/models/metaresearch/llama-3.2/transformers/3b-instruct/1",
}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_key: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load a model and tokenizer from Kaggle input space.

    Uses device_map="auto" so HuggingFace distributes layers across
    whatever GPUs (or CPU) are available — no manual device management needed.

    Args:
        model_key:  "base" or "instruct"
        dtype:      bfloat16 recommended for 8B on a single GPU

    Returns:
        (model, tokenizer)

    Example:
        model, tokenizer = load_model_and_tokenizer("base")
    """
    if model_key not in PATHS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(PATHS)}")

    path = PATHS[model_key]
    print(f"Loading '{model_key}' from {path} ...")

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        device_map="auto",  # handles single/multi-GPU and CPU offload automatically
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    devices   = set(str(p.device) for p in model.parameters())
    print(f"  {n_params:.1f}B parameters | devices: {devices}")
    return model, tokenizer


# ── Device helper ─────────────────────────────────────────────────────────────

def get_model_device(model) -> torch.device:
    """
    Return the primary device of the model.

    With device_map="auto", the embedding layer is always placed on the
    first device, so sending inputs there is always correct.
    next(model.parameters()) points to the embedding weights.
    """
    return next(model.parameters()).device


# ── Tokenization helpers ──────────────────────────────────────────────────────

def tokenize(
    prompts: list[str],
    tokenizer,
    model,
    max_length: int = 256,
) -> dict:
    """
    Tokenize a list of prompts and move tensors to the model's primary device.
    Device is derived from the model — never hardcoded.

    Returns a dict of tensors ready for model input.
    """
    device = get_model_device(model)
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)


def apply_chat_template(prompt: str, tokenizer) -> str:
    """
    Wrap a raw prompt in the instruct chat template.
    Use this when running the instruct model so the format matches training.
    For the base model, pass the raw prompt directly.
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Hook registration ─────────────────────────────────────────────────────────

class HookManager:
    """
    Registers and removes forward hooks on transformer layers.

    Hooks capture the residual stream (hidden states) and/or MLP
    intermediate activations at specified layers.

    Usage — explicit:
        hooks = HookManager(model)
        hooks.register_residual_hooks(layers=[10, 15, 20])
        with torch.no_grad():
            model(**inputs)
        activations = hooks.get_activations()
        hooks.remove()

    Usage — context manager (recommended):
        with hooks.record(layers=range(32), mode="neurons"):
            model(**inputs)
        activations = hooks.get_activations()
    """

    def __init__(self, model):
        self.model = model
        self._handles: list = []
        self._storage: dict[str, torch.Tensor] = {}

    def _get_layer(self, layer_idx: int):
        """Return the transformer block at layer_idx (Llama-style)."""
        return self.model.model.layers[layer_idx]

    # ── Hook types ────────────────────────────────────────────────────────────

    def register_residual_hooks(self, layers: list[int]):
        """
        Capture the residual stream output (hidden state) after each layer.
        Stored under key "residual_{layer_idx}".
        Shape: [batch, seq_len, hidden_size]

        This is what Paper 1 operates on.
        """
        for idx in layers:
            layer = self._get_layer(idx)

            def make_hook(i):
                # def hook(module, input, output):
                #     # Llama layer output is a tuple; [0] is the hidden state
                #     self._storage[f"residual_{i}"] = output[0].detach().cpu()
                def hook(module, input, output):
                    # print(f"output type: {type(output)}")
                    # print(f"len(output): {len(output)}")
                    # for j, o in enumerate(output):
                    #     if hasattr(o, 'shape'):
                    #         print(f"  output[{j}] shape: {o.shape}")
                    #     else:
                    #         print(f"  output[{j}]: {o}")
                    # output IS the hidden
                    self._storage[f"residual_{i}"] = output.detach().cpu()
                return hook

            self._handles.append(layer.register_forward_hook(make_hook(idx)))

    def register_mlp_neuron_hooks(self, layers: list[int]):
        """
        Capture neuron activations BEFORE the down-projection.
        Stored under key "neurons_{layer_idx}".
        Shape: [batch, seq_len, intermediate_size]

        For Llama's SwiGLU MLP:
            neurons = act_fn(gate_proj(x)) * up_proj(x)
        By hooking down_proj's input we get exactly this gated intermediate,
        which is what Paper 2 calls "neuron activations."
        """
        for idx in layers:
            mlp = self._get_layer(idx).mlp

            def make_hook(i):
                def hook(module, input, output):
                    # input[0] is the tensor flowing into down_proj
                    self._storage[f"neurons_{i}"] = input[0].detach().cpu()
                return hook

            self._handles.append(mlp.down_proj.register_forward_hook(make_hook(idx)))

    # ── Storage access ────────────────────────────────────────────────────────

    def get_activations(self) -> dict[str, torch.Tensor]:
        """Return all captured activations (copy, not reference)."""
        return dict(self._storage)

    def clear(self):
        """Clear stored activations without removing hooks."""
        self._storage.clear()

    def remove(self):
        """Remove all registered hooks and clear storage."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._storage.clear()

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def record(self, layers: list[int], mode: str = "residual"):
        """
        Context manager for safe, automatic hook cleanup.

        Args:
            layers: layer indices to hook
            mode:
                "residual" → residual stream output (Paper 1)
                "neurons"  → pre-down-proj MLP activations (Paper 2)
                "all"      → both of the above

        Example:
            hook_mgr = HookManager(model)
            with hook_mgr.record(layers=range(32), mode="neurons"):
                with torch.no_grad():
                    model(**inputs)
            acts = hook_mgr.get_activations()
            # acts["neurons_15"] → tensor of shape [batch, seq, intermediate_size]
        """
        self.clear()
        if mode in ("residual", "all"):
            self.register_residual_hooks(layers)
        if mode in ("neurons", "all"):
            self.register_mlp_neuron_hooks(layers)
        try:
            yield self
        finally:
            self.remove()


# ── Generation helper ─────────────────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 200,
) -> list[str]:
    """
    Generate responses for a list of prompts.
    Device is inferred from the model — never hardcoded.

    Returns decoded response strings with the prompt stripped.
    """
    inputs    = tokenize(prompts, tokenizer, model)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,               # greedy — deterministic, good for evals
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only newly generated tokens (skip the prompt)
    return tokenizer.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )


# ── Model introspection ───────────────────────────────────────────────────────

def get_num_layers(model) -> int:
    """Number of transformer layers."""
    return len(model.model.layers)

def get_hidden_size(model) -> int:
    """Residual stream dimension (d_model)."""
    return model.config.hidden_size

def get_intermediate_size(model) -> int:
    """MLP intermediate dimension = number of neurons per layer."""
    return model.config.intermediate_size