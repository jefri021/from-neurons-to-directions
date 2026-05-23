"""
activation_store.py
-------------------
Runs batched forward passes over a prompt dataset, captures activations
via HookManager, and saves/loads them to disk.

Why this exists:
  Extracting activations from 32 layers across 200 prompts takes ~minutes.
  Without caching, you'd re-run that every time you tweak a metric or plot.
  This module runs it once and saves results as .pt files under results/activations/.

Typical usage:
  store = ActivationStore(model, tokenizer, save_dir="results/activations/base")
  store.collect(prompts, mode="neurons", layers=range(32), tag="harmful")
  store.save()

  # Later, in a different notebook cell / session:
  data = ActivationStore.load("results/activations/base/harmful_neurons.pt")
  # data["neurons_15"] → tensor [n_prompts, seq_len, intermediate_size]
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm

from model_utils import HookManager, tokenize


# ── ActivationStore ───────────────────────────────────────────────────────────

class ActivationStore:
    """
    Collects and caches activations for a set of prompts.

    Each call to .collect() runs a forward pass over all prompts in
    mini-batches, accumulates activations per layer, and stores them
    internally. Call .save() to write to disk, or .load() to restore.

    Storage layout on disk:
        {save_dir}/{tag}_{mode}.pt   → dict[str, Tensor]
        {save_dir}/{tag}_{mode}.json → metadata (model_key, layers, prompt count)

    Key naming in the dict matches HookManager:
        "residual_{layer}" → [n_prompts, seq_len, hidden_size]
        "neurons_{layer}"  → [n_prompts, seq_len, intermediate_size]
    """

    def __init__(self, model, tokenizer, save_dir: str = "results/activations"):
        self.model     = model
        self.tokenizer = tokenizer
        self.save_dir  = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._data: dict[str, torch.Tensor] = {}   # accumulated activations
        self._meta: dict = {}                       # metadata for the last collect()


    @staticmethod
    def _pad_to_same_seq_len(chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Pad all tensors in chunks to the same seq_len (dim 1).
        Pads on the LEFT with zeros, consistent with left-padding tokenization.
        Shape: [batch, seq_len, dim] → all padded to max seq_len in the list.
        """
        max_seq = max(t.shape[1] for t in chunks)
        padded = []
        for t in chunks:
            pad_len = max_seq - t.shape[1]
            if pad_len > 0:
                # F.pad format: (last_dim_right, last_dim_left, ..., first_dim_right, first_dim_left)
                # We want to pad dim 1 on the left → (0, 0, pad_len, 0)
                t = torch.nn.functional.pad(t, (0, 0, pad_len, 0))
            padded.append(t)
        return padded

    # ── Collection ────────────────────────────────────────────────────────────

    def collect(
        self,
        prompts: list[str],
        mode: str,
        layers: list[int],
        tag: str,
        batch_size: int = 8,
        max_length: int = 256,
        token_position: int = -1,
    ) -> dict[str, torch.Tensor]:
        """
        Run forward passes over all prompts and collect activations.

        Args:
            prompts:        List of raw text prompts.
            mode:           "residual", "neurons", or "all" (see HookManager).
            layers:         Which layer indices to hook.
            tag:            Label for this dataset (e.g. "harmful", "harmless").
                            Used as filename prefix when saving.
            batch_size:     Prompts per forward pass. Lower if you hit OOM.
            max_length:     Tokenizer truncation length.
            token_position: Which token position to extract per prompt.
                            -1 = last token (default, standard for decoder-only).
                            Use -1 unless you have a specific reason to change it.

        Returns:
            dict mapping key → Tensor of shape [n_prompts, dim]
            e.g. {"neurons_15": Tensor[200, 14336], "neurons_20": Tensor[200, 14336]}

        Note on token_position:
            We extract a single position per prompt (default: last token).
            This collapses seq_len → 1 for downstream analysis, which is
            standard in both papers. The full [batch, seq, dim] tensors are
            available in HookManager if you need them.
        """
        hook_mgr   = HookManager(self.model)
        layers     = list(layers)
        accumulated: dict[str, list[torch.Tensor]] = {
            f"{mode_prefix}_{l}": []
            for l in layers
            for mode_prefix in self._mode_prefixes(mode)
        }

        print(f"Collecting '{tag}' activations | mode={mode} | {len(prompts)} prompts | "
              f"{len(layers)} layers | batch_size={batch_size}")

        for batch_start in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[batch_start : batch_start + batch_size]
            inputs = tokenize(batch, self.tokenizer, self.model, max_length=max_length)

            with hook_mgr.record(layers=layers, mode=mode):
                with torch.no_grad():
                    self.model(**inputs)

                acts = hook_mgr.get_activations()

            # Extract the target token position from each activation
            for key, tensor in acts.items():
                # tensor shape: [batch, seq_len, dim]
                accumulated[key].append(tensor.cpu())  # [batch, seq_len, dim]

        # Concatenate across batches → [n_prompts, dim]
        self._data = {
            key: torch.cat(ActivationStore._pad_to_same_seq_len(chunks), dim=0)
            for key, chunks in accumulated.items()
        }
        self._meta = {
            "tag":            tag,
            "mode":           mode,
            "layers":         layers,
            "n_prompts":      len(prompts),
            "token_position": token_position,
            "max_length":     max_length,
        }

        print(f"  Done. Tensor shapes: { {k: list(v.shape) for k, v in self._data.items()} }")
        return self._data

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, tag: str = None, mode: str = None):
        """
        Save collected activations and metadata to disk.

        Filenames are derived from tag and mode (from the last collect() call
        unless overridden here).

        Files written:
            {save_dir}/{tag}_{mode}.pt    → torch tensors
            {save_dir}/{tag}_{mode}.json  → metadata
        """
        tag  = tag  or self._meta.get("tag",  "unknown")
        mode = mode or self._meta.get("mode", "unknown")
        stem = f"{tag}_{mode}"

        pt_path   = self.save_dir / f"{stem}.pt"
        json_path = self.save_dir / f"{stem}.json"

        torch.save(self._data, pt_path)
        with open(json_path, "w") as f:
            json.dump(self._meta, f, indent=2)

        print(f"Saved → {pt_path}  ({pt_path.stat().st_size / 1e6:.1f} MB)")

    @staticmethod
    def load(path: str) -> dict[str, torch.Tensor]:
        """
        Load saved activations from disk.

        Args:
            path: path to the .pt file (the .json metadata is loaded automatically
                  if it exists alongside it).

        Returns:
            dict[str, Tensor] — same format as collect() output.

        Example:
            data = ActivationStore.load("results/activations/base/harmful_neurons.pt")
            harmful_layer15 = data["neurons_15"]  # [n_prompts, intermediate_size]
        """
        pt_path   = Path(path)
        json_path = pt_path.with_suffix(".json")

        data = torch.load(pt_path, map_location="cpu")

        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
            print(f"Loaded {pt_path.name} | {meta}")
        else:
            print(f"Loaded {pt_path.name} (no metadata file found)")

        return data

    # ── Convenience: load a pair ──────────────────────────────────────────────

    @staticmethod
    def load_pair(
        dir_harmful: str,
        dir_harmless: str,
        mode: str,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load activations for a single layer from two saved files.

        Returns:
            (harmful_acts, harmless_acts) each of shape [n_prompts, dim]

        Example:
            harmful, harmless = ActivationStore.load_pair(
                dir_harmful  = "results/activations/instruct/harmful_neurons.pt",
                dir_harmless = "results/activations/instruct/harmless_neurons.pt",
                mode  = "neurons",
                layer = 15,
            )
        """
        harmful_data  = ActivationStore.load(dir_harmful)
        harmless_data = ActivationStore.load(dir_harmless)
        key = f"{mode}_{layer}"
        return harmful_data[key], harmless_data[key]

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _mode_prefixes(mode: str) -> list[str]:
        """Map mode string to the key prefixes HookManager uses."""
        if mode == "residual":
            return ["residual"]
        if mode == "neurons":
            return ["neurons"]
        if mode == "all":
            return ["residual", "neurons"]
        raise ValueError(f"Unknown mode '{mode}'. Choose: residual, neurons, all")