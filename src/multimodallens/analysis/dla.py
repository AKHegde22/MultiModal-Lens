"""Multimodal Direct Logit Attribution (DLA)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
import torch
from PIL import Image

from multimodallens.adapters.base import ModelAdapter
from multimodallens.analysis.hooking import list_hookable_layers
from multimodallens.core.hooks import ForwardHookCache, _extract_first_tensor
from multimodallens.analysis.logit_lens import _get_unembedding, _get_embedding_table, _forward_adapter


@dataclass
class HeadContribution:
    layer: int
    head: int  
    layer_name: str
    contribution_score: float
    norm: float


@dataclass
class MLPContribution:
    layer: int
    layer_name: str
    contribution_score: float
    norm: float


@dataclass
class DLAContribution:
    layer_name: str
    target_token: str
    contribution_score: float
    norm: float


@dataclass
class DLAResult:
    model_family: str
    model_name: str
    prompt: str
    target_token: str
    head_contributions: list[HeadContribution]
    mlp_contributions: list[MLPContribution]
    embedding_contribution: float
    total_logit: float
    residual_error: float

    @property
    def contributions(self) -> list[DLAContribution]:
        """Backwards-compatible: return all contributions as flat list."""
        out = []
        for hc in self.head_contributions:
            out.append(
                DLAContribution(
                    layer_name=f"{hc.layer_name}.head_{hc.head}",
                    target_token=self.target_token,
                    contribution_score=hc.contribution_score,
                    norm=hc.norm,
                )
            )
        for mc in self.mlp_contributions:
            out.append(
                DLAContribution(
                    layer_name=mc.layer_name,
                    target_token=self.target_token,
                    contribution_score=mc.contribution_score,
                    norm=mc.norm,
                )
            )
        out.append(
            DLAContribution(
                layer_name="embedding",
                target_token=self.target_token,
                contribution_score=self.embedding_contribution,
                norm=0.0,
            )
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "target_token": self.target_token,
            "head_contributions": [
                {
                    "layer": c.layer,
                    "head": c.head,
                    "layer_name": c.layer_name,
                    "contribution_score": c.contribution_score,
                    "norm": c.norm,
                }
                for c in self.head_contributions
            ],
            "mlp_contributions": [
                {
                    "layer": c.layer,
                    "layer_name": c.layer_name,
                    "contribution_score": c.contribution_score,
                    "norm": c.norm,
                }
                for c in self.mlp_contributions
            ],
            "embedding_contribution": self.embedding_contribution,
            "total_logit": self.total_logit,
            "residual_error": self.residual_error,
            "contributions": [
                {
                    "layer_name": c.layer_name,
                    "target_token": c.target_token,
                    "contribution_score": c.contribution_score,
                    "norm": c.norm,
                }
                for c in self.contributions
            ],
        }

    def to_html(self) -> str:
        """Render interactive HTML dashboard string."""
        from multimodallens.viz.interactive import create_dla_waterfall_html

        return create_dla_waterfall_html(self)

    def to_plotly(self) -> Any:
        """Return Plotly Figure object."""
        from multimodallens.viz.interactive import create_dla_plotly_figure

        return create_dla_plotly_figure(self)

    def save_html(self, filepath: str) -> None:
        """Save interactive HTML dashboard to disk."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())


class ForwardPreHookCache:
    """Capture layer inputs using register_forward_pre_hook."""

    def __init__(self, model: torch.nn.Module, layer_names: list[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self) -> None:
        modules = dict(self.model.named_modules())
        for name in self.layer_names:
            module = modules.get(name)
            if module is None:
                continue

            def _pre_hook(_mod: torch.nn.Module, inputs: tuple[Any, ...], n: str = name) -> None:
                tensor = _extract_first_tensor(inputs)
                if tensor is not None:
                    self.activations[n] = tensor.detach().cpu().clone()

            self._handles.append(module.register_forward_pre_hook(_pre_hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ForwardPreHookCache":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _get_layer_idx(name: str) -> int:
    match = re.search(r"\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    match = re.search(r"\.(\d+)$", name)
    if match:
        return int(match.group(1))
    return -1


def _get_n_heads_head_dim(model_config: Any, in_features: int) -> tuple[int, int]:
    if model_config is None:
        return 1, in_features

    n_heads = getattr(
        model_config,
        "num_attention_heads",
        getattr(model_config, "num_heads", getattr(model_config, "n_head", None)),
    )
    if n_heads is None:
        return 1, in_features

    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = in_features // n_heads

    if in_features != n_heads * head_dim:
        return 1, in_features

    return n_heads, head_dim


def run_multimodal_dla(
    adapter: ModelAdapter,
    image: Image.Image,
    prompt: str,
    target_token: str | int,
    layer_names: list[str] | None = None,
    include_patterns: list[str] | None = None,
) -> DLAResult:
    """Compute Direct Logit Attribution (DLA) of intermediate layer outputs and attention heads to target token logit."""
    adapter.ensure_loaded()
    assert adapter.model is not None

    unembed = _get_unembedding(adapter)
    unembed_weight = getattr(unembed, "weight", None) if unembed is not None else None

    if unembed_weight is None:
        unembed_weight = _get_embedding_table(adapter)

    if unembed_weight is None or not torch.is_tensor(unembed_weight):
        raise RuntimeError("Model does not expose output or input embeddings required for Direct Logit Attribution.")

    batch = adapter.prepare(image, prompt)
    selected = layer_names or list_hookable_layers(adapter, include_patterns=include_patterns)
    if not selected:
        raise RuntimeError("No hookable layers were found for DLA analysis.")

    # Target token id resolution
    target_id: int
    target_str: str
    if isinstance(target_token, int):
        target_id = target_token
        target_str = str(target_token)
    else:
        target_str = target_token
        tokenizer = getattr(adapter, "tokenizer", None)
        if tokenizer is None and hasattr(adapter, "processor"):
            tokenizer = getattr(adapter.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
            target_id = int(tokenizer.convert_tokens_to_ids(target_token))
        else:
            target_id = 0



    o_projs = [l for l in selected if l.endswith("o_proj") or l.endswith("c_proj") or (l.endswith("dense") and "attn" in l)]
    mlps = [l for l in selected if l.endswith("mlp") or l.endswith("ffn")]

    embed_layer = None
    for name, module in adapter.model.named_modules():
        if isinstance(module, torch.nn.Embedding) and "position" not in name.lower():
            embed_layer = name
            break

    out_layers = mlps.copy()
    if embed_layer:
        out_layers.append(embed_layer)

    with torch.no_grad():
        with ForwardHookCache(adapter.model, out_layers, max_tokens=None) as out_cache:
            with ForwardPreHookCache(adapter.model, o_projs) as in_cache:
                outputs = _forward_adapter(adapter, batch.model_inputs, requires_grad=False)

    logits = getattr(outputs, "logits", outputs)
    total_logit = 0.0
    if torch.is_tensor(logits):
        if logits.ndim == 3:
            total_logit = float(logits[0, -1, target_id].item())
        elif logits.ndim == 2:
            total_logit = float(logits[-1, target_id].item())

    d = unembed_weight[target_id].to(device=adapter.device, dtype=torch.float32)

    head_contributions: list[HeadContribution] = []
    mlp_contributions: list[MLPContribution] = []

    modules = dict(adapter.model.named_modules())
    config = getattr(adapter.model, "config", None)

    total_attn_sum = 0.0

    for o_proj_name in o_projs:
        layer_idx = _get_layer_idx(o_proj_name)
        inp = in_cache.activations.get(o_proj_name)
        if inp is None:
            continue

        if inp.ndim == 3:
            h_inp = inp[0, -1].to(device=adapter.device, dtype=torch.float32)
        elif inp.ndim == 2:
            h_inp = inp[-1].to(device=adapter.device, dtype=torch.float32)
        else:
            continue

        mod = modules[o_proj_name]
        W_O = getattr(mod, "weight", None)
        if W_O is None:
            continue

        if isinstance(mod, torch.nn.Linear):
            W_O = W_O.to(device=adapter.device, dtype=torch.float32)
            out_features, in_features = W_O.shape
        elif W_O.ndim == 2:
            W_O = W_O.t().to(device=adapter.device, dtype=torch.float32)
            out_features, in_features = W_O.shape
        else:
            continue

        n_heads, head_dim = _get_n_heads_head_dim(config, in_features)

        h_inp_heads = h_inp.view(n_heads, head_dim)
        W_O_reshaped = W_O.view(out_features, n_heads, head_dim)

        head_out = torch.einsum("ohd,hd->ho", W_O_reshaped, h_inp_heads)
        if head_out.shape[-1] != d.shape[-1]:
            vis_proj = getattr(adapter.model, "visual_projection", None)
            if vis_proj is not None and callable(vis_proj):
                try:
                    head_out = vis_proj(head_out)
                except Exception:
                    continue
            else:
                continue

        scores = torch.mv(head_out, d)

        for head_idx in range(n_heads):
            score = float(scores[head_idx].item())
            norm_val = float(torch.norm(head_out[head_idx]).item())
            head_contributions.append(
                HeadContribution(
                    layer=layer_idx,
                    head=head_idx,
                    layer_name=o_proj_name,
                    contribution_score=score,
                    norm=norm_val,
                )
            )
            total_attn_sum += score

    total_mlp_sum = 0.0
    for mlp_name in mlps:
        layer_idx = _get_layer_idx(mlp_name)
        out = out_cache.activations.get(mlp_name)
        if out is None:
            continue

        if out.ndim == 3:
            h_out = out[0, -1].to(device=adapter.device, dtype=torch.float32)
        elif out.ndim == 2:
            h_out = out[-1].to(device=adapter.device, dtype=torch.float32)
        else:
            continue

        if h_out.shape[-1] != d.shape[-1]:
            vis_proj = getattr(adapter.model, "visual_projection", None)
            if vis_proj is not None and callable(vis_proj):
                try:
                    h_out = vis_proj(h_out)
                except Exception:
                    continue
            else:
                continue

        score = float(torch.dot(h_out, d).item())
        norm_val = float(torch.norm(h_out).item())
        mlp_contributions.append(
            MLPContribution(
                layer=layer_idx,
                layer_name=mlp_name,
                contribution_score=score,
                norm=norm_val,
            )
        )
        total_mlp_sum += score

    embed_contrib = 0.0
    if embed_layer:
        emb_out = out_cache.activations.get(embed_layer)
        if emb_out is not None:
            if emb_out.ndim == 3:
                h_emb = emb_out[0, -1].to(device=adapter.device, dtype=torch.float32)
            elif emb_out.ndim == 2:
                h_emb = emb_out[-1].to(device=adapter.device, dtype=torch.float32)
            else:
                h_emb = None
            if h_emb is not None:
                if h_emb.shape[-1] != d.shape[-1]:
                    vis_proj = getattr(adapter.model, "visual_projection", None)
                    if vis_proj is not None and callable(vis_proj):
                        try:
                            h_emb = vis_proj(h_emb)
                        except Exception:
                            h_emb = None
                    else:
                        h_emb = None
                if h_emb is not None:
                    embed_contrib = float(torch.dot(h_emb, d).item())

    residual_error = total_logit - (total_attn_sum + total_mlp_sum + embed_contrib)

    return DLAResult(
        model_family=adapter.family,
        model_name=adapter.model_name,
        prompt=prompt,
        target_token=target_str,
        head_contributions=head_contributions,
        mlp_contributions=mlp_contributions,
        embedding_contribution=embed_contrib,
        total_logit=total_logit,
        residual_error=residual_error,
    )
