"""Gradio user interface for MultimodalLens."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import zipfile

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from multimodallens.config import DEFAULT_MODELS
from multimodallens.core.pipeline import LensPipeline
from multimodallens.core.registry import SUPPORTED_FAMILIES
from multimodallens.eval.runner import EvaluationRunner
from multimodallens.viz.plots import plot_alignment, plot_faithfulness_curves, plot_token_scores
from multimodallens.utils.image_ops import mask_top_patches, overlay_heatmap


PIPELINE = LensPipeline()
EVAL_RUNNER = EvaluationRunner(PIPELINE)
ROOT_DIR = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def _make_artifact_run_dir(prefix: str) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = ARTIFACTS_DIR / f"{prefix}_{stamp}"
    run_dir = base
    idx = 1
    while run_dir.exists():
        run_dir = ARTIFACTS_DIR / f"{base.name}_{idx}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _zip_run_dir(run_dir: Path) -> str:
    zip_path = run_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in run_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(run_dir))
    return str(zip_path)


def _save_plot(fig: object, path: Path) -> None:
    save_fn = getattr(fig, "savefig", None)
    if callable(save_fn):
        save_fn(path, dpi=180, bbox_inches="tight")


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _to_json_text(payload: dict[str, Any], max_chars: int = 50000) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated in UI, full data available in download bundle>..."


def _parse_list_field(raw: str) -> list[str] | None:
    items = [x.strip() for x in re.split(r"[,\n]", raw) if x.strip()]
    return items or None


def _parse_int_list_field(raw: str) -> list[int] | None:
    items = _parse_list_field(raw)
    if not items:
        return None
    try:
        return [int(x) for x in items]
    except ValueError as exc:
        raise gr.Error("Positions must be a comma-separated list of integers (e.g., -1,0,5).") from exc


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _export_explore_bundle(
    result,
    overlay: np.ndarray,
    align_fig: object,
    token_fig: object,
    token_df: pd.DataFrame,
    diagnostics: str,
    faith_fig: object | None,
    faith_df: pd.DataFrame,
    mechanistic_summary: str | None = None,
    mechanistic_table: pd.DataFrame | None = None,
    mechanistic_payload: dict[str, Any] | None = None,
    mechanistic_layer_arrays: dict[str, np.ndarray] | None = None,
) -> str:
    run_dir = _make_artifact_run_dir("explore")

    Image.fromarray(overlay.astype(np.uint8)).save(run_dir / "attention_overlay.png")
    _save_plot(align_fig, run_dir / "alignment_plot.png")
    _save_plot(token_fig, run_dir / "token_scores_plot.png")
    token_df.to_csv(run_dir / "token_scores.csv", index=False)
    pd.DataFrame(result.alignment_matrix).to_csv(run_dir / "alignment_matrix.csv", index=False)
    (run_dir / "diagnostics.md").write_text(diagnostics, encoding="utf-8")

    maps_dir = run_dir / "attention_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for name, array in result.attention_maps.items():
        np.save(maps_dir / f"{name}.npy", np.asarray(array))

    if faith_fig is not None:
        _save_plot(faith_fig, run_dir / "faithfulness_curves.png")
    if not faith_df.empty:
        faith_df.to_csv(run_dir / "faithfulness_metrics.csv", index=False)

    if mechanistic_payload is not None:
        mech_dir = run_dir / "mechanistic"
        mech_dir.mkdir(parents=True, exist_ok=True)
        _save_json(mech_dir / "result.json", mechanistic_payload)
        if mechanistic_summary:
            (mech_dir / "summary.md").write_text(mechanistic_summary, encoding="utf-8")
        if mechanistic_table is not None and not mechanistic_table.empty:
            mechanistic_table.to_csv(mech_dir / "table.csv", index=False)

        if mechanistic_layer_arrays:
            act_dir = mech_dir / "layer_activations"
            act_dir.mkdir(parents=True, exist_ok=True)
            layer_index: list[dict[str, str]] = []
            for layer_name, values in mechanistic_layer_arrays.items():
                filename = f"{_safe_name(layer_name)}.npy"
                np.save(act_dir / filename, np.asarray(values))
                layer_index.append({"layer_name": layer_name, "file": f"layer_activations/{filename}"})
            _save_json(mech_dir / "layer_index.json", {"layers": layer_index})

    _save_json(
        run_dir / "summary.json",
        {
            "model_family": result.model_family,
            "model_name": result.model_name,
            "prompt": result.prompt,
            "global_score": float(result.global_score),
            "num_tokens": len(result.tokens),
            "patch_grid": list(result.patch_grid),
            "image_size": list(result.image_size),
            "tokens": result.tokens,
        },
    )
    return _zip_run_dir(run_dir)


def _export_compare_bundle(
    family: str,
    model_name: str,
    prompt_a: str,
    prompt_b: str,
    compare_df: pd.DataFrame,
    overlay_a: np.ndarray,
    overlay_b: np.ndarray,
    token_fig_a: object,
    token_fig_b: object,
    summary: str,
) -> str:
    run_dir = _make_artifact_run_dir("compare")

    compare_df.to_csv(run_dir / "comparison_summary.csv", index=False)
    Image.fromarray(overlay_a.astype(np.uint8)).save(run_dir / "overlay_prompt_a.png")
    Image.fromarray(overlay_b.astype(np.uint8)).save(run_dir / "overlay_prompt_b.png")
    _save_plot(token_fig_a, run_dir / "token_similarity_prompt_a.png")
    _save_plot(token_fig_b, run_dir / "token_similarity_prompt_b.png")
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    _save_json(
        run_dir / "metadata.json",
        {
            "model_family": family,
            "model_name": model_name,
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
        },
    )
    return _zip_run_dir(run_dir)


def _export_eval_bundle(df: pd.DataFrame, summary: str, dataset_path: Path, family: str, model_name: str) -> str:
    run_dir = _make_artifact_run_dir("eval")

    df.to_csv(run_dir / "eval_results.csv", index=False)
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    _save_json(
        run_dir / "metadata.json",
        {
            "dataset_path": str(dataset_path),
            "model_family": family,
            "model_name": model_name,
            "rows": int(len(df)),
            "columns": list(df.columns),
        },
    )
    return _zip_run_dir(run_dir)


def _default_model_for_family(family: str) -> str:
    return DEFAULT_MODELS.get(family, DEFAULT_MODELS["auto"])


def _validate_inputs(image: Image.Image | None, prompt: str) -> None:
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt.strip():
        raise gr.Error("Please provide a non-empty prompt.")


def _collect_preflight_lines(
    family: str,
    model_name: str,
    trust_remote_code: bool,
) -> tuple[list[str], list[str]]:
    report = PIPELINE.preflight(
        family=family,
        model_name=model_name,
        trust_remote_code=trust_remote_code,
    )
    lines = [
        f"Requested family: `{report.requested_family}`",
        f"Resolved family: `{report.resolved_family}`",
    ]
    if report.model_type:
        lines.append(f"Model type: `{report.model_type}`")

    notes: list[str] = []
    for warning in report.warnings:
        notes.append(f"Preflight warning: {warning}")
    for error in report.errors:
        notes.append(f"Preflight error: {error}")
    return lines, notes


def _run_mechanistic_suite(
    family: str,
    model_name: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    image: Image.Image,
    prompt: str,
    reference_result,
    include_patterns_raw: str,
    positions_raw: str,
    layer_index: float,
    top_k: float,
    max_tokens: float,
    max_layers: float,
    mask_fraction: float,
    visual_only: bool,
) -> tuple[str, pd.DataFrame, str, dict[str, Any], dict[str, np.ndarray]]:
    include_patterns = _parse_list_field(include_patterns_raw)
    positions = _parse_int_list_field(positions_raw)
    layer_index_i = int(layer_index)
    top_k_i = max(1, int(top_k))
    max_tokens_i = int(max_tokens)
    max_layers_i = int(max_layers)

    payload: dict[str, Any] = {
        "model_family": family,
        "model_name": model_name,
        "prompt": prompt,
        "probes": {},
    }
    rows: list[dict[str, Any]] = []
    summary_lines = [f"Mechanistic suite: `{model_name}` ({family})"]
    layer_arrays: dict[str, np.ndarray] = {}
    rgb = image.convert("RGB")

    def _record_error(probe: str, exc: Exception) -> None:
        message = str(exc)
        payload["probes"][probe] = {"error": message}
        rows.append({"probe": probe, "metric": "error", "value": message})
        summary_lines.append(f"- {probe}: failed ({message})")

    try:
        layers = PIPELINE.list_layers(
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            include_patterns=include_patterns,
        )
        payload["probes"]["layers"] = {
            "num_layers": len(layers),
            "layers": layers,
        }
        rows.append({"probe": "layers", "metric": "num_layers", "value": len(layers)})
        summary_lines.append(f"- layers: `{len(layers)}` discovered")
    except Exception as exc:  # pragma: no cover - runtime model variance
        _record_error("layers", exc)

    try:
        captured = PIPELINE.capture_internals(
            family=family,
            model_name=model_name,
            image=rgb,
            prompt=prompt,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            include_patterns=include_patterns,
            max_tokens=max_tokens_i,
        )
        payload["probes"]["cache"] = {
            "num_layers": len(captured.layers),
            "layers": [
                {
                    "layer_name": x.layer_name,
                    "shape": list(x.shape),
                }
                for x in captured.layers
            ],
        }
        rows.append({"probe": "cache", "metric": "captured_layers", "value": len(captured.layers)})
        summary_lines.append(f"- cache: `{len(captured.layers)}` layers captured")
        for item in captured.layers:
            layer_arrays[f"cache_{item.layer_name}"] = np.asarray(item.values)
    except Exception as exc:  # pragma: no cover - runtime model variance
        _record_error("cache", exc)

    try:
        patch_scores = reference_result.attention_maps["vision_rollout"].reshape(-1)
        target_image = mask_top_patches(
            image=rgb,
            patch_scores=patch_scores,
            patch_grid=reference_result.patch_grid,
            mask_fraction=float(mask_fraction),
        )
        patched = PIPELINE.activation_patch(
            family=family,
            model_name=model_name,
            source_image=rgb,
            target_image=target_image,
            prompt=prompt,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            layer_index=layer_index_i,
            include_patterns=include_patterns,
            visual_only=visual_only,
        )
        payload["probes"]["patch"] = _to_jsonable(patched)
        rows.extend(
            [
                {"probe": "patch", "metric": "baseline_score", "value": patched.baseline_score},
                {"probe": "patch", "metric": "patched_score", "value": patched.patched_score},
                {"probe": "patch", "metric": "delta_score", "value": patched.delta_score},
            ]
        )
        summary_lines.append(f"- patch: delta `{patched.delta_score:.6f}` at `{patched.layer_name}`")
    except Exception as exc:  # pragma: no cover - runtime model variance
        _record_error("patch", exc)

    try:
        lens = PIPELINE.logit_lens(
            family=family,
            model_name=model_name,
            image=rgb,
            prompt=prompt,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            include_patterns=include_patterns,
            positions=positions,
            top_k=top_k_i,
            max_layers=max_layers_i if max_layers_i > 0 else None,
        )
        payload["probes"]["logit_lens"] = _to_jsonable(lens)
        rows.append({"probe": "logit_lens", "metric": "decoded_steps", "value": len(lens.steps)})
        summary_lines.append(f"- logit lens: `{len(lens.steps)}` decoded steps")
    except Exception as exc:  # pragma: no cover - runtime model variance
        _record_error("logit_lens", exc)

    try:
        circuits = PIPELINE.grounding_heads(
            family=family,
            model_name=model_name,
            image=rgb,
            prompt=prompt,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            mask_fraction=float(mask_fraction),
            top_k=top_k_i,
        )
        payload["probes"]["grounding"] = _to_jsonable(circuits)
        rows.append({"probe": "grounding", "metric": "top_heads", "value": len(circuits.heads)})
        rows.append({"probe": "grounding", "metric": "score_drop", "value": circuits.score_drop})
        summary_lines.append(f"- grounding: `{len(circuits.heads)}` heads ranked")
    except Exception as exc:  # pragma: no cover - runtime model variance
        _record_error("grounding", exc)

    summary = "\n".join(summary_lines)
    table_df = pd.DataFrame(rows)
    preview = _to_json_text(payload)
    return summary, table_df, preview, payload, layer_arrays


def run_explore(
    family: str,
    model_name: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    compute_gradients: bool,
    run_faithfulness: bool,
    run_mechanistic_suite: bool,
    mech_include_patterns_raw: str,
    mech_positions_raw: str,
    mech_layer_index: float,
    mech_top_k: float,
    mech_max_tokens: float,
    mech_max_layers: float,
    mech_mask_fraction: float,
    mech_visual_only: bool,
    image: Image.Image | None,
    prompt: str,
):
    try:
        _validate_inputs(image, prompt)
        assert image is not None

        preflight_lines, preflight_notes = _collect_preflight_lines(
            family=family,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )

        result = PIPELINE.analyze(
            family=family,
            model_name=model_name,
            image=image.convert("RGB"),
            prompt=prompt,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            compute_gradients=compute_gradients,
            run_faithfulness=run_faithfulness,
        )

        vision_map = result.attention_maps["vision_rollout"]
        overlay = overlay_heatmap(image, vision_map, alpha=0.42, colormap="inferno")

        align_fig = plot_alignment(result.tokens, result.alignment_matrix)
        token_fig = plot_token_scores(result.tokens, result.token_scores)

        token_df = pd.DataFrame(
            {
                "token": result.tokens,
                "score": result.token_scores,
            }
        ).sort_values("score", ascending=False)

        lines = [
            *preflight_lines,
            f"Model: `{result.model_name}` ({result.model_family})",
            f"Global score: `{result.global_score:.6f}`",
            f"Patch grid: `{result.patch_grid[0]} x {result.patch_grid[1]}`",
            f"Tokens: `{len(result.tokens)}`",
        ]

        if "vision_grad" in result.attention_maps:
            lines.append("Gradient attribution computed: `yes`")

        if preflight_notes:
            lines.extend(preflight_notes)

        faith_fig = None
        faith_df = pd.DataFrame(columns=["metric", "value"])

        if result.faithfulness is not None:
            faith_fig = plot_faithfulness_curves(
                result.faithfulness.deletion_curve,
                result.faithfulness.insertion_curve,
            )
            faith_rows = [
                ("counterfactual_drop", result.faithfulness.counterfactual_drop),
                ("attn_grad_spearman", result.faithfulness.attn_grad_spearman),
                ("deletion_auc", float(sum(result.faithfulness.deletion_curve))),
                ("insertion_auc", float(sum(result.faithfulness.insertion_curve))),
            ]
            faith_df = pd.DataFrame(faith_rows, columns=["metric", "value"])

        mech_summary = "Mechanistic suite not run. Enable `Run mechanistic suite` to compute all 4 probes."
        mech_table = pd.DataFrame(columns=["probe", "metric", "value"])
        mech_json = "{}"
        mech_payload: dict[str, Any] | None = None
        mech_layers: dict[str, np.ndarray] | None = None

        if run_mechanistic_suite:
            (
                mech_summary,
                mech_table,
                mech_json,
                mech_payload,
                mech_layers,
            ) = _run_mechanistic_suite(
                family=family,
                model_name=model_name,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                image=image,
                prompt=prompt,
                reference_result=result,
                include_patterns_raw=mech_include_patterns_raw,
                positions_raw=mech_positions_raw,
                layer_index=mech_layer_index,
                top_k=mech_top_k,
                max_tokens=mech_max_tokens,
                max_layers=mech_max_layers,
                mask_fraction=mech_mask_fraction,
                visual_only=mech_visual_only,
            )

        diagnostics = "\n".join(lines)
        explore_bundle = _export_explore_bundle(
            result=result,
            overlay=overlay,
            align_fig=align_fig,
            token_fig=token_fig,
            token_df=token_df,
            diagnostics=diagnostics,
            faith_fig=faith_fig,
            faith_df=faith_df,
            mechanistic_summary=mech_summary,
            mechanistic_table=mech_table,
            mechanistic_payload=mech_payload,
            mechanistic_layer_arrays=mech_layers,
        )

        return (
            result.global_score,
            overlay,
            align_fig,
            token_fig,
            token_df,
            diagnostics,
            faith_fig,
            faith_df,
            mech_summary,
            mech_table,
            mech_json,
            explore_bundle,
        )
    except gr.Error:
        raise
    except Exception as exc:
        if "OutOfMemoryError" in type(exc).__name__ or "CUDA out of memory" in str(exc):
            raise gr.Error("GPU out of memory. Try a smaller model or precision='float16' / 'bfloat16'.") from exc
        raise gr.Error(f"Analysis failed: {exc}") from exc


def run_compare(
    family: str,
    model_name: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    image: Image.Image | None,
    prompt_a: str,
    prompt_b: str,
):
    try:
        _validate_inputs(image, prompt_a)
        _validate_inputs(image, prompt_b)
        assert image is not None

        preflight_lines, preflight_notes = _collect_preflight_lines(
            family=family,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )

        res_a, res_b = PIPELINE.compare_prompts(
            family=family,
            model_name=model_name,
            image=image.convert("RGB"),
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        overlay_a = overlay_heatmap(image, res_a.attention_maps["vision_rollout"], alpha=0.42, colormap="inferno")
        overlay_b = overlay_heatmap(image, res_b.attention_maps["vision_rollout"], alpha=0.42, colormap="inferno")

        token_fig_a = plot_token_scores(res_a.tokens, res_a.token_scores)
        token_fig_b = plot_token_scores(res_b.tokens, res_b.token_scores)

        compare_df = pd.DataFrame(
            [
                {
                    "prompt": prompt_a,
                    "global_score": res_a.global_score,
                    "num_tokens": len(res_a.tokens),
                    "num_patches": res_a.patch_grid[0] * res_a.patch_grid[1],
                },
                {
                    "prompt": prompt_b,
                    "global_score": res_b.global_score,
                    "num_tokens": len(res_b.tokens),
                    "num_patches": res_b.patch_grid[0] * res_b.patch_grid[1],
                },
            ]
        )

        delta = res_a.global_score - res_b.global_score
        summary = (
            "\n".join(preflight_lines)
            + "\n"
            f"Prompt A score: `{res_a.global_score:.6f}`\n"
            f"Prompt B score: `{res_b.global_score:.6f}`\n"
            f"Delta (A-B): `{delta:.6f}`"
        )
        if preflight_notes:
            summary += "\n" + "\n".join(preflight_notes)

        bundle = _export_compare_bundle(
            family=family,
            model_name=model_name,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            compare_df=compare_df,
            overlay_a=overlay_a,
            overlay_b=overlay_b,
            token_fig_a=token_fig_a,
            token_fig_b=token_fig_b,
            summary=summary,
        )

        return compare_df, overlay_a, overlay_b, token_fig_a, token_fig_b, summary, bundle
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Comparison failed: {exc}") from exc


def run_eval(
    dataset_path: str,
    dataset_file: str | None,
    family: str,
    model_name: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    compute_gradients: bool,
    run_faithfulness: bool,
):
    try:
        effective_path = dataset_file if dataset_file else dataset_path
        if not effective_path:
            raise gr.Error("Please enter a JSONL dataset path or upload a JSONL file.")

        path = Path(effective_path).expanduser()
        if not path.exists():
            raise gr.Error(f"Dataset file not found: {path}")

        preflight_lines, preflight_notes = _collect_preflight_lines(
            family=family,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )

        df = EVAL_RUNNER.run(
            dataset_path=path,
            family=family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            compute_gradients=compute_gradients,
            run_faithfulness=run_faithfulness,
        )

        summary_lines = [
            *preflight_lines,
            f"Rows evaluated: `{len(df)}`",
            f"Mean global score: `{df['global_score'].mean():.6f}`" if not df.empty else "No rows evaluated.",
        ]

        if "counterfactual_drop" in df.columns:
            summary_lines.append(f"Mean counterfactual drop: `{df['counterfactual_drop'].mean():.6f}`")
        if "attn_grad_spearman" in df.columns:
            summary_lines.append(f"Mean attn-grad spearman: `{df['attn_grad_spearman'].mean():.6f}`")
        summary_lines.extend(preflight_notes)

        summary = "\n".join(summary_lines)
        bundle = _export_eval_bundle(df=df, summary=summary, dataset_path=path, family=family, model_name=model_name)

        return df, summary, bundle
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Evaluation failed: {exc}") from exc


def _bind_family_to_default(family: str):
    return gr.update(value=_default_model_for_family(family))


def build_app() -> gr.Blocks:
    with gr.Blocks(title="MultimodalLens v0.2.0", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# MultimodalLens 🔍\n"
            "*Mechanistic Interpretability Toolkit for Vision-Language Models*\n\n"
            "Analyze attention rollout, token-patch alignment, logit lens predictions, "
            "activation patching, and visual grounding circuits."
        )

        with gr.Tab("Explore"):
            with gr.Row():
                model_name = gr.Textbox(
                    value=DEFAULT_MODELS["auto"],
                    label="Model Checkpoint",
                    placeholder="e.g. openai/clip-vit-base-patch32, llava-hf/llava-1.5-7b-hf",
                    scale=3,
                )
                family = gr.Dropdown(
                    choices=SUPPORTED_FAMILIES,
                    value="auto",
                    label="Model Family",
                    allow_custom_value=True,
                    scale=1,
                )

            with gr.Accordion("Model & Runtime Settings", open=False):
                with gr.Row():
                    device = gr.Dropdown(choices=["auto", "cpu", "cuda", "mps"], value="auto", label="Device")
                    dtype = gr.Dropdown(
                        choices=["float16", "bfloat16", "float32"],
                        value="float16",
                        label="Precision",
                    )
                    trust_remote_code = gr.Checkbox(value=False, label="trust_remote_code")

            with gr.Row():
                image = gr.Image(type="pil", label="Input Image")
                prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Describe the image or ask a question...")

            with gr.Row():
                compute_gradients = gr.Checkbox(value=False, label="Compute gradient attributions")
                run_faithfulness = gr.Checkbox(value=False, label="Run faithfulness diagnostics")
                run_mechanistic_suite = gr.Checkbox(value=False, label="Run mechanistic suite (4 probes)")

            with gr.Accordion("Advanced Mechanistic Probes Options", open=False):
                with gr.Row():
                    mech_layer_index = gr.Number(value=12, precision=0, label="Patch Layer Index")
                    mech_top_k = gr.Number(value=20, precision=0, label="Top-K (logit lens/grounding)")
                    mech_mask_fraction = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Mask Fraction (patch/grounding)",
                    )
                    mech_visual_only = gr.Checkbox(value=True, label="Visual-only patching when supported")

                with gr.Row():
                    mech_patterns = gr.Textbox(
                        label="Layer Patterns (regex; comma/newline separated)",
                        lines=2,
                        placeholder="language_model\\.model\\.layers, vision_model\\.encoder\\.layers",
                    )
                    mech_positions = gr.Textbox(label="Logit Lens Positions", value="-1")
                    mech_max_tokens = gr.Number(value=256, precision=0, label="Cache Max Tokens")
                    mech_max_layers = gr.Number(value=0, precision=0, label="Logit Lens Max Layers (0 = all)")

            run_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                global_score = gr.Number(label="Global Score")
                diagnostics = gr.Markdown(label="Diagnostics")

            with gr.Row():
                attention_overlay = gr.Image(type="numpy", label="Attention Overlay")
                alignment_plot = gr.Plot(label="Token-Image Alignment")

            with gr.Row():
                token_plot = gr.Plot(label="Per-token Similarity")
                token_table = gr.Dataframe(label="Token Scores", interactive=False)

            with gr.Row():
                faith_plot = gr.Plot(label="Faithfulness Curves")
                faith_table = gr.Dataframe(label="Faithfulness Metrics", interactive=False)

            with gr.Row():
                mech_summary = gr.Markdown(label="Mechanistic Summary")

            with gr.Row():
                mech_table = gr.Dataframe(label="Mechanistic Metrics", interactive=False)
                mech_json = gr.Textbox(label="Mechanistic JSON Preview", lines=12)

            explore_bundle = gr.File(label="Download Explore Bundle")

            family.change(_bind_family_to_default, inputs=[family], outputs=[model_name])
            run_btn.click(
                run_explore,
                inputs=[
                    family,
                    model_name,
                    device,
                    dtype,
                    trust_remote_code,
                    compute_gradients,
                    run_faithfulness,
                    run_mechanistic_suite,
                    mech_patterns,
                    mech_positions,
                    mech_layer_index,
                    mech_top_k,
                    mech_max_tokens,
                    mech_max_layers,
                    mech_mask_fraction,
                    mech_visual_only,
                    image,
                    prompt,
                ],
                outputs=[
                    global_score,
                    attention_overlay,
                    alignment_plot,
                    token_plot,
                    token_table,
                    diagnostics,
                    faith_plot,
                    faith_table,
                    mech_summary,
                    mech_table,
                    mech_json,
                    explore_bundle,
                ],
            )

        with gr.Tab("Compare"):
            with gr.Row():
                c_model = gr.Textbox(value=DEFAULT_MODELS["auto"], label="Model Checkpoint", scale=3)
                c_family = gr.Dropdown(
                    choices=SUPPORTED_FAMILIES,
                    value="auto",
                    label="Model Family",
                    allow_custom_value=True,
                    scale=1,
                )

            with gr.Accordion("Runtime Settings", open=False):
                with gr.Row():
                    c_device = gr.Dropdown(choices=["auto", "cpu", "cuda", "mps"], value="auto", label="Device")
                    c_dtype = gr.Dropdown(
                        choices=["float16", "bfloat16", "float32"],
                        value="float16",
                        label="Precision",
                    )
                    c_trust_remote_code = gr.Checkbox(value=False, label="trust_remote_code")

            c_image = gr.Image(type="pil", label="Input Image")
            with gr.Row():
                prompt_a = gr.Textbox(label="Prompt A", lines=3)
                prompt_b = gr.Textbox(label="Prompt B", lines=3)
            c_run = gr.Button("Compare Prompts", variant="primary")

            compare_table = gr.Dataframe(label="Comparison Summary", interactive=False)
            with gr.Row():
                overlay_a = gr.Image(type="numpy", label="Prompt A Overlay")
                overlay_b = gr.Image(type="numpy", label="Prompt B Overlay")
            with gr.Row():
                token_fig_a = gr.Plot(label="Prompt A Token Similarity")
                token_fig_b = gr.Plot(label="Prompt B Token Similarity")
            compare_summary = gr.Markdown(label="Score Delta")
            compare_bundle = gr.File(label="Download Compare Bundle")

            c_family.change(_bind_family_to_default, inputs=[c_family], outputs=[c_model])
            c_run.click(
                run_compare,
                inputs=[
                    c_family,
                    c_model,
                    c_device,
                    c_dtype,
                    c_trust_remote_code,
                    c_image,
                    prompt_a,
                    prompt_b,
                ],
                outputs=[compare_table, overlay_a, overlay_b, token_fig_a, token_fig_b, compare_summary, compare_bundle],
            )

        with gr.Tab("Eval"):
            with gr.Row():
                e_model = gr.Textbox(value=DEFAULT_MODELS["auto"], label="Model Checkpoint", scale=3)
                e_family = gr.Dropdown(
                    choices=SUPPORTED_FAMILIES,
                    value="auto",
                    label="Model Family",
                    allow_custom_value=True,
                    scale=1,
                )

            with gr.Accordion("Runtime Settings", open=False):
                with gr.Row():
                    e_device = gr.Dropdown(choices=["auto", "cpu", "cuda", "mps"], value="auto", label="Device")
                    e_dtype = gr.Dropdown(
                        choices=["float16", "bfloat16", "float32"],
                        value="float16",
                        label="Precision",
                    )
                    e_trust = gr.Checkbox(value=False, label="trust_remote_code")

            with gr.Row():
                dataset_path = gr.Textbox(
                    label="Dataset JSONL Path",
                    placeholder="/absolute/path/to/dataset.jsonl",
                    scale=2,
                )
                dataset_file = gr.File(
                    label="Upload Dataset JSONL File",
                    file_types=[".jsonl"],
                    scale=2,
                )

            with gr.Row():
                e_grad = gr.Checkbox(value=False, label="Compute gradients")
                e_faith = gr.Checkbox(value=False, label="Run faithfulness")

            e_run = gr.Button("Run Eval", variant="primary")

            eval_df = gr.Dataframe(label="Evaluation Results", interactive=False)
            eval_summary = gr.Markdown(label="Summary")
            eval_bundle = gr.File(label="Download Eval Bundle")

            e_family.change(_bind_family_to_default, inputs=[e_family], outputs=[e_model])
            e_run.click(
                run_eval,
                inputs=[
                    dataset_path,
                    dataset_file,
                    e_family,
                    e_model,
                    e_device,
                    e_dtype,
                    e_trust,
                    e_grad,
                    e_faith,
                ],
                outputs=[eval_df, eval_summary, eval_bundle],
            )

    return demo
