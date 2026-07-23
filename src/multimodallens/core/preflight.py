"""Model compatibility preflight checks for publish/readiness workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import AutoConfig

from multimodallens.core.registry import resolve_family


MULTIMODAL_MODEL_TYPE_HINTS = {
    "clip",
    "siglip",
    "siglip2",
    "altclip",
    "xclip",
    "blip2",
    "instructblip",
    "llava",
    "llava_next",
    "llava_onevision",
    "qwen2_vl",
    "qwen2_5_vl",
    "idefics2",
    "idefics3",
    "paligemma",
    "mllama",
    "internvl",
    "minicpmv",
    "smolvlm",
    "kosmos2",
    "florence2",
}


@dataclass(slots=True)
class ModelPreflightReport:
    """Result of compatibility checks before running heavy model analysis."""

    requested_family: str
    resolved_family: str
    model_name: str
    model_type: str | None
    architectures: list[str] = field(default_factory=list)
    has_vision_tower: bool = False
    requires_trust_remote_code: bool = False
    supports_explore: bool = False
    supports_compare: bool = False
    supports_eval: bool = False
    supports_layers: bool = False
    supports_cache: bool = False
    supports_patch: bool = False
    supports_logit_lens: bool = False
    supports_grounding: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _normalize_model_type(value: object) -> str:
    return str(value or "").lower().replace("-", "_")


def _infer_has_vision(cfg: object, model_type: str, architectures: list[str]) -> bool:
    if getattr(cfg, "vision_config", None) is not None:
        return True
    if "vision" in model_type or "vl" in model_type:
        return True
    if model_type in MULTIMODAL_MODEL_TYPE_HINTS:
        return True
    return any("vision" in arch.lower() or "vl" in arch.lower() for arch in architectures)


def run_model_preflight(
    family: str,
    model_name: str,
    trust_remote_code: bool = False,
) -> ModelPreflightReport:
    """Run a lightweight config-based compatibility check for selected model/family."""
    try:
        resolved_family = resolve_family(
            family=family,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        resolved_family = "unknown"

    report = ModelPreflightReport(
        requested_family=family,
        resolved_family=resolved_family,
        model_name=model_name,
        model_type=None,
    )

    cfg = None
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception as exc:
        report.errors.append(f"Could not load model config for preflight: {exc}")

    if cfg is not None:
        model_type = _normalize_model_type(getattr(cfg, "model_type", ""))
        architectures = [str(x) for x in getattr(cfg, "architectures", [])]

        report.model_type = model_type or None
        report.architectures = architectures
        report.has_vision_tower = _infer_has_vision(cfg, model_type, architectures)
        report.requires_trust_remote_code = bool(getattr(cfg, "auto_map", None))

        if report.requires_trust_remote_code and not trust_remote_code:
            report.warnings.append(
                "Model declares custom auto_map entries. Set trust_remote_code=True if loading fails."
            )

        if not report.has_vision_tower:
            report.warnings.append(
                "Model config does not clearly expose vision components; this may not be a multimodal checkpoint."
            )

        if report.resolved_family == "llava" and model_type and model_type not in MULTIMODAL_MODEL_TYPE_HINTS:
            report.warnings.append(
                "Family resolved to llava-style decoder path using heuristic fallback; runtime behavior may vary."
            )
    else:
        report.warnings.append(
            "Config lookup failed during preflight; runtime may still work if checkpoint is reachable at run time."
        )

    base_supported = report.has_vision_tower and not report.errors
    report.supports_explore = base_supported
    report.supports_compare = base_supported
    report.supports_eval = base_supported
    report.supports_layers = base_supported
    report.supports_cache = base_supported
    report.supports_patch = base_supported
    report.supports_logit_lens = base_supported
    report.supports_grounding = base_supported

    canonical_families = {"clip", "blip2", "llava", "qwen2_vl", "pixtral", "idefics3", "paligemma"}
    if report.resolved_family not in canonical_families:
        report.errors.append(
            f"Resolved family '{report.resolved_family}' does not map to a canonical adapter implementation."
        )

    return report
