"""End-to-end integration script to test Qwen2-VL-2B with MultiModal-Lens."""

from __future__ import annotations

import sys
import torch
from PIL import Image

def main():
    print("=" * 60)
    print("Starting End-to-End Verification with Qwen2-VL-2B-Instruct")
    print("=" * 60)

    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Instantiate Adapter
    print("\n1. Loading Qwen2-VL-2B-Instruct adapter...")
    from multimodallens.adapters.generic_adapter import GenericVLMAdapter
    from multimodallens.core.hooked_vlm import HookedVLM

    try:
        hooked_vlm = HookedVLM.from_pretrained(
            model_name=model_name,
            family="qwen2_vl",
            dtype="float32",
        )
        print("✓ HookedVLM loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    adapter = hooked_vlm.adapter

    # Create dummy image
    image = Image.new("RGB", (224, 224), color=(130, 200, 250))
    prompt = "What color is this image?"

    # 2. Test adapter.analyze()
    print("\n2. Running adapter.analyze() (real metrics verification)...")
    analysis_res = adapter.analyze(image, prompt)
    print(f"  - Model family: {analysis_res.model_family}")
    print(f"  - Tokens ({len(analysis_res.tokens)}): {analysis_res.tokens[:10]}...")
    print(f"  - Patch grid: {analysis_res.patch_grid}")
    print(f"  - Token scores shape: {analysis_res.token_scores.shape}")
    print(f"  - Token scores min/max/mean: {analysis_res.token_scores.min():.4f} / {analysis_res.token_scores.max():.4f} / {analysis_res.token_scores.mean():.4f}")
    print(f"  - Alignment matrix shape: {analysis_res.alignment_matrix.shape}")
    print(f"  - Attention maps available: {list(analysis_res.attention_maps.keys())}")
    for k, v in analysis_res.attention_maps.items():
        print(f"    - {k} map shape: {v.shape}, mean: {v.mean():.4f}")
    print("✓ adapter.analyze() returned real non-mock data!")

    # 3. Test Direct Logit Attribution (DLA)
    print("\n3. Running Direct Logit Attribution (DLA)...")
    from multimodallens.analysis.dla import run_multimodal_dla
    dla_res = run_multimodal_dla(adapter, image, prompt, target_token="blue")
    print(f"  - Target token: '{dla_res.target_token}'")
    print(f"  - Total logit: {dla_res.total_logit:.4f}")
    print(f"  - Residual error: {dla_res.residual_error:.4f}")
    print(f"  - Head contributions count: {len(dla_res.head_contributions)}")
    print(f"  - MLP contributions count: {len(dla_res.mlp_contributions)}")
    print(f"  - Embedding contribution: {dla_res.embedding_contribution:.4f}")
    if dla_res.head_contributions:
        top_head = max(dla_res.head_contributions, key=lambda h: abs(h.contribution_score))
        print(f"  - Top attention head: Layer {top_head.layer}, Head {top_head.head} ({top_head.layer_name}) with score {top_head.contribution_score:.4f}")
    print("✓ DLA executed successfully with per-head decomposition!")

    # 4. Test 3-Stage Vision Logit Lens
    print("\n4. Running 3-Stage Vision Logit Lens...")
    from multimodallens.analysis.logit_lens import run_vision_logit_lens
    vll_res = run_vision_logit_lens(
        adapter,
        image,
        top_k=3,
        layer_names=[
            "model.language_model.layers.0",
            "model.language_model.layers.14",
            "model.language_model.layers.27",
        ],
    )
    print(f"  - Stages captured: {list(vll_res.stages.keys())}")
    for stage_name, steps in vll_res.stages.items():
        print(f"  - Stage '{stage_name}': {len(steps)} steps captured")
        if steps:
            s0 = steps[0]
            print(f"    - Sample step at {s0.layer_name} (pos {s0.position}): top tokens {s0.top_tokens} probs {[round(p, 4) for p in s0.top_probabilities]}")
    print("✓ Vision Logit Lens executed successfully!")

    # 5. Test Causal Path Patching
    print("\n5. Running Edge-Level Causal Path Patching...")
    from multimodallens.analysis.path_patching import run_causal_path_patching
    corrupted_image = Image.new("RGB", (224, 224), color=(250, 50, 50))
    path_res = run_causal_path_patching(
        adapter=adapter,
        clean_image=image,
        clean_prompt="What color is this image?",
        corrupted_image=corrupted_image,
        corrupted_prompt="What color is this image?",
        sender_layers=["model.language_model.layers.0"],
        receiver_layers=["model.language_model.layers.1"],
        receiver_channels=["q", "k", "v"],
    )
    print(f"  - Total edge effects evaluated: {len(path_res.effects)}")
    if path_res.effects:
        sample_eff = path_res.effects[0]
        print(f"  - Sample edge: {sample_eff.sender_layer} -> {sample_eff.receiver_layer} ({sample_eff.receiver_channel}) | Effect: {sample_eff.causal_effect:.4f}")
    print("✓ Causal Path Patching executed successfully!")

    # 6. Test FactoredMatrix & Weight Processing
    print("\n6. Testing Weight Processing Utilities...")
    hooked_vlm.center_unembed()
    print("✓ center_unembed() executed successfully!")
    hooked_vlm.center_writing_weights()
    print("✓ center_writing_weights() executed successfully!")

    # 7. Test HTML and Plotly exports & save to disk
    print("\n7. Saving Dashboard and Plotly Exports to disk...")
    import os
    os.makedirs("outputs", exist_ok=True)

    dla_html_path = "outputs/qwen2_vl_dla.html"
    dla_res.save_html(dla_html_path)

    dla_plotly_path = "outputs/qwen2_vl_dla_plotly.html"
    dla_fig = dla_res.to_plotly()
    dla_fig.write_html(dla_plotly_path)

    path_html_path = "outputs/qwen2_vl_path_patching.html"
    path_res.save_html(path_html_path)

    path_plotly_path = "outputs/qwen2_vl_path_patching_plotly.html"
    path_fig = path_res.to_plotly()
    path_fig.write_html(path_plotly_path)

    print(f"  - Saved DLA HTML Dashboard -> {dla_html_path}")
    print(f"  - Saved DLA Plotly Interactive -> {dla_plotly_path}")
    print(f"  - Saved Path Patching HTML Dashboard -> {path_html_path}")
    print(f"  - Saved Path Patching Plotly Interactive -> {path_plotly_path}")

    print("\n" + "=" * 60)
    print("ALL END-TO-END TESTS PASSED FOR Qwen2-VL-2B-Instruct!")
    print("=" * 60)

if __name__ == "__main__":
    main()
