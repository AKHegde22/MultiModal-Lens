import pytest
import plotly.graph_objects as go

from multimodallens.analysis.dla import DLAResult, HeadContribution, MLPContribution
from multimodallens.analysis.path_patching import PathPatchingResult, EdgeEffect
from multimodallens.viz.interactive import (
    create_dla_waterfall_html,
    create_path_patching_html,
    create_dla_plotly_figure,
    create_path_patching_plotly_figure,
    create_interactive_dashboard_html,
)


def test_dla_interactive_viz_html_and_plotly():
    head_contrib = HeadContribution(
        layer=0,
        head=0,
        layer_name="layers.0.self_attn.o_proj",
        contribution_score=0.5,
        norm=1.2,
    )
    mlp_contrib = MLPContribution(
        layer=0,
        layer_name="layers.0.mlp",
        contribution_score=0.3,
        norm=0.8,
    )
    dla_res = DLAResult(
        model_family="llava",
        model_name="toy",
        prompt="a cat",
        target_token="cat",
        head_contributions=[head_contrib],
        mlp_contributions=[mlp_contrib],
        embedding_contribution=0.1,
        total_logit=0.9,
        residual_error=0.0,
    )

    html = dla_res.to_html()
    assert "Direct Logit Attribution" in html
    assert "layers.0.self_attn.o_proj" in html

    fig = dla_res.to_plotly()
    assert isinstance(fig, go.Figure)


def test_path_patching_interactive_viz_html_and_plotly():
    effect = EdgeEffect(
        sender_layer="layers.0",
        sender_head=None,
        receiver_layer="layers.1",
        receiver_head=None,
        receiver_channel="q",
        clean_metric=10.0,
        corrupt_metric=5.0,
        patched_metric=8.0,
        causal_effect=0.6,
    )
    path_res = PathPatchingResult(
        model_family="llava",
        model_name="toy",
        prompt="a cat",
        effects=[effect],
    )

    html = path_res.to_html()
    assert "Causal Path Patching" in html

    fig = path_res.to_plotly()
    assert isinstance(fig, go.Figure)


def test_create_interactive_dashboard_html():
    dashboard = create_interactive_dashboard_html()
    assert "No interactive components" in dashboard


def test_new_plotly_figures():
    from multimodallens.types import LogitLensResult, LogitLensStep, LayerActivationRun, LayerActivation
    from multimodallens.viz.interactive import (
        create_logit_lens_plotly_figure,
        create_alignment_plotly_figure,
        create_attention_plotly_figure,
        create_residual_trajectory_plotly_figure,
    )
    import numpy as np

    # Test Logit Lens Plotly
    ll_step = LogitLensStep(
        layer_name="layer.0",
        position=0,
        top_tokens=["cat", "dog"],
        top_probabilities=[0.8, 0.1],
    )
    ll_res = LogitLensResult(
        model_family="llava",
        model_name="toy",
        prompt="cat",
        steps=[ll_step],
    )
    fig_ll = create_logit_lens_plotly_figure(ll_res)
    assert isinstance(fig_ll, go.Figure)

    # Test Alignment Plotly
    matrix = np.random.randn(3, 4)
    fig_align = create_alignment_plotly_figure(matrix, ["a", "b", "c"])
    assert isinstance(fig_align, go.Figure)

    # Test Attention Plotly
    attn_mat = np.random.rand(4, 4)
    fig_attn = create_attention_plotly_figure(attn_mat)
    assert isinstance(fig_attn, go.Figure)

    # Test Residual Trajectory Plotly
    l0 = LayerActivation(layer_name="layer.0", shape=(1, 16), values=np.random.randn(1, 16))
    l1 = LayerActivation(layer_name="layer.1", shape=(1, 16), values=np.random.randn(1, 16))
    run = LayerActivationRun(
        model_family="llava",
        model_name="toy",
        prompt="cat",
        layers=[l0, l1],
        tokens=["cat"],
    )
    fig_traj = create_residual_trajectory_plotly_figure(run)
    assert isinstance(fig_traj, go.Figure)

