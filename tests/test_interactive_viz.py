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
