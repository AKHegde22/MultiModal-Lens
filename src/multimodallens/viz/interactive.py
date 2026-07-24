"""Interactive HTML dashboard generator for notebook, web, and export rendering."""

from __future__ import annotations

import json
from typing import Any
from multimodallens.analysis.dla import DLAResult
from multimodallens.analysis.path_patching import PathPatchingResult
from multimodallens.types import AnalysisResult, LogitLensResult


def create_dla_waterfall_html(dla_result: DLAResult) -> str:
    """Generate interactive HTML/JS waterfall plot for Direct Logit Attribution."""
    data = dla_result.to_dict()
    data_json = json.dumps(data)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Multimodal DLA Waterfall: {dla_result.target_token}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h2 {{ margin-top: 0; color: #38bdf8; }}
        .bar-container {{ margin-bottom: 12px; }}
        .bar-label {{ font-size: 13px; font-weight: 600; font-family: monospace; color: #94a3b8; margin-bottom: 4px; display: flex; justify-content: space-between; }}
        .bar-track {{ background: #334155; border-radius: 6px; height: 24px; overflow: hidden; position: relative; display: flex; align-items: center; }}
        .bar-fill {{ height: 100%; transition: width 0.4s ease; border-radius: 4px; }}
        .bar-positive {{ background: linear-gradient(90deg, #0ea5e9, #38bdf8); }}
        .bar-negative {{ background: linear-gradient(90deg, #f43f5e, #fb7185); }}
        .bar-val {{ position: absolute; right: 10px; font-size: 12px; font-weight: bold; color: #ffffff; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>🔍 Direct Logit Attribution for Target: "{dla_result.target_token}"</h2>
        <p style="color: #94a3b8; font-size: 14px;">Model: {dla_result.model_name} ({dla_result.model_family})</p>
        <div id="dla-stats" style="color: #94a3b8; font-size: 14px; margin-bottom: 12px; font-weight: bold;"></div>
        <div id="dla-bars"></div>
    </div>
    <script>
        const data = {data_json};
        
        const statsContainer = document.getElementById('dla-stats');
        if (data.total_logit !== undefined) {{
            statsContainer.innerHTML = `Total Logit: ${{data.total_logit.toFixed(4)}} &nbsp;|&nbsp; Residual Error: ${{data.residual_error ? data.residual_error.toFixed(4) : 0}}`;
        }}
        
        const container = document.getElementById('dla-bars');
        const maxScore = Math.max(...data.contributions.map(c => Math.abs(c.contribution_score)), 1e-5);

        data.contributions.forEach(c => {{
            const pct = Math.min(100, Math.max(5, (Math.abs(c.contribution_score) / maxScore) * 100));
            const isPos = c.contribution_score >= 0;
            const fillClass = isPos ? 'bar-positive' : 'bar-negative';
            
            let headsHtml = '';
            if (c.head_contributions && c.head_contributions.length > 0) {{
                headsHtml = '<div style="margin-left: 12px; font-size: 11px; margin-top: 6px; display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 4px;">';
                c.head_contributions.forEach(h => {{
                    headsHtml += `<div style="background: #334155; padding: 2px 6px; border-radius: 4px; display: flex; justify-content: space-between;"><span>H${{h.head_index !== undefined ? h.head_index : h.head}}</span><span style="color: ${{h.contribution_score >= 0 ? '#38bdf8' : '#fb7185'}}">${{h.contribution_score.toFixed(4)}}</span></div>`;
                }});
                headsHtml += '</div>';
            }}

            const item = document.createElement('div');
            item.className = 'bar-container';
            item.innerHTML = `
                <div class="bar-label">
                    <span>${{c.layer_name}}</span>
                    <span>Norm: ${{c.norm.toFixed(2)}}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill ${{fillClass}}" style="width: ${{pct}}%;"></div>
                    <span class="bar-val">${{c.contribution_score.toFixed(4)}}</span>
                </div>
                ${{headsHtml}}
            `;
            container.appendChild(item);
        }});
    </script>
</body>
</html>"""


def create_path_patching_html(path_result: PathPatchingResult) -> str:
    """Generate interactive heatmap matrix HTML for causal path patching."""
    data = path_result.to_dict()
    data_json = json.dumps(data)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Causal Path Patching Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; }}
        h2 {{ color: #a855f7; margin-top: 0; }}
        .matrix {{ display: grid; gap: 8px; margin-top: 20px; }}
        .cell {{ padding: 12px; border-radius: 6px; text-align: center; font-size: 13px; font-weight: bold; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>⚡ Causal Path Patching Intervention Matrix</h2>
        <p style="color: #94a3b8;">Prompt: "{path_result.prompt}"</p>
        <div id="matrix-container"></div>
    </div>
    <script>
        const data = {data_json};
        const container = document.getElementById('matrix-container');
        
        data.effects.forEach(eff => {{
            const div = document.createElement('div');
            const score = eff.causal_effect;
            const alpha = Math.min(1, Math.max(0.1, Math.abs(score)));
            const bg = score >= 0 ? `rgba(168, 85, 247, ${{alpha}})` : `rgba(239, 68, 68, ${{alpha}})`;
            div.className = 'cell';
            div.style.background = bg;
            
            let sender = eff.sender_layer;
            if (eff.sender_head !== undefined && eff.sender_head !== null) sender += `.H${{eff.sender_head}}`;
            
            let receiver = eff.receiver_layer;
            if (eff.receiver_head !== undefined && eff.receiver_head !== null) receiver += `.H${{eff.receiver_head}}`;
            
            let channelInfo = '';
            if (eff.receiver_channel) channelInfo = `<br><span style="color:#d8b4fe; font-size:11px;">Channel: ${{eff.receiver_channel}}</span>`;
            
            div.innerHTML = `${{sender}} ➔ ${{receiver}}<br>Effect: ${{score.toFixed(3)}}${{channelInfo}}`;
            container.appendChild(div);
        }});
    </script>
</body>
</html>"""


def create_dla_plotly_figure(dla_result: DLAResult) -> Any:
    """Generate a Plotly Figure for Direct Logit Attribution (DLA)."""
    import plotly.graph_objects as go

    data = dla_result.to_dict()
    contributions = data.get("contributions", [])

    labels = [c["layer_name"] for c in contributions]
    scores = [c["contribution_score"] for c in contributions]
    colors = ["#0ea5e9" if s >= 0 else "#f43f5e" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{s:.4f}" for s in scores],
            textposition="auto",
        )
    )

    title_text = (
        f"Direct Logit Attribution: '{dla_result.target_token}'<br>"
        f"<sup>Model: {dla_result.model_name} | Total Logit: {dla_result.total_logit:.4f} | Error: {dla_result.residual_error:.4f}</sup>"
    )

    fig.update_layout(
        title=title_text,
        xaxis_title="Logit Contribution Score",
        yaxis_title="Component / Layer",
        template="plotly_dark",
        margin=dict(l=150, r=40, t=80, b=40),
    )
    return fig


def create_path_patching_plotly_figure(path_result: PathPatchingResult) -> Any:
    """Generate a Plotly Figure heatmap for Causal Path Patching."""
    import plotly.graph_objects as go

    data = path_result.to_dict()
    effects = data.get("effects", [])

    if not effects:
        fig = go.Figure()
        fig.update_layout(title="No Path Patching Effects Available", template="plotly_dark")
        return fig

    senders = sorted(list({e["sender_layer"] for e in effects}))
    receivers = sorted(list({e["receiver_layer"] for e in effects}))

    z_matrix = []
    text_matrix = []

    for r in receivers:
        z_row = []
        text_row = []
        for s in senders:
            matching = [
                e for e in effects if e["sender_layer"] == s and e["receiver_layer"] == r
            ]
            if matching:
                avg_effect = sum(m["causal_effect"] for m in matching) / len(matching)
                z_row.append(avg_effect)
                text_row.append(f"Effect: {avg_effect:.3f}<br>({len(matching)} channels)")
            else:
                z_row.append(0.0)
                text_row.append("N/A")
        z_matrix.append(z_row)
        text_matrix.append(text_row)

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=senders,
            y=receivers,
            text=text_matrix,
            hoverinfo="text",
            colorscale="RdBu",
            reversescale=True,
            zmid=0,
        )
    )

    fig.update_layout(
        title=f"Causal Path Patching Matrix<br><sup>Prompt: {path_result.prompt}</sup>",
        xaxis_title="Sender Component",
        yaxis_title="Receiver Component",
        template="plotly_dark",
    )
    return fig


def create_interactive_dashboard_html(
    analysis: AnalysisResult | None = None,
    dla: DLAResult | None = None,
    path_patch: PathPatchingResult | None = None,
    logit_lens: LogitLensResult | None = None,
) -> str:
    """Combine analysis results into a single multi-tab interactive dashboard HTML."""
    tabs_html = ""
    if dla:
        tabs_html += create_dla_waterfall_html(dla)
    if path_patch:
        tabs_html += create_path_patching_html(path_patch)
    if not tabs_html:
        tabs_html = "<div>No interactive components were generated.</div>"
    return tabs_html


def create_logit_lens_plotly_figure(logit_lens_result: LogitLensResult) -> Any:
    """Generate an interactive Plotly heatmap for Logit Lens predictions across layers."""
    import plotly.graph_objects as go

    if hasattr(logit_lens_result, "to_dict") and callable(getattr(logit_lens_result, "to_dict")):
        data = logit_lens_result.to_dict()
    else:
        from dataclasses import asdict
        data = asdict(logit_lens_result)

    steps = data.get("steps", [])

    if not steps:
        fig = go.Figure()
        fig.update_layout(title="No Logit Lens Data Available", template="plotly_dark")
        return fig

    layers = [s["layer_name"] for s in steps]
    top_tokens = [s.get("top_tokens", ["?"])[0] if s.get("top_tokens") else "?" for s in steps]
    top_probs = [
        s.get("top_probabilities", s.get("top_probs", [0.0]))[0]
        if (s.get("top_probabilities") or s.get("top_probs"))
        else 0.0
        for s in steps
    ]

    text_matrix = [[f"Token: '{t}'<br>Prob: {p:.4f}"] for t, p in zip(top_tokens, top_probs)]
    z_matrix = [[p] for p in top_probs]

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            y=layers,
            x=["Rank 1 Prediction"],
            text=text_matrix,
            hoverinfo="text",
            colorscale="Viridis",
            showscale=True,
        )
    )

    fig.update_layout(
        title=f"Multimodal Logit Lens Trajectory<br><sup>Prompt: '{logit_lens_result.prompt}'</sup>",
        xaxis_title="Top Predicted Token",
        yaxis_title="Layer",
        template="plotly_dark",
    )
    return fig


def create_alignment_plotly_figure(
    alignment_matrix: Any,
    tokens: list[str],
    patch_labels: list[str] | None = None,
) -> Any:
    """Generate an interactive Plotly heatmap for text-token to vision-patch alignment."""
    import numpy as np
    import plotly.graph_objects as go

    matrix = np.asarray(alignment_matrix)
    if matrix.ndim != 2:
        fig = go.Figure()
        fig.update_layout(title="Invalid Alignment Matrix Shape", template="plotly_dark")
        return fig

    num_tokens, num_patches = matrix.shape
    y_labels = tokens[:num_tokens] if len(tokens) >= num_tokens else [f"Tok {i}" for i in range(num_tokens)]
    x_labels = patch_labels[:num_patches] if patch_labels and len(patch_labels) >= num_patches else [f"Patch {j}" for j in range(num_patches)]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale="Plasma",
            hoverinfo="x+y+z",
        )
    )

    fig.update_layout(
        title="Cross-Modal Token-Patch Alignment Heatmap",
        xaxis_title="Vision Patches",
        yaxis_title="Text Tokens",
        template="plotly_dark",
    )
    return fig


def create_attention_plotly_figure(
    attn_matrix: Any,
    source_labels: list[str] | None = None,
    target_labels: list[str] | None = None,
    title: str = "Attention Pattern Map",
) -> Any:
    """Generate an interactive Plotly heatmap for attention pattern maps."""
    import numpy as np
    import plotly.graph_objects as go

    matrix = np.asarray(attn_matrix)
    if matrix.ndim != 2:
        fig = go.Figure()
        fig.update_layout(title="Invalid Attention Matrix Shape", template="plotly_dark")
        return fig

    num_src, num_tgt = matrix.shape
    y_labels = source_labels[:num_src] if source_labels and len(source_labels) >= num_src else [f"Src {i}" for i in range(num_src)]
    x_labels = target_labels[:num_tgt] if target_labels and len(target_labels) >= num_tgt else [f"Tgt {j}" for j in range(num_tgt)]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale="Cividis",
            hoverinfo="x+y+z",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Key / Target Tokens",
        yaxis_title="Query / Source Tokens",
        template="plotly_dark",
    )
    return fig


def create_residual_trajectory_plotly_figure(activations_run: Any) -> Any:
    """Generate a 2D PCA scatter plot showing residual stream representation trajectory across layers."""
    import numpy as np
    import plotly.graph_objects as go

    layers = getattr(activations_run, "layers", [])
    if not layers:
        fig = go.Figure()
        fig.update_layout(title="No Layer Activations Available for Trajectory", template="plotly_dark")
        return fig

    layer_names = []
    vectors = []

    for layer in layers:
        val = np.asarray(layer.values)
        if val.size == 0:
            continue
        # Mean pool over batch & sequence dimensions to get a single vector per layer
        while val.ndim > 1:
            val = val.mean(axis=0)
        layer_names.append(layer.layer_name)
        vectors.append(val)

    if len(vectors) < 2:
        fig = go.Figure()
        fig.update_layout(title="Insufficient Layers for Trajectory SVD", template="plotly_dark")
        return fig

    matrix = np.stack(vectors, axis=0)  # [num_layers, d_model]
    matrix_centered = matrix - matrix.mean(axis=0, keepdims=True)

    # Compute 2D SVD / PCA
    try:
        U, S, _ = np.linalg.svd(matrix_centered, full_matrices=False)
        coords = U[:, :2] * S[:2]
    except Exception:
        coords = np.zeros((len(vectors), 2))

    x_coords = coords[:, 0]
    y_coords = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(x_coords)

    fig = go.Figure()

    # Trajectory line
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines+markers+text",
            text=[f"L{i}" for i in range(len(layer_names))],
            textposition="top center",
            marker=dict(
                size=10,
                color=list(range(len(layer_names))),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Layer Index"),
            ),
            line=dict(color="#38bdf8", width=2),
            hovertext=layer_names,
            hoverinfo="text+x+y",
        )
    )

    fig.update_layout(
        title="Residual Stream State Trajectory Across Layers (2D PCA)",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        template="plotly_dark",
    )
    return fig


