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
        <div id="dla-bars"></div>
    </div>
    <script>
        const data = {data_json};
        const container = document.getElementById('dla-bars');
        const maxScore = Math.max(...data.contributions.map(c => Math.abs(c.contribution_score)), 1e-5);

        data.contributions.forEach(c => {{
            const pct = Math.min(100, Math.max(5, (Math.abs(c.contribution_score) / maxScore) * 100));
            const isPos = c.contribution_score >= 0;
            const fillClass = isPos ? 'bar-positive' : 'bar-negative';

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
            div.innerHTML = `${{eff.sender_layer}} ➔ ${{eff.receiver_layer}}<br>Effect: ${{score.toFixed(3)}}`;
            container.appendChild(div);
        }});
    </script>
</body>
</html>"""


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
