# Visualization - Plotly charts, premium purple-integrated theme

import plotly.graph_objects as go
import numpy as np
import json
import os

try:
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(_base_dir, 'evaluation_results.json'), 'r') as f:
        _eval_data = json.load(f)
except Exception:
    _eval_data = None


# Design tokens aligned with CSS custom properties
_T = {
    "bg": "rgba(0,0,0,0)",
    "paper": "rgba(0,0,0,0)",
    "grid": "rgba(128,0,32,0.06)",
    "grid_major": "rgba(128,0,32,0.09)",
    "text": "#1a1014",
    "dim": "#8a7b82",
    "muted": "#5e4d54",
    "accent": "#800020",
    "accent_bright": "#a61a3c",
    "success": "#34d399",
    "warning": "#fbbf24",
    "danger": "#f87171",
    "card_bg": "#f8f3f5",
}

_CLASS_COLORS = {"High": _T["success"], "Medium": _T["warning"], "Low": _T["danger"]}


def _base_layout(**overrides):
    """Shared Plotly layout — matches dashboard design system."""
    layout = dict(
        paper_bgcolor=_T["paper"],
        plot_bgcolor=_T["bg"],
        font=dict(family="Inter, -apple-system, sans-serif", color=_T["text"], size=12),
        margin=dict(l=12, r=12, t=32, b=12),
        hoverlabel=dict(
            bgcolor="#f0e8eb",
            bordercolor="rgba(128,0,32,0.2)",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#1a1014",
        ),
        dragmode=False,
    )
    layout.update(overrides)
    return layout


# -------- Chart builders --------

def create_probability_bar_chart(classes, probs):
    pcts = [p * 100 for p in probs]
    fig = go.Figure(
        go.Bar(
            y=classes, x=pcts, orientation="h",
            marker=dict(
                color=[_CLASS_COLORS.get(c, _T["accent"]) for c in classes],
                cornerradius=5,
                line=dict(width=0),
            ),
            text=[f"{p:.1f}%" for p in pcts],
            textposition="inside",
            textfont=dict(size=13, family="Inter"),
            hovertemplate="<b>%{y}</b>  %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            height=190,
            xaxis=dict(
                range=[0, 105], showgrid=True, gridcolor=_T["grid"],
                zeroline=False, ticksuffix="%", tickfont=dict(size=10, color=_T["dim"]),
            ),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color=_T["text"])),
        )
    )
    return fig


def create_probability_pie_chart(classes, probs):
    fig = go.Figure(
        go.Pie(
            labels=classes,
            values=[p * 100 for p in probs],
            hole=0.62,
            marker=dict(
                colors=[_CLASS_COLORS.get(c, _T["accent"]) for c in classes],
                line=dict(color="#ffffff", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=11, color="white", family="Inter"),
            hovertemplate="<b>%{label}</b>  %{percent}<extra></extra>",
            pull=[0.015] * len(classes),
            rotation=90,
        )
    )
    fig.update_layout(
        **_base_layout(
            height=290,
            showlegend=False,
            annotations=[dict(
                text="Class<br>Dist.",
                x=0.5, y=0.5, font_size=11, font_color=_T["dim"],
                showarrow=False, font=dict(family="Inter"),
            )],
        )
    )
    return fig


def create_confidence_gauge(confidence, prediction):
    color = _CLASS_COLORS.get(prediction, _T["accent"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number=dict(suffix="%", font=dict(size=34, color=color, family="Inter")),
            gauge=dict(
                axis=dict(
                    range=[0, 100], tickwidth=1,
                    tickcolor=_T["dim"], tickfont=dict(size=9, color=_T["dim"]),
                ),
                bar=dict(color=color, thickness=0.22),
                bgcolor=_T["card_bg"], borderwidth=0,
                steps=[
                    dict(range=[0, 33], color="rgba(248,113,113,0.05)"),
                    dict(range=[33, 66], color="rgba(251,191,36,0.05)"),
                    dict(range=[66, 100], color="rgba(52,211,153,0.05)"),
                ],
                threshold=dict(line=dict(color="white", width=2), thickness=0.65, value=confidence * 100),
            ),
            title=dict(text=f"Confidence -- {prediction}", font=dict(size=11, color=_T["dim"])),
        )
    )
    fig.update_layout(**_base_layout(height=250, margin=dict(l=20, r=20, t=36, b=0)))
    return fig


def create_feature_importance_chart(features, importances):
    f_rev, i_rev = features[::-1], importances[::-1]
    n = len(f_rev)
    colors = [f"rgba(139,92,246,{0.30 + 0.70 * i / max(n - 1, 1):.2f})" for i in range(n)]
    fig = go.Figure(
        go.Bar(
            y=f_rev, x=i_rev, orientation="h",
            marker=dict(color=colors, cornerradius=3, line=dict(width=0)),
            text=[f"{v:.4f}" for v in i_rev],
            textposition="outside",
            textfont=dict(size=9.5, color=_T["dim"], family="Inter"),
            hovertemplate="<b>%{y}</b>  %{x:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            height=max(280, n * 22 + 60),
            xaxis=dict(showgrid=True, gridcolor=_T["grid"], zeroline=False,
                       tickfont=dict(size=9, color=_T["dim"])),
            yaxis=dict(showgrid=False, tickfont=dict(size=10.5, color=_T["muted"])),
        )
    )
    return fig


def create_metrics_bar_chart(metrics):
    names = list(metrics.keys())
    vals = [v * 100 for v in metrics.values()]
    colors = ["#800020", "#a61a3c", "#d5c2c7", "#34d399"][: len(names)]
    fig = go.Figure(
        go.Bar(
            x=names, y=vals,
            marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(size=12, color=_T["text"], family="Inter"),
            hovertemplate="<b>%{x}</b>  %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            height=290,
            xaxis=dict(showgrid=False, tickfont=dict(size=11, color=_T["text"])),
            yaxis=dict(
                range=[0, 100], showgrid=True, gridcolor=_T["grid"],
                zeroline=False, ticksuffix="%", tickfont=dict(size=10, color=_T["dim"]),
            ),
        )
    )
    return fig


def create_confusion_matrix_illustration(classes=None):
    if _eval_data:
        pct = np.array(_eval_data['confusion_matrix_percent'], dtype=float)
        classes = _eval_data['classes']
    else:
        m = np.array([[68.5, 15.75, 15.75], [15.75, 68.5, 15.75], [15.75, 15.75, 68.5]], dtype=float)
        pct = (m / m.sum(axis=1, keepdims=True) * 100).round(1)
        classes = ["Low", "Medium", "High"]
    fig = go.Figure(
        go.Heatmap(
            z=pct, x=classes, y=classes,
            colorscale=[
                [0, "#ffffff"], [0.30, "#f5eef1"],
                [0.55, "#a61a3c"], [1.0, "#800020"],
            ],
            text=[[f"{v:.1f}%" for v in row] for row in pct],
            texttemplate="%{text}",
            textfont=dict(size=13, family="Inter"),
            hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>%{z:.1f}%<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        **_base_layout(
            height=330,
            xaxis=dict(title="Predicted", side="bottom", tickfont=dict(size=11, color=_T["text"])),
            yaxis=dict(title="Actual", autorange="reversed", tickfont=dict(size=11, color=_T["text"])),
        )
    )
    return fig


def create_class_distribution_chart(classes=None):
    if _eval_data:
        dist = _eval_data['class_distribution']
    else:
        dist = {"Low": 33.3, "Medium": 33.4, "High": 33.3}
    names = list(dist.keys())
    vals = list(dist.values())
    fig = go.Figure(
        go.Bar(
            x=names, y=vals,
            marker=dict(
                color=[_CLASS_COLORS.get(c, _T["accent"]) for c in names],
                cornerradius=6, line=dict(width=0),
            ),
            text=[f"{v}%" for v in vals],
            textposition="outside",
            textfont=dict(size=12, color=_T["text"], family="Inter"),
            hovertemplate="<b>%{x}</b>  ~%{y}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            height=270,
            xaxis=dict(showgrid=False, tickfont=dict(size=11, color=_T["text"])),
            yaxis=dict(
                range=[0, 60], showgrid=True, gridcolor=_T["grid"],
                zeroline=False, ticksuffix="%", tickfont=dict(size=10, color=_T["dim"]),
            ),
        )
    )
    return fig
