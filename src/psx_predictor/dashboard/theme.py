"""Design tokens, page styling and the shared Plotly look for the dashboard.

The colours match `.streamlit/config.toml`. Streamlit switches between its light and
dark theme (menu > Settings > Theme); `current_mode()` reads which one is active so
charts and the small HTML pieces (cards, chips) follow it.
"""

from dataclasses import dataclass
from html import escape
from typing import Literal

import plotly.graph_objects as go
import streamlit as st

Mode = Literal["dark", "light"]


@dataclass(frozen=True)
class Palette:
    """Colour tokens for one theme mode."""

    bg: str
    surface: str
    surface2: str
    border: str
    text: str
    muted: str
    accent: str
    accent_soft: str
    up: str
    down: str
    warn: str
    up_soft: str
    down_soft: str
    hold_soft: str
    warn_soft: str
    sma20: str
    sma50: str


DARK = Palette(
    bg="#0b0f17",
    surface="#121826",
    surface2="#1a2234",
    border="#253048",
    text="#e6ebf5",
    muted="#97a3b9",
    accent="#3b82f6",
    accent_soft="rgba(59,130,246,0.14)",
    up="#34c77b",
    down="#f26b6b",
    warn="#f2b84b",
    up_soft="rgba(52,199,123,0.16)",
    down_soft="rgba(242,107,107,0.16)",
    hold_soft="rgba(151,163,185,0.16)",
    warn_soft="rgba(242,184,75,0.16)",
    sma20="#f2b84b",
    sma50="#a78bfa",
)

LIGHT = Palette(
    bg="#f4f6fa",
    surface="#ffffff",
    surface2="#eef1f6",
    border="#dde3ec",
    text="#0f172a",
    muted="#556072",
    accent="#2563eb",
    accent_soft="rgba(37,99,235,0.10)",
    up="#0f7a43",
    down="#c0322f",
    warn="#9a6200",
    up_soft="#e3f5ea",
    down_soft="#fbe7e6",
    hold_soft="#eceff4",
    warn_soft="#fbf0da",
    sma20="#b7791f",
    sma50="#7c3aed",
)

FONT_FAMILY = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
MONO_FAMILY = "'JetBrains Mono', ui-monospace, Consolas, monospace"


def current_mode() -> Mode:
    """The active Streamlit theme ('dark' or 'light'); dark when it can't be read."""
    try:
        theme = st.context.theme
        mode = getattr(theme, "type", None) or (
            theme.get("type") if isinstance(theme, dict) else None
        )
    except Exception:
        mode = None
    return "light" if mode == "light" else "dark"


def palette(mode: Mode | None = None) -> Palette:
    """Colour tokens for `mode` (default: the active theme)."""
    return LIGHT if (mode or current_mode()) == "light" else DARK


def signal_colors(signal: str, pal: Palette | None = None) -> tuple[str, str]:
    """(background, foreground) colours for a BUY / HOLD / SELL chip."""
    p = pal or palette()
    sig = (signal or "").upper()
    if sig == "BUY":
        return p.up_soft, p.up
    if sig == "SELL":
        return p.down_soft, p.down
    return p.hold_soft, p.muted


def style_figure(fig: go.Figure, height: int | None = None, mode: Mode | None = None) -> go.Figure:
    """Apply the dashboard look to any Plotly figure (fonts, grid, transparent background)."""
    p = palette(mode)
    fig.update_layout(
        template="plotly_dark" if (mode or current_mode()) == "dark" else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": p.text, "size": 12},
        title=None,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": p.surface,
            "bordercolor": p.border,
            "font": {"family": FONT_FAMILY, "color": p.text},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": p.muted},
        },
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=p.border,
        tickfont={"color": p.muted},
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=p.border,
        griddash="dot",
        linecolor=p.border,
        tickfont={"color": p.muted, "family": MONO_FAMILY},
        title_font={"color": p.muted},
        zeroline=False,
    )
    if height:
        fig.update_layout(height=height)
    return fig


def inject_css() -> None:
    """Page-level CSS: tighter layout and the card/chip classes used by components.ui."""
    p = palette()
    st.html(
        f"""
<style>
.block-container {{ padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stMetricValue"] {{ font-family: {MONO_FAMILY}; }}
.psx-card {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 14px;
  padding: 16px; display: flex; flex-direction: column; gap: 4px; height: 100%;
  box-sizing: border-box; }}
.psx-label {{ color: {p.muted}; font-size: 12px; font-weight: 500; text-transform: uppercase;
  letter-spacing: 0.04em; }}
.psx-value {{ font-family: {MONO_FAMILY}; font-size: 26px; font-weight: 600; color: {p.text};
  font-variant-numeric: tabular-nums; line-height: 1.2; }}
.psx-note {{ font-size: 12px; font-family: {MONO_FAMILY}; }}
.psx-muted {{ color: {p.muted}; }}
.psx-up {{ color: {p.up}; }}
.psx-down {{ color: {p.down}; }}
.psx-warn {{ color: {p.warn}; }}
.psx-chip {{ display: inline-block; min-width: 48px; text-align: center; padding: 4px 10px;
  border-radius: 6px; font-size: 12px; font-weight: 700; }}
.psx-pill {{ display: inline-flex; align-items: center; gap: 8px; height: 30px; padding: 0 12px;
  border-radius: 999px; background: {p.surface2}; color: {p.muted}; font-size: 12px; }}
.psx-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.psx-row {{ display: flex; align-items: center; gap: 10px; padding: 6px 0; }}
.psx-row + .psx-row {{ border-top: 1px solid {p.border}; }}
.psx-step {{ width: 22px; height: 22px; border-radius: 50%; display: inline-grid;
  place-items: center; font-size: 12px; font-weight: 700; flex: none; }}
.psx-sub {{ color: {p.muted}; margin-top: -0.6rem; margin-bottom: 0.6rem; }}
.psx-heat {{ border-radius: 10px; padding: 10px 12px; }}
.psx-heat b {{ font-family: {MONO_FAMILY}; font-size: 16px; display: block; }}
.psx-disclaimer {{ color: {p.muted}; font-size: 12px; margin-top: 1.5rem; }}
</style>
"""
    )


def text(value: object) -> str:
    """HTML-escape any value for the small HTML snippets."""
    return escape(str(value))
