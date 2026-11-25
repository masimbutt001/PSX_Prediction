"""Small styled building blocks (KPI cards, signal chips, status rows) for the dashboard."""

from typing import Optional, Sequence

import streamlit as st

from psx_predictor.dashboard.components.help import helped
from psx_predictor.dashboard.theme import palette, signal_colors, text

Tone = str  # "up" | "down" | "warn" | "muted" | "accent"


def fmt_pct(value: Optional[float], digits: int = 2, arrow: bool = True) -> str:
    """'▲ +1.42%' / '▼ -0.74%' / '—'. Arrows keep up and down readable without colour."""
    if value is None or value != value:  # None or NaN
        return "—"
    sign = "+" if value > 0 else ""
    mark = ("▲ " if value > 0 else "▼ " if value < 0 else "") if arrow else ""
    return f"{mark}{sign}{value:.{digits}f}%"


def tone_of(value: Optional[float]) -> Tone:
    """'up', 'down' or 'muted' depending on the sign of `value`."""
    if value is None or value != value:
        return "muted"
    return "up" if value > 0 else "down" if value < 0 else "muted"


def page_header(title: str, subtitle: str = "") -> None:
    """Page title with a muted one-line subtitle."""
    st.markdown(f"## {title}")
    if subtitle:
        st.html(f'<div class="psx-sub">{text(subtitle)}</div>')


def kpi_card(label: str, value: str, note: str = "", tone: Tone = "muted") -> str:
    """HTML for one KPI card; render a row of them with `kpi_row`."""
    return (
        f'<div class="psx-card"><div class="psx-label">{text(label)}</div>'
        f'<div class="psx-value">{text(value)}</div>'
        f'<div class="psx-note psx-{tone}">{text(note)}</div></div>'
    )


def kpi_row(
    cards: Sequence[tuple[str, str, str, Tone]],
    helps: Optional[Sequence[Optional[str]]] = None,
) -> None:
    """A responsive row of KPI cards: (label, value, note, tone) each.

    `helps` gives the explanation key for each card (see dashboard/help_text.py); a card
    with a key gets a (?) button in its corner.
    """
    keys = list(helps) if helps is not None else [None] * len(cards)
    cols = st.columns(len(cards))
    for col, (label, value, note, tone), key in zip(cols, cards, keys, strict=True):
        with col:
            if key:
                with helped(key):
                    st.html(kpi_card(label, value, note, tone))
            else:
                st.html(kpi_card(label, value, note, tone))


def signal_chip(signal: Optional[str]) -> str:
    """HTML chip for BUY / HOLD / SELL."""
    sig = (signal or "N/A").upper()
    bg, fg = signal_colors(sig)
    return f'<span class="psx-chip" style="background:{bg};color:{fg}">{text(sig)}</span>'


def status_pill(label: str, ok: bool = True, warn: bool = False) -> str:
    """Rounded pill with a coloured status dot, e.g. data freshness."""
    p = palette()
    dot = p.warn if warn else (p.up if ok else p.down)
    return (
        f'<span class="psx-pill"><span class="psx-dot" style="background:{dot}"></span>'
        f"{text(label)}</span>"
    )


def step_row(name: str, status: str, note: str = "") -> str:
    """One line of the pipeline health list: icon, step name, note."""
    p = palette()
    status = status.upper()
    if status == "SUCCESS":
        icon, bg, fg = "✓", p.up_soft, p.up
    elif status == "SKIPPED":
        icon, bg, fg = "–", p.hold_soft, p.muted
    else:
        icon, bg, fg = "!", p.down_soft, p.down
    return (
        f'<div class="psx-row"><span class="psx-step" style="background:{bg};color:{fg}" '
        f'aria-hidden="true">{icon}</span><span style="flex:1">{text(name)}</span>'
        f'<span style="font-size:12px;color:{fg}">{text(note)}</span></div>'
    )


def sentiment_tag(label: Optional[str]) -> str:
    """POS / NEG / NEU tag for a news headline."""
    p = palette()
    lab = (label or "").upper()
    if lab.startswith("POS"):
        bg, fg, short = p.up_soft, p.up, "POS"
    elif lab.startswith("NEG"):
        bg, fg, short = p.down_soft, p.down, "NEG"
    else:
        bg, fg, short = p.hold_soft, p.muted, "NEU"
    return (
        f'<span class="psx-chip" style="min-width:36px;font-size:11px;background:{bg};'
        f'color:{fg}">{short}</span>'
    )


def card(html_body: str) -> None:
    """Render arbitrary HTML inside a card container."""
    st.html(f'<div class="psx-card">{html_body}</div>')


def disclaimer() -> None:
    """Footer line shown on every page."""
    st.html(
        '<div class="psx-disclaimer">Research and decision support only. '
        "Not financial advice.</div>"
    )
