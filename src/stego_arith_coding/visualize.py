"""
Simple SVG visualization for arithmetic coding steps.

Reads coding_data.json produced by encode/decode and renders an SVG
that shows the current interval per step, cumulative boundaries, and
the selected sub-interval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _scale(val: int, max_val: int, x0: float, x1: float) -> float:
    return x0 + (float(val) / float(max_val)) * (x1 - x0)


def render_svg_from_json(json_path: str, svg_path: str) -> None:
    """Render a simple SVG visualization from coding_data.json.

    Args:
        json_path: Path to coding_data.json
        svg_path: Path to write visualization.svg
    """
    data: Dict[str, Any]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    precision: int = int(data.get("precision", 16))
    steps: List[Dict[str, Any]] = list(data.get("steps", []))
    mode: str = str(data.get("mode", "encode"))

    max_val = 2**precision
    width = 1000
    height_per_step = 80
    margin_top = 30
    margin_bottom = 30
    margin_left = 80
    margin_right = 40
    total_height = (
        margin_top + margin_bottom + max(height_per_step * max(1, len(steps)), 60)
    )

    x0 = margin_left
    x1 = width - margin_right

    svg_lines: List[str] = []
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}">'
    )
    svg_lines.append(
        f'<rect x="0" y="0" width="{width}" height="{total_height}" fill="white" />'
    )
    svg_lines.append(
        f'<text x="{width/2:.1f}" y="20" font-family="monospace" font-size="14" text-anchor="middle">{mode.upper()} visualization ({len(steps)} steps, precision={precision})</text>'
    )

    if not steps:
        svg_lines.append(
            '<text x="20" y="50" font-family="monospace" font-size="14">No steps to visualize.</text>'
        )
        svg_lines.append("</svg>")
        Path(svg_path).write_text("\n".join(svg_lines), encoding="utf-8")
        return

    for idx, step in enumerate(steps, start=1):
        y = margin_top + (idx - 1) * height_per_step + 10
        label_y = y - 5

        # Baseline representing [0, 2^precision)
        svg_lines.append(
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#333" stroke-width="1" />'
        )
        svg_lines.append(
            f'<text x="{x0-10}" y="{label_y}" font-family="monospace" font-size="12" text-anchor="end">step {idx}</text>'
        )

        cur_before = step.get("cur_interval_before", [0, max_val])
        new_range = step.get("new_range", cur_before)
        cum_probs = step.get("cum_probs", [])
        tokens = step.get("tokens", [])
        selection = step.get("selection", 0)

        # Draw current interval (thicker black) with tooltip and endpoint labels
        cb0, cb1 = int(cur_before[0]), int(cur_before[1])
        interval_width = max(cb1 - cb0, 0)
        x_cb0 = _scale(cb0, max_val, x0, x1)
        x_cb1 = _scale(cb1, max_val, x0, x1)
        svg_lines.append(
            f'<g><line x1="{x_cb0:.2f}" y1="{y}" '
            f'x2="{x_cb1:.2f}" y2="{y}" stroke="#000" stroke-width="3" />'
            f"<title>current interval [{cb0}, {cb1}) width={interval_width} (~{(interval_width/max_val if max_val else 0):.6f} of total)</title></g>"
        )
        # Left/right numeric labels for the current interval
        svg_lines.append(
            f'<text x="{x_cb0:.2f}" y="{y-12}" font-family="monospace" font-size="11" text-anchor="start" fill="#000">{cb0}</text>'
        )
        svg_lines.append(
            f'<text x="{x_cb1:.2f}" y="{y-12}" font-family="monospace" font-size="11" text-anchor="end" fill="#000">{cb1}</text>'
        )

        # Draw message_idx for encoding mode as a blue dot on the current interval line
        if mode == "encode":
            msg_idx = step.get("message_idx")
            if msg_idx is not None:
                try:
                    x_msg = _scale(int(msg_idx), max_val, x0, x1)
                    svg_lines.append(
                        f'<circle cx="{x_msg:.2f}" cy="{y:.2f}" r="5" fill="#1e90ff"><title>message_idx={int(msg_idx)}</title></circle>'
                    )
                except Exception:
                    # Ignore any casting/scaling issues for robustness
                    pass

        # If previous step emitted bits, draw rescale connectors from previous selected range to this interval
        if idx > 1:
            prev_step = steps[idx - 2]
            prev_emitted = int(prev_step.get("num_bits_emitted", 0))
            if prev_emitted > 0:
                prev_nr = prev_step.get(
                    "new_range", prev_step.get("cur_interval_after", [0, max_val])
                )
                p0, p1 = int(prev_nr[0]), int(prev_nr[1])
                y_prev = margin_top + (idx - 2) * height_per_step + 10
                x_p0 = _scale(p0, max_val, x0, x1)
                x_p1 = _scale(p1, max_val, x0, x1)
                # Dashed connectors from previous selected sub-interval to current rescaled interval
                svg_lines.append(
                    f'<g stroke="#1e90ff" stroke-width="1" stroke-dasharray="4 3" fill="none">'
                    f'<line x1="{x_p0:.2f}" y1="{y_prev+8:.2f}" x2="{x_cb0:.2f}" y2="{y-8:.2f}" />'
                    f'<line x1="{x_p1:.2f}" y1="{y_prev+8:.2f}" x2="{x_cb1:.2f}" y2="{y-8:.2f}" />'
                    f"</g>"
                )
                # Endpoint markers and tooltip
                svg_lines.append(
                    f'<circle cx="{x_cb0:.2f}" cy="{y-8:.2f}" r="2" fill="#1e90ff"><title>rescaled: [{p0}, {p1}) → [{cb0}, {cb1})</title></circle>'
                )
                svg_lines.append(
                    f'<circle cx="{x_cb1:.2f}" cy="{y-8:.2f}" r="2" fill="#1e90ff"></circle>'
                )

        # Draw translucent bands for each token sub-interval with tooltips
        prev = cb0
        for j, cp in enumerate(cum_probs):
            b = int(prev)
            t = int(cp)
            x_b = _scale(b, max_val, x0, x1)
            x_t = _scale(t, max_val, x0, x1)
            w = max(x_t - x_b, 0.5)
            token = tokens[j] if isinstance(tokens, list) and j < len(tokens) else ""
            prob = (t - b) / interval_width if interval_width > 0 else 0.0
            sel_mark = " (selected)" if int(selection) == j else ""
            tooltip = f"token[{j}]: {token}{sel_mark}\nrange=[{b}, {t}) width={t-b}\nprob≈{prob:.6f}"
            svg_lines.append(
                f'<rect x="{x_b:.2f}" y="{y-10}" width="{w:.2f}" height="20" fill="#4a90e2" fill-opacity="0.08" stroke="none"><title>{_esc(tooltip)}</title></rect>'
            )
            prev = cp

        # Draw cumulative boundaries as small ticks with tooltips
        for j, cp in enumerate(cum_probs):
            x = _scale(int(cp), max_val, x0, x1)
            svg_lines.append(
                f'<g><line x1="{x:.2f}" y1="{y-6}" x2="{x:.2f}" y2="{y+6}" stroke="#888" stroke-width="1" /><title>boundary after token[{j}] at {int(cp)}</title></g>'
            )

        # Draw selected sub-interval in red with detailed tooltip
        nr0, nr1 = int(new_range[0]), int(new_range[1])
        selected_token = (
            tokens[int(selection)]
            if isinstance(tokens, list) and 0 <= int(selection) < len(tokens)
            else ""
        )
        emitted_bits = step.get("emitted_bits", [])
        num_bits = step.get("num_bits_emitted", 0)
        message_idx = step.get("message_idx")
        extra = f"\nmessage_idx={message_idx}" if message_idx is not None else ""
        tooltip_sel = f"selected token[{int(selection)}]: {selected_token}\nnew_range=[{nr0}, {nr1}) width={nr1-nr0}\nemitted_bits={emitted_bits} (n={num_bits}){extra}"
        svg_lines.append(
            f'<g><line x1="{_scale(nr0, max_val, x0, x1):.2f}" y1="{y}" '
            f'x2="{_scale(nr1, max_val, x0, x1):.2f}" y2="{y}" stroke="#c00" stroke-width="5" />'
            f"<title>{_esc(tooltip_sel)}</title></g>"
        )

        # Token label
        token_label = ""
        if isinstance(tokens, list) and 0 <= int(selection) < len(tokens):
            token = tokens[int(selection)]
            token_label = _esc(str(token))
        bits_str = (
            "".join(str(int(b)) for b in emitted_bits)
            if isinstance(emitted_bits, list)
            else ""
        )
        svg_lines.append(
            f'<text x="{_scale(nr0, max_val, x0, x1):.2f}" y="{y+18}" font-family="monospace" font-size="12" fill="#c00">sel={selection} {token_label}</text>'
        )
        svg_lines.append(
            f'<text x="{x0}" y="{y+34}" font-family="monospace" font-size="12" fill="#333">bits={num_bits} {bits_str}</text>'
        )

    svg_lines.append("</svg>")
    Path(svg_path).write_text("\n".join(svg_lines), encoding="utf-8")
