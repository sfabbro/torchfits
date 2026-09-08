"""Shared HTML table renderer for notebook ``_repr_html_`` output.

One place owns the scrollable-table styling so the HDU, header, and
HDU-list reprs stay visually consistent without duplicating CSS.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from typing import Optional

_CONTAINER_STYLE = (
    "max-height: 400px; overflow: auto; "
    "border: 1px solid rgba(128, 128, 128, 0.3); margin-bottom: 1em;"
)
_TABLE_STYLE = "border-collapse: collapse; width: 100%; margin: 0;"
_TH_COL_STYLE = (
    "text-align: {align}; padding: 8px; position: sticky; top: 0; "
    "background-color: var(--theme-ui-colors-background, white); "
    "border-bottom: 2px solid rgba(128, 128, 128, 0.3); z-index: 1;"
)
_TH_ROW_STYLE = (
    "font-weight: {weight}; text-align: {align}; padding: 8px; "
    "border-bottom: 1px solid rgba(128, 128, 128, 0.2);"
)
_TD_STYLE = (
    "text-align: {align}; padding: 8px; {extra}"
    "border-bottom: 1px solid rgba(128, 128, 128, 0.2);"
)


def render_html_table(
    aria_label: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    aligns: Optional[Sequence[str]] = None,
    cell_extra: Optional[Sequence[str]] = None,
    first_col_bold: bool = False,
) -> str:
    """Render ``rows`` as the scrollable table used by ``_repr_html_``.

    The first cell of each row becomes a row header (``<th scope="row">``).
    Every value is HTML-escaped; ``aligns`` and ``cell_extra`` are
    per-column CSS text-align values and extra td style fragments.
    """
    ncols = len(headers)
    align_list = list(aligns) if aligns is not None else ["left"] * ncols
    extra_list = list(cell_extra) if cell_extra is not None else [""] * ncols
    if len(align_list) != ncols or len(extra_list) != ncols:
        raise ValueError("aligns/cell_extra must match the header count")

    weight = "bold" if first_col_bold else "normal"
    parts = [
        f'<div tabindex="0" aria-label="{html.escape(aria_label)}"'
        f" style='{_CONTAINER_STYLE}'>",
        f"<table style='{_TABLE_STYLE}'>",
        "<thead><tr>",
    ]
    for header, align in zip(headers, align_list):
        parts.append(
            f"<th scope=\"col\" style='{_TH_COL_STYLE.format(align=align)}'>"
            f"{html.escape(header)}</th>"
        )
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col_idx, (value, align, extra) in enumerate(
            zip(row, align_list, extra_list)
        ):
            text = html.escape(str(value))
            if col_idx == 0:
                style = _TH_ROW_STYLE.format(weight=weight, align=align)
                parts.append(f"<th scope=\"row\" style='{style}'>{text}</th>")
            else:
                style = _TD_STYLE.format(align=align, extra=extra)
                parts.append(f"<td style='{style}'>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)
