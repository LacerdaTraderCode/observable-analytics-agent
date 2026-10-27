"""Plain-text tables for command line output."""

from typing import Any


def render_table(title: str, header: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return f"{title}\n  (no data)"
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(name), *(len(row[index]) for row in cells)) for index, name in enumerate(header)
    ]
    lines = ["  ".join(name.ljust(width) for name, width in zip(header, widths, strict=True))]
    lines += [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in cells
    ]
    return "\n".join([title, *lines])
