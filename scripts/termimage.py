"""Render images directly in the terminal.

Two paths, chosen automatically:

* iTerm2 / Warp support a real inline-image protocol — actual pixels.
* Every other modern terminal (Terminal.app, kitty, VS Code) supports 24-bit
  colour, so images render as Unicode half-blocks: each character cell shows two
  vertical pixels (top as foreground, bottom as background), which doubles the
  vertical resolution and looks like a real, if chunky, image.

No external dependencies beyond Pillow and numpy, which the project already uses.
"""

from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np
from PIL import Image

UPPER_HALF = "▀"  # ▀
RESET = "\x1b[0m"


def _to_pil(array: np.ndarray) -> Image.Image:
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.dstack([array] * 3)
    if array.dtype != np.uint8:
        peak = float(array.max()) if array.size else 0.0
        array = array.astype(np.float32)
        array = array / peak if peak > 1.0 else array
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array[..., :3], "RGB")


def _supports_inline() -> bool:
    if os.environ.get("TERMIMAGE_FORCE") == "blocks":
        return False
    if os.environ.get("TERMIMAGE_FORCE") == "inline":
        return True
    program = os.environ.get("TERM_PROGRAM", "")
    return program in ("iTerm.app", "WarpTerminal") or bool(os.environ.get("ITERM_SESSION_ID"))


def _render_inline(image: Image.Image, cols: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f"\x1b]1337;File=inline=1;width={cols};preserveAspectRatio=1:"
        f"{payload}\x07\n"
    )


def _render_blocks(image: Image.Image, cols: int) -> str:
    # Two vertical pixels per character row, so the pixel grid is cols x (rows*2).
    aspect = image.height / image.width
    rows = max(1, int(cols * aspect * 0.5))
    resized = image.resize((cols, rows * 2), Image.BILINEAR)
    px = np.asarray(resized)

    out = []
    for r in range(rows):
        top = px[2 * r]
        bottom = px[2 * r + 1]
        line = []
        for c in range(cols):
            tr, tg, tb = top[c]
            br, bg, bb = bottom[c]
            line.append(f"\x1b[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m{UPPER_HALF}")
        out.append("".join(line) + RESET)
    return "\n".join(out) + "\n"


def show(array: np.ndarray, cols: int = 60, caption: str | None = None) -> None:
    """Print an image to the terminal. ``cols`` is the width in characters."""
    image = _to_pil(array)
    if caption:
        print(caption)
    if _supports_inline():
        sys.stdout.write(_render_inline(image, cols))
    else:
        sys.stdout.write(_render_blocks(image, cols))
    sys.stdout.flush()


def show_side_by_side(left: np.ndarray, right: np.ndarray,
                      labels: tuple[str, str] = ("", ""), cols: int = 42) -> None:
    """Print two images next to each other (block mode only; inline stacks)."""
    if _supports_inline():
        show(left, cols=cols, caption=labels[0])
        show(right, cols=cols, caption=labels[1])
        return

    li = _render_blocks(_to_pil(left), cols).rstrip("\n").split("\n")
    ri = _render_blocks(_to_pil(right), cols).rstrip("\n").split("\n")
    rows = max(len(li), len(ri))
    li += [" " * cols] * (rows - len(li))
    ri += [" " * cols] * (rows - len(ri))
    if any(labels):
        print(f"{labels[0]:<{cols}}   {labels[1]}")
    for left_row, right_row in zip(li, ri):
        print(f"{left_row}{RESET}   {right_row}{RESET}")
