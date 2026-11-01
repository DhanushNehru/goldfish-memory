#!/usr/bin/env python3
"""Render assets/demo.gif: a scripted terminal session using REAL CLI output.

The session lines below are produced by actually running `goldfish` (see
BENCHMARKS.md for the environment); this script only animates them so the
README demo stays reproducible. No network is needed or used.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 880, 500
BG = (30, 32, 40)
FG = (220, 223, 228)
PROMPT = (125, 211, 168)
DIM = (140, 145, 155)
FPS = 12

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# (command, real captured output lines)
SESSION = [
    ("goldfish init", ["Initialized goldfish-memory store at .goldfish/memory.db"]),
    (
        'goldfish add --kind decision "Chose SQLite + sqlite-vec: one file for all memory"',
        ["saved memory #1 (decision)"],
    ),
    (
        'goldfish add --kind error "Login 500s when session cookie missing; fixed in auth.py"',
        ["saved memory #2 (error)"],
    ),
    (
        'goldfish add --kind note "API rate limit is per API key, not per IP"',
        ["saved memory #3 (note)"],
    ),
    (
        'goldfish search "why sqlite for storage"',
        [
            "[+0.677] #1 (decision) Chose SQLite + sqlite-vec: one file for all memory",
            "[-0.049] #2 (error) Login 500s when session cookie is missing; fixed in a...",
            "[-0.061] #3 (note) API rate limit is per API key, not per IP",
        ],
    ),
    (
        'goldfish inject "add rate-limit headers to the login endpoint"',
        [
            "## Relevant memory (goldfish-memory)",
            "- [note] API rate limit is per API key, not per IP",
            "- [error] Login 500s when session cookie is missing; fixed in auth.py",
            "- [decision] Chose SQLite + sqlite-vec: one file for all memory",
        ],
    ),
]


def render(out: Path) -> None:
    font = ImageFont.truetype(FONT, 17)
    font_b = ImageFont.truetype(FONT_B, 17)
    line_h = 26
    frames: list[Image.Image] = []
    history: list[tuple[str, tuple[int, int, int], ImageFont.FreeTypeFont]] = []

    def frame(lines: list[tuple[str, tuple[int, int, int], ImageFont.FreeTypeFont]], repeat: int = 1) -> None:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, W - 1, 34), 8, fill=(43, 46, 56))
        for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse((16 + i * 26, 11, 30 + i * 26, 25), fill=c)
        d.text((W // 2 - 70, 9), "goldfish-memory", font=font, fill=DIM)
        y = 48
        for text, color, f in lines:
            if y > H - 20:
                break
            d.text((18, y), text, font=f, fill=color)
            y += line_h
        for _ in range(repeat):
            frames.append(img)

    for cmd, output in SESSION:
        typed = "$ "
        text = "$ " + cmd
        for i in range(len("$ "), len(text) + 1, 2):
            typed = text[:i]
            frame(history + [(typed + "▌", PROMPT, font_b)])
        frame(history + [(text, PROMPT, font_b)], repeat=3)
        history.append(("$ " + cmd, PROMPT, font_b))
        for line in output:
            history.append((line, FG, font))
            frame(history, repeat=2)
        frame(history, repeat=6)  # 0.5s dwell after each command
    frame(history, repeat=FPS * 2)  # 2s hold at the end

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"wrote {out} ({len(frames)} frames, ~{len(frames) / FPS:.0f}s)")


if __name__ == "__main__":
    render(Path(__file__).parent.parent / "assets" / "demo.gif")
