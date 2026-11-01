#!/usr/bin/env python3
"""Render assets/architecture.png: the goldfish-memory architecture diagram."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 560
BG = (255, 255, 255)
INK = (33, 37, 41)
ACCENT = (30, 120, 80)
BOX = (240, 246, 242)
EDGE = (120, 130, 125)
WARN = (176, 42, 42)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def box(d, xy, title, lines, fill=BOX, outline=ACCENT, title_color=ACCENT):
    x0, y0 = xy[0], xy[1]
    d.rounded_rectangle(xy, 10, fill=fill, outline=outline, width=2)
    d.text((x0 + 14, y0 + 10), title, font=ImageFont.truetype(SANS_B, 17), fill=title_color)
    f = ImageFont.truetype(SANS, 14)
    for i, line in enumerate(lines):
        d.text((x0 + 14, y0 + 38 + i * 20), line, font=f, fill=INK)


def arrow(d, p0, p1, label="", color=EDGE):
    d.line([p0, p1], fill=color, width=2)
    x, y = p1
    d.polygon([(x, y), (x - 7, y - 6), (x - 7, y + 6)], fill=color) if p1[0] > p0[0] else d.polygon(
        [(x, y), (x + 7, y - 6), (x + 7, y + 6)], fill=color
    )
    if label:
        mx, my = (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2
        d.text((mx - 40, my - 18), label, font=ImageFont.truetype(SANS, 12), fill=color)


def render(out: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((30, 20), "goldfish-memory architecture", font=ImageFont.truetype(SANS_B, 24), fill=INK)

    box(
        d,
        (30, 80, 300, 210),
        "your coding agent",
        [
            "any agent that can run a CLI",
            "or import a Python package",
            "",
            "goldfish add / search / inject",
        ],
    )
    box(
        d,
        (360, 80, 640, 210),
        "MemoryEngine (Python)",
        [
            "capture()   search()",
            "inject()    compact()",
            "pin()       forget()",
        ],
    )
    box(
        d,
        (700, 50, 970, 135),
        "Embedder (swappable)",
        [
            "all-MiniLM-L6-v2 (bundled)",
            "quantized ONNX, CPU",
        ],
    )
    box(
        d,
        (700, 155, 970, 240),
        "compaction",
        [
            "clusters old memories, keeps",
            "extractive summaries",
        ],
    )
    box(
        d,
        (200, 300, 560, 420),
        "one SQLite file",
        [
            ".goldfish/memory.db",
            "memories: text, kind, metadata,",
            "memory_vec: sqlite-vec index",
        ],
    )
    box(
        d,
        (620, 300, 970, 420),
        "offline guarantees",
        [
            "no networking code;",
            "CI: suite + demo pass",
            "with no network route",
        ],
        fill=(252, 244, 244),
        outline=WARN,
        title_color=WARN,
    )

    arrow(d, (300, 145), (360, 145), "CLI / API")
    arrow(d, (640, 115), (700, 92))
    arrow(d, (640, 175), (700, 197))
    arrow(d, (470, 210), (430, 300), "embed + store")
    arrow(d, (620, 360), (560, 360))
    d.text(
        (30, 470),
        "memory in -> embedded -> indexed -> retrieved -> injected into the next session's prompt",
        font=ImageFont.truetype(SANS, 15),
        fill=EDGE,
    )
    img.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    render(Path(__file__).parent.parent / "assets" / "architecture.png")
