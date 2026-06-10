"""Render ``docs/architecture.png`` deterministically.

Uses Pillow (already a transitive dep via Gradio) so no new runtime
package is needed. Lays out labeled boxes for each architectural
layer and draws arrows showing data flow.

Regenerate after any architecture change with:

    poetry run python scripts/render_architecture.py

The PNG is committed alongside this script so reviewers see the
artifact directly on GitHub. The Mermaid text source is in
``docs/architecture.mmd`` for those who prefer to render with
``mmdc``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "architecture.png"

W, H = 1600, 1100
BG = (255, 255, 255)
GRID = (240, 240, 240)


@dataclass(frozen=True)
class Box:
    label: str
    sub: tuple[str, ...]
    x: int
    y: int
    w: int
    h: int
    fill: tuple[int, int, int]
    stroke: tuple[int, int, int]


BOXES: list[Box] = [
    Box(
        "Patient / caller",
        ("User-facing entry",),
        x=60,
        y=40,
        w=300,
        h=70,
        fill=(243, 244, 246),
        stroke=(75, 85, 99),
    ),
    Box(
        "FastAPI service",
        ("POST /chat", "POST /evaluate", "GET /health"),
        x=60,
        y=170,
        w=300,
        h=120,
        fill=(220, 252, 231),
        stroke=(22, 101, 52),
    ),
    Box(
        "Customer YAML",
        ("ophthalmology.yaml", "orthopedics.yaml"),
        x=460,
        y=170,
        w=300,
        h=120,
        fill=(254, 249, 195),
        stroke=(146, 64, 14),
    ),
    Box(
        "Agent  (ReAct loop)",
        (
            "LLM  (gpt-4o-mini / FakeLLM)",
            "Tools registry",
            "search_slots / book / cancel /",
            "check_eligibility / create_pms_task",
        ),
        x=860,
        y=170,
        w=460,
        h=160,
        fill=(224, 231, 255),
        stroke=(49, 46, 129),
    ),
    Box(
        "RAG  (FAISS + TF-IDF)",
        ("rules.faiss", "rules_meta.json"),
        x=860,
        y=380,
        w=220,
        h=110,
        fill=(224, 242, 254),
        stroke=(12, 74, 110),
    ),
    Box(
        "Structured store",
        ("structured.sqlite", "providers / slots /", "appointments / eligibility"),
        x=1100,
        y=380,
        w=220,
        h=110,
        fill=(224, 242, 254),
        stroke=(12, 74, 110),
    ),
    Box(
        "Sentinel trust engine",
        (
            "Guardrails  (emergency, clinical, PHI)",
            "LLM-as-Judge  (booking + hallucination)",
            "Escalation scorer  (5 signals -> 0-1)",
        ),
        x=460,
        y=420,
        w=380,
        h=160,
        fill=(254, 226, 226),
        stroke=(127, 29, 29),
    ),
    Box(
        "Observability",
        ("traces.jsonl  spans + tokens + cost", "audit.jsonl  PHI-redacted"),
        x=60,
        y=420,
        w=380,
        h=110,
        fill=(243, 232, 255),
        stroke=(88, 28, 135),
    ),
    Box(
        "Simulation harness",
        ("100 personas / customer", "scripted FakeLLM (CI) +", "live OpenAI (staging)"),
        x=60,
        y=600,
        w=380,
        h=130,
        fill=(255, 237, 213),
        stroke=(154, 52, 18),
    ),
    Box(
        "Evaluation harness",
        ("python -m clarion.eval --customer X", "runner / metrics / reporter"),
        x=480,
        y=600,
        w=380,
        h=130,
        fill=(255, 237, 213),
        stroke=(154, 52, 18),
    ),
    Box(
        "report_<customer>.json",
        ("11 metrics", "schema_version 1.0.0  (locked)"),
        x=900,
        y=600,
        w=320,
        h=110,
        fill=(220, 252, 231),
        stroke=(22, 101, 52),
    ),
    Box(
        "trace_<customer>.json",
        ("per-scenario rows for Trace Explorer", "schema_version 1.0.0  (locked)"),
        x=1260,
        y=600,
        w=300,
        h=110,
        fill=(220, 252, 231),
        stroke=(22, 101, 52),
    ),
    Box(
        "Gradio UI",
        (
            "Live Agent  (gr.ChatInterface)",
            "Quality Metrics  (report.json)",
            "Escalations  (report.json)",
            "Trace Explorer  (trace.json)",
            "Customer switcher",
        ),
        x=460,
        y=800,
        w=680,
        h=180,
        fill=(224, 231, 255),
        stroke=(49, 46, 129),
    ),
    Box(
        "Deployment",
        (
            "Phase 15 container image",
            "HF Spaces / Cloud Run / Render / Fly.io",
            "Same image, no code changes",
        ),
        x=1180,
        y=800,
        w=380,
        h=180,
        fill=(254, 249, 195),
        stroke=(146, 64, 14),
    ),
]


ARROWS: list[tuple[int, int, str]] = [
    (0, 1, "HTTP"),
    (1, 3, "agent.chat"),
    (2, 3, "config"),
    (3, 4, "retrieve"),
    (3, 5, "tools"),
    (3, 6, "guarded"),
    (3, 7, "spans"),
    (8, 3, "scripted runs"),
    (9, 8, "drives"),
    (9, 6, "grades"),
    (9, 7, "reads"),
    (9, 10, "writes"),
    (9, 11, "writes"),
    (10, 12, "reads"),
    (11, 12, "reads"),
    (12, 1, "Live Agent -> /chat"),
    (1, 13, "hosted by"),
    (12, 13, "hosted by"),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for cand in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ):
        if Path(cand).is_file():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    title_font = _load_font(20)
    body_font = _load_font(14)

    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.w, box.y + box.h),
        radius=10,
        fill=box.fill,
        outline=box.stroke,
        width=2,
    )

    title_bbox = draw.textbbox((0, 0), box.label, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(
        (box.x + (box.w - title_w) // 2, box.y + 10),
        box.label,
        fill=(17, 24, 39),
        font=title_font,
    )

    y = box.y + 40
    for line in box.sub:
        line_bbox = draw.textbbox((0, 0), line, font=body_font)
        lw = line_bbox[2] - line_bbox[0]
        draw.text(
            (box.x + (box.w - lw) // 2, y),
            line,
            fill=(55, 65, 81),
            font=body_font,
        )
        y += 18


def _center(box: Box) -> tuple[int, int]:
    return box.x + box.w // 2, box.y + box.h // 2


def _edge_point(box: Box, toward_x: int, toward_y: int) -> tuple[int, int]:
    cx, cy = _center(box)
    half_w = box.w / 2
    half_h = box.h / 2
    dx = toward_x - cx
    dy = toward_y - cy
    if dx == 0 and dy == 0:
        return cx, cy
    scale_x = half_w / abs(dx) if dx != 0 else float("inf")
    scale_y = half_h / abs(dy) if dy != 0 else float("inf")
    scale = min(scale_x, scale_y)
    return int(cx + dx * scale), int(cy + dy * scale)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    boxes: list[Box],
    src_i: int,
    dst_i: int,
    label: str,
) -> None:
    sx, sy = _center(boxes[src_i])
    dx, dy = _center(boxes[dst_i])

    sx, sy = _edge_point(boxes[src_i], dx, dy)
    dx, dy = _edge_point(boxes[dst_i], sx, sy)

    draw.line((sx, sy, dx, dy), fill=(75, 85, 99), width=2)

    angle = math.atan2(dy - sy, dx - sx)
    head = 8
    hx1 = dx - head * math.cos(angle - math.pi / 6)
    hy1 = dy - head * math.sin(angle - math.pi / 6)
    hx2 = dx - head * math.cos(angle + math.pi / 6)
    hy2 = dy - head * math.sin(angle + math.pi / 6)
    draw.polygon([(dx, dy), (hx1, hy1), (hx2, hy2)], fill=(75, 85, 99))

    if label:
        mid_x = (sx + dx) // 2
        mid_y = (sy + dy) // 2
        body_font = _load_font(12)
        tw_bbox = draw.textbbox((0, 0), label, font=body_font)
        tw = tw_bbox[2] - tw_bbox[0]
        th = tw_bbox[3] - tw_bbox[1]
        draw.rectangle(
            (
                mid_x - tw // 2 - 4,
                mid_y - th // 2 - 2,
                mid_x + tw // 2 + 4,
                mid_y + th // 2 + 2,
            ),
            fill=(255, 255, 255),
        )
        draw.text(
            (mid_x - tw // 2, mid_y - th // 2 - 2),
            label,
            fill=(75, 85, 99),
            font=body_font,
        )


def main() -> int:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    for x in range(0, W, 50):
        draw.line((x, 0, x, H), fill=GRID)
    for y in range(0, H, 50):
        draw.line((0, y, W, y), fill=GRID)

    title_font = _load_font(28)
    title = "Clarion -- Architecture"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 5), title, fill=(17, 24, 39), font=title_font)

    for box in BOXES:
        _draw_box(draw, box)

    for src, dst, label in ARROWS:
        _draw_arrow(draw, BOXES, src, dst, label)

    footer_font = _load_font(12)
    footer = (
        "Source: scripts/render_architecture.py   |   "
        "Mermaid: docs/architecture.mmd   |   "
        "regenerate: poetry run python scripts/render_architecture.py"
    )
    draw.text((20, H - 20), footer, fill=(107, 114, 128), font=footer_font)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH, format="PNG", optimize=True)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
