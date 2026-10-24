"""
tests/create_sample.py – generate tests/sample_notes.png using Pillow.
Run once: python tests/create_sample.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 800, 560
BG = (255, 255, 255)
FG = (10, 10, 80)

LINES = [
    "Physics Notes — Chapter 4: Forces & Motion",
    "",
    "Newton's Second Law:  F = m × a",
    "  F = force (Newtons, N)",
    "  m = mass (kilograms, kg)",
    "  a = acceleration (m/s²)",
    "",
    "Weight:  W = m × g   (g ≈ 9.8 m/s²)",
    "",
    "Momentum:  p = m × v",
    "  p = momentum (kg·m/s)",
    "  v = velocity (m/s)",
    "",
    "Kinetic Energy:  KE = ½ m v²  (Joules, J)",
    "",
    "Work done:  W = F × d × cos(θ)  (J)",
]


def make_sample_image(output_path: Path) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG)
    draw = ImageDraw.Draw(img)

    # Try to load a monospace font; fall back to Pillow default
    font_large = None
    font_normal = None
    try:
        font_large = ImageFont.truetype("cour.ttf", 22)   # Courier New on Windows
        font_normal = ImageFont.truetype("cour.ttf", 18)
    except OSError:
        try:
            font_large = ImageFont.truetype("DejaVuSansMono.ttf", 22)
            font_normal = ImageFont.truetype("DejaVuSansMono.ttf", 18)
        except OSError:
            font_large = ImageFont.load_default()
            font_normal = font_large

    x, y = 40, 30
    line_height = 30

    for i, line in enumerate(LINES):
        font = font_large if i == 0 else font_normal
        color = (60, 0, 120) if i == 0 else FG
        draw.text((x, y), line, font=font, fill=color)
        y += line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    print(f"Saved sample image → {output_path}")


if __name__ == "__main__":
    make_sample_image(Path(__file__).parent / "sample_notes.png")
