#!/usr/bin/env python3
"""Generate the entrance marker: marker.png (the ARKit reference image) and marker.pdf
(A4 at 300 dpi, the image printed at exactly 18.0 cm wide).

ARKit image detection likes high contrast, lots of non-repeating detail and a broad
grey-level histogram, so this is a deterministic collage of random shapes.
Re-running produces the same image (fixed seed). If you change it, copy marker.png
into ios/Sightline/Resources/ again.
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DPI = 300
WIDTH_CM = 18.0
PX = round(WIDTH_CM / 2.54 * DPI)  # 2126 px
A4 = (round(21.0 / 2.54 * DPI), round(29.7 / 2.54 * DPI))  # 2480 x 3508


def font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def make_marker() -> Image.Image:
    rnd = random.Random(20261024)
    img = Image.new("RGB", (PX, PX), "white")
    d = ImageDraw.Draw(img)
    palette = [(0, 0, 0), (30, 30, 30), (220, 40, 50), (20, 90, 200), (250, 190, 0), (0, 150, 90), (120, 120, 120), (255, 255, 255)]
    for _ in range(260):
        c = rnd.choice(palette)
        x, y = rnd.uniform(0, PX), rnd.uniform(0, PX)
        s = rnd.uniform(30, 260)
        k = rnd.random()
        if k < 0.35:
            d.ellipse([x - s, y - s * rnd.uniform(0.4, 1), x + s, y + s], fill=c)
        elif k < 0.7:
            pts = [(x + rnd.uniform(-s, s) * 1.6, y + rnd.uniform(-s, s) * 1.6) for _ in range(rnd.randint(3, 6))]
            d.polygon(pts, fill=c)
        else:
            d.line([(x, y), (x + rnd.uniform(-2, 2) * s, y + rnd.uniform(-2, 2) * s)], fill=c, width=rnd.randint(10, 45))
    # a bold, asymmetric title bar so the marker has an obvious "up"
    d.rectangle([0, 0, PX, 300], fill="black")
    d.text((70, 150), "FINDMYTEAM", font=font(185), fill="white", anchor="lm")
    d.polygon([(PX - 200, 240), (PX - 120, 60), (PX - 40, 240)], fill=(220, 40, 50))
    d.rectangle([0, 0, PX - 1, PX - 1], outline="black", width=28)
    return img


def make_pdf(marker: Image.Image) -> Image.Image:
    page = Image.new("RGB", A4, "white")
    x0 = (A4[0] - PX) // 2
    y0 = 300
    page.paste(marker, (x0, y0))
    d = ImageDraw.Draw(page)
    f, small = font(60), font(44)
    y = y0 + PX + 90
    d.text((A4[0] // 2, y), "FINDMYTEAM ENTRANCE MARKER", font=f, fill="black", anchor="mm")
    y += 90
    for line in (
        "Print at 100% / Actual size (no 'fit to page'). The image must be exactly 18.0 cm wide.",
        "Check with the 10 cm ruler below. Tape flat on a wall by the entrance, arrow up, ~1.4 m high.",
        "Don't move it during the demo: every phone's map is anchored to it.",
    ):
        d.text((A4[0] // 2, y), line, font=small, fill="black", anchor="mm")
        y += 64
    # 10 cm ruler with cm ticks
    y += 50
    cm = DPI / 2.54
    rx0 = (A4[0] - 10 * cm) / 2
    d.line([(rx0, y), (rx0 + 10 * cm, y)], fill="black", width=6)
    for i in range(11):
        h = 40 if i % 5 == 0 else 24
        d.line([(rx0 + i * cm, y - h), (rx0 + i * cm, y)], fill="black", width=5)
    d.text((A4[0] // 2, y + 50), "10 cm", font=small, fill="black", anchor="mm")
    return page


if __name__ == "__main__":
    m = make_marker()
    m.save(HERE / "marker.png", dpi=(DPI, DPI))
    make_pdf(m).save(HERE / "marker.pdf", resolution=DPI)
    copies = [HERE.parent / "ios" / "Sightline" / "Resources" / "marker.png",  # iPhone reference image
              HERE.parent / "android" / "app" / "src" / "main" / "assets" / "marker.png",  # Android reference image
              HERE.parent / "web" / "marker.png"]  # shown 15 cm wide on the commander's screen
    for c in copies:
        c.parent.mkdir(parents=True, exist_ok=True)
        m.save(c, dpi=(DPI, DPI))
    print(f"marker.png {PX}x{PX}px (= {WIDTH_CM} cm at {DPI} dpi), marker.pdf A4, copied to iOS, Android and the website")
