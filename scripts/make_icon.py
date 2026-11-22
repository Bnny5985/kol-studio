"""Draw the app icon (Hebrew letter ק with sound waves) -> packaging/mac/icon.icns + packaging/windows/icon.ico"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
bg = Image.new("RGBA", (S, S))
top, bottom = (58, 98, 222), (109, 74, 216)
for y in range(S):  # vertical gradient
    t = y / S
    ImageDraw.Draw(bg).line([(0, y), (S, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([40, 40, S - 40, S - 40], radius=220, fill=255)
img.paste(bg, (0, 0), mask)
d = ImageDraw.Draw(img)
font = None
for f in ["/System/Library/Fonts/Supplemental/Arial Hebrew.ttc", "/System/Library/Fonts/ArialHB.ttc",
          "C:/Windows/Fonts/arialbd.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]:
    try:
        font = ImageFont.truetype(f, 560)
        break
    except OSError:
        continue
d.text((S * 0.62, S * 0.50), "ק", font=font, fill=(255, 255, 255, 255), anchor="mm")
for i, r in enumerate((90, 160, 230)):  # sound waves to the left of the letter (Hebrew reads right to left)
    box = [S * 0.42 - r, S * 0.50 - r, S * 0.42 + r, S * 0.50 + r]
    d.arc(box, 150, 210, fill=(255, 210, 94, 255 - i * 55), width=38)

(ROOT / "packaging/mac").mkdir(parents=True, exist_ok=True)
(ROOT / "packaging/windows").mkdir(parents=True, exist_ok=True)
img.save(ROOT / "packaging/icon.png")
img.save(ROOT / "packaging/windows/icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
if sys.platform == "darwin":
    with tempfile.TemporaryDirectory() as t:
        iconset = Path(t) / "icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            img.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            img.resize((size * 2, size * 2), Image.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(ROOT / "packaging/mac/icon.icns")], check=True)
print("icons written")
