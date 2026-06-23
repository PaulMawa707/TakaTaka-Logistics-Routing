"""Generate takataka.gif from takataka.png for the login hero."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "takataka.png"
OUT_DIR = ROOT / "static" / "img"
OUT = OUT_DIR / "takataka.gif"

FRAMES = 18
DURATION_MS = 120


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        raise SystemExit(f"Source image not found: {SRC}")

    base = Image.open(SRC).convert("RGBA")
    w, h = base.size
    frames = []
    for i in range(FRAMES):
        scale = 1.0 + 0.04 * (i / (FRAMES - 1))
        nw, nh = int(w * scale), int(h * scale)
        resized = base.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - w) // 2
        top = (nh - h) // 2
        cropped = resized.crop((left, top, left + w, top + h))
        frames.append(cropped.convert("RGB"))

    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
