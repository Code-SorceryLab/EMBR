"""Make the portraits' flat backgrounds transparent, without eating Dawn's white hair.

The source art is drawn on a near-white field, and Dawn's hair is *also* near-white, so a
naive "make white transparent" pass would delete half of her. The fix is a border flood fill:
transparency only spreads inward from the edges and stops at the first dark outline, so the
enclosed white of her hair is never reached. A short feather softens the cut edge.

    python assets/portraits/cutout.py            # process the Dawn portraits in place

Re-running is safe: the field stays clear and the subject stays opaque (the 1px feather may
re-touch the cut edge by a shade, so it is not byte-for-byte identical). The player icon is a
circular badge that is meant to keep its ground, so it is not processed.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from PIL import Image, ImageFilter

HERE = Path(__file__).parent
#: The four Dawn expressions. The player badge keeps its disc, so it is not in this list.
TARGETS = ("dawn-warm", "dawn-neutral", "dawn-suspicious", "dawn-betrayed")

#: A pixel counts as "background" only if every channel is at least this bright. High, so only
#: the paper-white field is caught and the grey shading in her hair is safe.
WHITE = 236
#: How far a pixel may sit from pure white and still be flooded, as a sum over channels. Keeps
#: the faint cream of the field in scope without reaching the drawn linework.
TOLERANCE = 54


def _is_background(pixel: tuple[int, int, int, int]) -> bool:
    r, g, b, a = pixel
    if a == 0:
        return True
    return min(r, g, b) >= WHITE and (255 - r) + (255 - g) + (255 - b) <= TOLERANCE


def cut_out(path: Path) -> None:
    """Flood the outer white field to transparent from every border pixel inward."""
    image = image_rgba(path)
    width, height = image.size
    px = image.load()

    seen = bytearray(width * height)
    queue: deque[tuple[int, int]] = deque()

    def consider(x: int, y: int) -> None:
        if 0 <= x < width and 0 <= y < height and not seen[y * width + x]:
            seen[y * width + x] = 1
            if _is_background(px[x, y]):
                queue.append((x, y))

    for x in range(width):
        consider(x, 0)
        consider(x, height - 1)
    for y in range(height):
        consider(0, y)
        consider(width - 1, y)

    while queue:
        x, y = queue.popleft()
        px[x, y] = (255, 255, 255, 0)
        consider(x + 1, y)
        consider(x - 1, y)
        consider(x, y + 1)
        consider(x, y - 1)

    _feather(image).save(path)


def image_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _feather(image: Image.Image) -> Image.Image:
    """Soften the alpha edge by one pixel so the cut does not read as a hard sticker line."""
    alpha = image.getchannel("A").filter(ImageFilter.GaussianBlur(0.6))
    image.putalpha(alpha)
    return image


def main() -> None:
    for name in TARGETS:
        path = HERE / f"{name}.png"
        if not path.exists():
            print(f"skip {name}: not found")
            continue
        cut_out(path)
        print(f"cut {name}")


if __name__ == "__main__":
    main()
