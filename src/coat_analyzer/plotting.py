from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REGION_STYLE = {
    "left": (35, 105, 210),
    "middle": (92, 55, 150),
    "right": (215, 65, 65),
}
REGION_LABEL = {"left": "Left", "middle": "Middle", "right": "Right"}


def save_distribution_plot(
    path: str | Path,
    title: str,
    series: dict[str, dict[str, object]],
    calibration_label: str,
) -> None:
    """Draw relative-frequency colour-depth distributions without a plotting dependency."""
    width, height = 1200, 760
    left, right, top, bottom = 105, 55, 95, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28, bold=True)
    label_font = _font(19)
    small_font = _font(16)
    legend_font = _font(18, bold=True)

    maximum = 0.0
    for item in series.values():
        values = np.asarray(item["mean"], dtype=float)
        sem = np.asarray(item.get("sem", np.zeros_like(values)), dtype=float)
        maximum = max(maximum, float(np.max(values + sem)) if values.size else 0.0)
    maximum = max(maximum * 1.12, 0.01)

    draw.text((left, 28), title, fill=(25, 32, 44), font=title_font)
    draw.text((width - right, 38), calibration_label, fill=(85, 96, 112), font=small_font, anchor="ra")
    for tick in range(0, 6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * tick / 5
        draw.line((left, y, left + plot_w, y), fill=(225, 229, 236), width=1)
        draw.text((left - 12, y), f"{value:.3f}", fill=(90, 100, 115), font=small_font, anchor="rm")
    for depth in range(0, 101, 20):
        x = left + plot_w * depth / 100
        draw.line((x, top + plot_h, x, top + plot_h + 7), fill=(65, 73, 86), width=2)
        draw.text((x, top + plot_h + 13), str(depth), fill=(65, 73, 86), font=small_font, anchor="ma")
    draw.line((left, top, left, top + plot_h), fill=(45, 52, 64), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(45, 52, 64), width=2)

    for region in ("left", "middle", "right"):
        if region not in series:
            continue
        item = series[region]
        mean = np.asarray(item["mean"], dtype=float)
        sem = np.asarray(item.get("sem", np.zeros_like(mean)), dtype=float)
        color = REGION_STYLE[region]
        centers = np.arange(mean.size, dtype=float) + 0.5
        points = [
            (left + plot_w * x / 100, top + plot_h - plot_h * y / maximum)
            for x, y in zip(centers, mean)
        ]
        if np.any(sem > 0):
            upper = [
                (left + plot_w * x / 100, top + plot_h - plot_h * min(y + e, maximum) / maximum)
                for x, y, e in zip(centers, mean, sem)
            ]
            lower = [
                (left + plot_w * x / 100, top + plot_h - plot_h * max(y - e, 0) / maximum)
                for x, y, e in zip(centers[::-1], mean[::-1], sem[::-1])
            ]
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).polygon(upper + lower, fill=(*color, 42))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(canvas)
        if len(points) >= 2:
            draw.line(points, fill=color, width=4, joint="curve")
        median = item.get("median")
        if median is not None:
            x = left + plot_w * float(median) / 100
            for y in range(top, top + plot_h, 14):
                draw.line((x, y, x, min(y + 7, top + plot_h)), fill=color, width=2)

    draw.text((left + plot_w / 2, height - 47), "Colour depth (0 = white, 100 = black)",
              fill=(35, 43, 55), font=label_font, anchor="ma")
    vertical = Image.new("RGBA", (260, 40), (0, 0, 0, 0))
    ImageDraw.Draw(vertical).text((130, 20), "Relative frequency", fill=(35, 43, 55, 255),
                                  font=label_font, anchor="mm")
    vertical = vertical.rotate(90, expand=True)
    canvas.paste(vertical, (17, int(top + plot_h / 2 - vertical.height / 2)), vertical)
    draw = ImageDraw.Draw(canvas)
    legend_x = left
    for region in ("left", "middle", "right"):
        if region not in series:
            continue
        item = series[region]
        color = REGION_STYLE[region]
        draw.line((legend_x, 73, legend_x + 28, 73), fill=color, width=5)
        suffix = f" (n={item['n']})" if item.get("n") else ""
        label = REGION_LABEL[region] + suffix
        draw.text((legend_x + 37, 73), label, fill=(40, 48, 60), font=legend_font, anchor="lm")
        legend_x += 190
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG")


def frequencies_from_l_histogram(histogram: dict[str, object]) -> np.ndarray:
    """Reverse L* bins so the x-axis increases from white to black."""
    counts = np.asarray(histogram["counts"], dtype=float)[::-1]
    return counts / max(float(counts.sum()), 1.0)


def _font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()
