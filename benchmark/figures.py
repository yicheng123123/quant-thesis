"""Generate standardized PNG and SVG figures from benchmark summary rows."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 900
BACKGROUND = "#f7f8fb"
TEXT = "#172033"
MUTED = "#657089"
GRID = "#d8deea"
TRACK = "#e9edf5"
FRONTIER = "#16845b"
COLORS = {
    "FP16": "#2878ff",
    "GPTQ": "#7357d9",
    "AWQ": "#e08a1e",
}


def label(row: dict[str, Any]) -> str:
    special = {
        "awq_g64_duo_false": "AWQ G64 duo=False",
        "awq_g64_duo_true": "AWQ G64 duo=True",
    }
    if row.get("experiment_id") in special:
        return special[str(row["experiment_id"])]
    group_size = row.get("group_size")
    return str(row["method"]) if group_size is None else f"{row['method']} G{group_size}"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _method_color(row: dict[str, Any]) -> str:
    return COLORS.get(str(row["method"]), "#718096")


def _svg_text(value: Any) -> str:
    return escape(str(value))


def _write_svg(path: Path, title: str, subtitle: str, body: Iterable[str]) -> None:
    content = "\n".join(body)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>{_svg_text(title)}</title>
<desc>{_svg_text(subtitle)}</desc>
<rect width="100%" height="100%" fill="{BACKGROUND}"/>
<text x="70" y="70" font-family="Segoe UI, Arial, sans-serif" font-size="36" font-weight="700" fill="{TEXT}">{_svg_text(title)}</text>
<text x="70" y="104" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="{MUTED}">{_svg_text(subtitle)}</text>
{content}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _base_image(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((70, 38), title, font=_font(36, bold=True), fill=TEXT)
    draw.text((70, 84), subtitle, font=_font(17), fill=MUTED)
    return image, draw


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, Any]],
    y: int,
    note: str = "Muted = legacy frozen row",
) -> None:
    x = 70
    for method in ("FP16", "GPTQ", "AWQ"):
        if not any(row["method"] == method for row in rows):
            continue
        draw.rounded_rectangle((x, y, x + 18, y + 18), radius=4, fill=COLORS[method])
        draw.text((x + 28, y - 2), method, font=_font(16), fill=TEXT)
        x += 120
    draw.text((x + 12, y - 2), note, font=_font(16), fill=MUTED)


def _bar_figure(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    stem: str,
    title: str,
    subtitle: str,
    field: str,
    unit: str,
    digits: int,
    comparable_only: bool = False,
    std_field: str | None = None,
    std_unit: str | None = None,
    display_scale: float = 1.0,
    comparable_field: str | None = None,
    mute_legacy: bool = True,
) -> list[Path]:
    chart_rows = [
        row
        for row in rows
        if row.get(field) is not None
        and (not comparable_only or bool(row.get("system_metrics_comparable")))
        and (comparable_field is None or bool(row.get(comparable_field)))
    ]
    values = [
        float(row[field]) + (float(row.get(std_field) or 0) if std_field else 0)
        for row in chart_rows
    ]
    maximum = max(values, default=1.0) or 1.0
    x0, x1 = 300, 1450
    y0, row_height = 190, 92
    plot_height = row_height * max(len(chart_rows), 1)
    image, draw = _base_image(title, subtitle)
    body: list[str] = []

    for tick in range(6):
        value = maximum * 1.15 * tick / 5
        x = x0 + (x1 - x0) * tick / 5
        draw.line((x, y0 - 18, x, y0 + plot_height - 18), fill=GRID, width=2)
        draw.text((x - 18, y0 + plot_height + 4), f"{value * display_scale:.{digits}f}", font=_font(15), fill=MUTED)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{y0 - 18}" x2="{x:.1f}" y2="{y0 + plot_height - 18}" stroke="{GRID}" stroke-width="2"/>',
                f'<text x="{x:.1f}" y="{y0 + plot_height + 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{MUTED}">{value * display_scale:.{digits}f}</text>',
            ]
        )

    for index, row in enumerate(chart_rows):
        value = float(row[field])
        y = y0 + index * row_height
        bar_y = y + 10
        width = (x1 - x0) * value / (maximum * 1.15)
        method_color = _method_color(row)
        alpha = 150 if mute_legacy and row.get("provenance_status") == "legacy_frozen" else 255
        fill = method_color if alpha == 255 else "#9ea8bd"
        draw.text((70, y + 4), label(row), font=_font(20, bold=True), fill=TEXT)
        draw.text((70, y + 34), "legacy frozen" if alpha != 255 else "v1 rerun", font=_font(15), fill=MUTED)
        draw.rounded_rectangle((x0, bar_y, x1, bar_y + 34), radius=7, fill=TRACK)
        draw.rounded_rectangle((x0, bar_y, x0 + max(width, 4), bar_y + 34), radius=7, fill=fill)
        unit_separator = "" if unit == "%" else " "
        value_text = f"{value * display_scale:.{digits}f}{unit_separator}{unit}"
        if std_field and row.get(std_field) is not None:
            error_unit = std_unit or unit
            error_separator = "" if error_unit == "%" else " "
            value_text += (
                f" ± {float(row[std_field]) * display_scale:.{digits}f}"
                f"{error_separator}{error_unit}"
            )
            std = float(row[std_field])
            lower = x0 + (x1 - x0) * max(value - std, 0) / (maximum * 1.15)
            upper = x0 + (x1 - x0) * (value + std) / (maximum * 1.15)
            center = y + 27
            draw.line((lower, center, upper, center), fill=TEXT, width=3)
            draw.line((lower, center - 9, lower, center + 9), fill=TEXT, width=3)
            draw.line((upper, center - 9, upper, center + 9), fill=TEXT, width=3)
        draw.text((min(x0 + width + 14, 1320), y + 13), value_text, font=_font(17), fill=TEXT)
        body.extend(
            [
                f'<text x="70" y="{y + 28}" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="700" fill="{TEXT}">{_svg_text(label(row))}</text>',
                f'<text x="70" y="{y + 54}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{MUTED}">{"legacy frozen" if alpha != 255 else "v1 rerun"}</text>',
                f'<rect x="{x0}" y="{bar_y}" width="{x1 - x0}" height="34" rx="7" fill="{TRACK}"/>',
                f'<rect x="{x0}" y="{bar_y}" width="{max(width, 4):.1f}" height="34" rx="7" fill="{fill}"/>',
            ]
        )
        if std_field and row.get(std_field) is not None:
            std = float(row[std_field])
            lower = x0 + (x1 - x0) * max(value - std, 0) / (maximum * 1.15)
            upper = x0 + (x1 - x0) * (value + std) / (maximum * 1.15)
            center = y + 27
            body.extend(
                [
                    f'<line x1="{lower:.1f}" y1="{center}" x2="{upper:.1f}" y2="{center}" stroke="{TEXT}" stroke-width="3"/>',
                    f'<line x1="{lower:.1f}" y1="{center - 9}" x2="{lower:.1f}" y2="{center + 9}" stroke="{TEXT}" stroke-width="3"/>',
                    f'<line x1="{upper:.1f}" y1="{center - 9}" x2="{upper:.1f}" y2="{center + 9}" stroke="{TEXT}" stroke-width="3"/>',
                ]
            )
        body.append(
            f'<text x="{min(x0 + width + 14, 1320):.1f}" y="{y + 34}" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="{TEXT}">{_svg_text(value_text)}</text>'
        )

    legend_note = (
        "Error bars = standard error"
        if std_field and field.startswith("hellaswag_")
        else "Muted = legacy frozen row"
    )
    _draw_legend(draw, rows, 820, legend_note)
    body.extend(
        [
            f'<line x1="70" y1="838" x2="88" y2="838" stroke="{COLORS["FP16"]}" stroke-width="14" stroke-linecap="round"/>',
            f'<text x="98" y="844" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="{TEXT}">FP16</text>',
            f'<line x1="190" y1="838" x2="208" y2="838" stroke="{COLORS["GPTQ"]}" stroke-width="14" stroke-linecap="round"/>',
            f'<text x="218" y="844" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="{TEXT}">GPTQ</text>',
            f'<line x1="310" y1="838" x2="328" y2="838" stroke="{COLORS["AWQ"]}" stroke-width="14" stroke-linecap="round"/>',
            f'<text x="338" y="844" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="{TEXT}">AWQ</text>',
            f'<text x="450" y="844" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="{MUTED}">{_svg_text(legend_note)}</text>',
        ]
    )

    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    image.save(png_path, format="PNG", optimize=True)
    _write_svg(svg_path, title, subtitle, body)
    return [png_path, svg_path]


def _pareto(points: list[tuple[float, float, dict[str, Any]]]) -> list[tuple[float, float, dict[str, Any]]]:
    frontier = []
    for point in points:
        x, y, _ = point
        dominated = any(
            other_x <= x and other_y <= y and (other_x < x or other_y < y)
            for other_x, other_y, _ in points
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda item: item[0])


def _scatter_figure(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    stem: str,
    title: str,
    subtitle: str,
    x_field: str,
    x_label: str,
) -> list[Path]:
    points = [
        (float(row[x_field]), float(row["ppl_delta"]), row)
        for row in rows
        if row.get(x_field) is not None
        and row.get("ppl_delta") is not None
        and (x_field != "peak_vram_mib" or bool(row.get("system_metrics_comparable")))
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs, default=0), max(xs, default=1)
    ymin, ymax = min(ys, default=0), max(ys, default=1)
    xpad = max((xmax - xmin) * 0.12, 1.0)
    ypad = max((ymax - ymin) * 0.18, 0.2)
    xmin, xmax = max(0, xmin - xpad), xmax + xpad
    ymin, ymax = max(0, ymin - ypad), ymax + ypad
    x0, x1, y0, y1 = 190, 1450, 190, 760

    image, draw = _base_image(title, subtitle)
    body: list[str] = []
    def sx(value: float) -> float:
        return x0 + (value - xmin) / (xmax - xmin) * (x1 - x0)
    def sy(value: float) -> float:
        return y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)

    for tick in range(6):
        x_value = xmin + (xmax - xmin) * tick / 5
        x = sx(x_value)
        draw.line((x, y0, x, y1), fill=GRID, width=2)
        draw.text((x - 30, y1 + 16), f"{x_value:.0f}", font=_font(15), fill=MUTED)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{GRID}" stroke-width="2"/>',
                f'<text x="{x:.1f}" y="{y1 + 40}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{MUTED}">{x_value:.0f}</text>',
            ]
        )
    for tick in range(6):
        y_value = ymin + (ymax - ymin) * tick / 5
        y = sy(y_value)
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        draw.text((x0 - 72, y - 10), f"{y_value:.2f}", font=_font(15), fill=MUTED)
        body.extend(
            [
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="2"/>',
                f'<text x="{x0 - 18}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{MUTED}">{y_value:.2f}</text>',
            ]
        )

    frontier = _pareto(points)
    if len(frontier) > 1:
        draw.line([(sx(x), sy(y)) for x, y, _ in frontier], fill=FRONTIER, width=4)
        body.append(
            '<polyline points="' + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in frontier) + f'" fill="none" stroke="{FRONTIER}" stroke-width="4"/>'
        )

    for x_value, y_value, row in points:
        x, y = sx(x_value), sy(y_value)
        method_color = _method_color(row)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=method_color, outline=BACKGROUND, width=3)
        draw.text((x + 18, y - 12), label(row), font=_font(17, bold=True), fill=TEXT)
        body.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{method_color}" stroke="{BACKGROUND}" stroke-width="3"/>',
                f'<text x="{x + 18:.1f}" y="{y + 6:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="17" font-weight="700" fill="{TEXT}">{_svg_text(label(row))}</text>',
            ]
        )

    draw.text((x0 + (x1 - x0) / 2 - 120, 805), x_label, font=_font(18, bold=True), fill=TEXT)
    draw.text((30, 435), "ΔPPL vs FP16", font=_font(18, bold=True), fill=TEXT)
    body.extend(
        [
            f'<text x="{(x0 + x1) / 2:.1f}" y="850" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="{TEXT}">{_svg_text(x_label)}</text>',
            f'<text x="42" y="475" transform="rotate(-90 42 475)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="{TEXT}">ΔPPL vs FP16</text>',
            f'<text x="70" y="845" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="{FRONTIER}">Green line = Pareto frontier</text>',
        ]
    )

    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    image.save(png_path, format="PNG", optimize=True)
    _write_svg(svg_path, title, subtitle, body)
    return [png_path, svg_path]


def _downstream_scatter_figure(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> list[Path]:
    """Plot PPL delta against HellaSwag acc_norm with standard-error bars."""
    points = [
        (
            float(row["ppl_delta"]),
            float(row["hellaswag_acc_norm"]),
            float(row["hellaswag_acc_norm_stderr"]),
            row,
        )
        for row in rows
        if row.get("ppl_delta") is not None
        and row.get("hellaswag_acc_norm") is not None
        and row.get("hellaswag_acc_norm_stderr") is not None
        and row.get("downstream_quality_comparable")
    ]
    if not points:
        return []

    xs = [point[0] for point in points]
    y_lower = [point[1] - point[2] for point in points]
    y_upper = [point[1] + point[2] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(y_lower), max(y_upper)
    xpad = max((xmax - xmin) * 0.10, 0.15)
    ypad = max((ymax - ymin) * 0.18, 0.01)
    xmin, xmax = max(0.0, xmin - xpad), xmax + xpad
    ymin, ymax = max(0.0, ymin - ypad), min(1.0, ymax + ypad)
    x0, x1, y0, y1 = 190, 1450, 190, 760

    title = "PPL loss vs HellaSwag normalized accuracy"
    subtitle = "Zero-shot HellaSwag · vertical whiskers are standard error · association, not causation"
    image, draw = _base_image(title, subtitle)
    body: list[str] = []

    def sx(value: float) -> float:
        return x0 + (value - xmin) / (xmax - xmin) * (x1 - x0)

    def sy(value: float) -> float:
        return y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)

    for tick in range(6):
        value = xmin + (xmax - xmin) * tick / 5
        x = sx(value)
        draw.line((x, y0, x, y1), fill=GRID, width=2)
        draw.text((x - 24, y1 + 16), f"{value:.2f}", font=_font(15), fill=MUTED)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{GRID}" stroke-width="2"/>',
                f'<text x="{x:.1f}" y="{y1 + 40}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{MUTED}">{value:.2f}</text>',
            ]
        )
    for tick in range(6):
        value = ymin + (ymax - ymin) * tick / 5
        y = sy(value)
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        draw.text((x0 - 74, y - 10), f"{value * 100:.1f}%", font=_font(15), fill=MUTED)
        body.extend(
            [
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="2"/>',
                f'<text x="{x0 - 18}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="{MUTED}">{value * 100:.1f}%</text>',
            ]
        )

    label_offsets = {
        "awq_g64_duo_false": (-238, 16),
        "awq_g64_duo_true": (18, -38),
    }
    for x_value, y_value, stderr, row in points:
        x, y = sx(x_value), sy(y_value)
        low, high = sy(y_value - stderr), sy(y_value + stderr)
        color = _method_color(row)
        draw.line((x, high, x, low), fill=TEXT, width=3)
        draw.line((x - 9, high, x + 9, high), fill=TEXT, width=3)
        draw.line((x - 9, low, x + 9, low), fill=TEXT, width=3)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=color, outline=BACKGROUND, width=3)
        dx, dy = label_offsets.get(str(row.get("experiment_id")), (18, -12))
        draw.text((x + dx, y + dy), label(row), font=_font(17, bold=True), fill=TEXT)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{high:.1f}" x2="{x:.1f}" y2="{low:.1f}" stroke="{TEXT}" stroke-width="3"/>',
                f'<line x1="{x - 9:.1f}" y1="{high:.1f}" x2="{x + 9:.1f}" y2="{high:.1f}" stroke="{TEXT}" stroke-width="3"/>',
                f'<line x1="{x - 9:.1f}" y1="{low:.1f}" x2="{x + 9:.1f}" y2="{low:.1f}" stroke="{TEXT}" stroke-width="3"/>',
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}" stroke="{BACKGROUND}" stroke-width="3"/>',
                f'<text x="{x + dx:.1f}" y="{y + dy + 18:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="17" font-weight="700" fill="{TEXT}">{_svg_text(label(row))}</text>',
            ]
        )

    draw.text((690, 810), "ΔPPL vs FP16", font=_font(18, bold=True), fill=TEXT)
    axis_label = Image.new("RGBA", (260, 48), (0, 0, 0, 0))
    axis_draw = ImageDraw.Draw(axis_label)
    axis_draw.text((0, 8), "HellaSwag acc_norm", font=_font(18, bold=True), fill=TEXT)
    rotated_axis_label = axis_label.rotate(90, expand=True)
    image.paste(rotated_axis_label, (24, 350), rotated_axis_label)
    body.extend(
        [
            '<text x="820" y="850" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="{0}">ΔPPL vs FP16</text>'.format(TEXT),
            '<text x="42" y="475" transform="rotate(-90 42 475)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="{0}">HellaSwag acc_norm</text>'.format(TEXT),
        ]
    )

    png_path = output_dir / "ppl_vs_hellaswag_acc_norm.png"
    svg_path = output_dir / "ppl_vs_hellaswag_acc_norm.svg"
    image.save(png_path, format="PNG", optimize=True)
    _write_svg(svg_path, title, subtitle, body)
    return [png_path, svg_path]


def generate_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Generate the standardized v1.2 figure set in PNG and SVG formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(
        _bar_figure(
            rows,
            output_dir=output_dir,
            stem="ppl_comparison",
            title="Perplexity comparison",
            subtitle="WikiText-2 full test split · lower is better",
            field="ppl",
            unit="PPL",
            digits=4,
        )
    )
    paths.extend(
        _bar_figure(
            rows,
            output_dir=output_dir,
            stem="ppl_delta",
            title="Quality loss relative to FP16",
            subtitle="Absolute PPL delta · shorter is better",
            field="ppl_delta",
            unit="ΔPPL",
            digits=4,
        )
    )
    paths.extend(
        _bar_figure(
            rows,
            output_dir=output_dir,
            stem="vram_comparison",
            title="Peak allocated VRAM",
            subtitle="Comparable system rows only · AWQ excluded because comparable=false",
            field="peak_vram_mib",
            unit="MiB",
            digits=1,
            comparable_only=True,
        )
    )
    paths.extend(
        _bar_figure(
            rows,
            output_dir=output_dir,
            stem="throughput",
            title="Generation throughput",
            subtitle="Comparable system rows only · error bars show mean ± std when available",
            field="tokens_per_second_mean",
            unit="tok/s",
            digits=2,
            comparable_only=True,
            std_field="tokens_per_second_std",
        )
    )
    paths.extend(
        _scatter_figure(
            rows,
            output_dir=output_dir,
            stem="quality_vs_memory_tradeoff",
            title="Quality–memory trade-off",
            subtitle="Comparable system rows only · lower-left is better · green line is Pareto frontier",
            x_field="peak_vram_mib",
            x_label="Peak allocated VRAM (MiB)",
        )
    )
    paths.extend(
        _scatter_figure(
            rows,
            output_dir=output_dir,
            stem="quality_vs_checkpoint_tradeoff",
            title="Quality–checkpoint trade-off",
            subtitle="All rows with checkpoint size · lower-left is better · green line is Pareto frontier",
            x_field="checkpoint_mib",
            x_label="Checkpoint size (MiB)",
        )
    )
    paths.extend(
        _bar_figure(
            rows,
            output_dir=output_dir,
            stem="hellaswag_acc",
            title="HellaSwag accuracy",
            subtitle="Zero-shot continuation likelihood · error bars show standard error",
            field="hellaswag_acc",
            unit="%",
            digits=2,
            std_field="hellaswag_acc_stderr",
            std_unit="pp",
            display_scale=100.0,
            comparable_field="downstream_quality_comparable",
            mute_legacy=False,
        )
    )
    paths.extend(
        _bar_figure(
            rows,
            output_dir=output_dir,
            stem="hellaswag_acc_norm",
            title="HellaSwag normalized accuracy",
            subtitle="Primary downstream metric · zero-shot · error bars show standard error",
            field="hellaswag_acc_norm",
            unit="%",
            digits=2,
            std_field="hellaswag_acc_norm_stderr",
            std_unit="pp",
            display_scale=100.0,
            comparable_field="downstream_quality_comparable",
            mute_legacy=False,
        )
    )
    paths.extend(_downstream_scatter_figure(rows, output_dir=output_dir))
    return paths
