from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from .analysis import analyze_image, load_rgb
from .plotting import frequencies_from_l_histogram, save_distribution_plot
from .project_store import connect, get_image, image_path, open_project, save_measurement


REGION_NAMES = {"left": "左侧", "middle": "中间", "right": "右侧"}


def export_project(database_path: str | Path, output_root: str | Path | None = None) -> Path:
    database = Path(database_path).resolve()
    project = open_project(database)
    target_root = Path(output_root).expanduser().resolve() if output_root else Path(project["source_root"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = target_root / f"{database.stem}_results_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    annotated_dir = output / "annotated_images"
    annotated_dir.mkdir()

    with connect(database) as conn:
        image_rows = conn.execute(
            """SELECT image.id,image.relative_path,image.filename,image.status,image.low_quality,
                      experiment_group.name AS group_name
               FROM image LEFT JOIN experiment_group ON experiment_group.id=image.group_id
               ORDER BY image.relative_path COLLATE NOCASE"""
        ).fetchall()

    summary_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    legacy_histogram_rows: list[dict[str, object]] = []
    center_index_rows: list[dict[str, object]] = []
    hair_area_rows: list[dict[str, object]] = []
    manifest_images: list[dict[str, object]] = []
    group_distributions: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {
        "raw": {}, "gray_corrected": {}
    }
    for row in image_rows:
        if row["status"] != "reviewed" or row["low_quality"]:
            manifest_images.append({
                "image_id": row["id"], "relative_path": row["relative_path"],
                "status": row["status"], "exported": False,
            })
            continue
        annotation = get_image(database, int(row["id"]))
        split_left = float(project["standard_left"] if project["use_standard_export"] else annotation["split_left"])
        split_right = float(project["standard_right"] if project["use_standard_export"] else annotation["split_right"])
        source = image_path(database, int(row["id"]))
        image = load_rgb(source)
        result = analyze_image(
            image, annotation["roi"], annotation["left_edges"], annotation["right_edges"],
            annotation["exclusions"], annotation["gray_card"], split_left, split_right,
            float(project["gray_target_l"]),
            float(annotation["white_depth_min"]), float(annotation["white_depth_max"]),
            float(annotation["black_depth_min"]), float(annotation["black_depth_max"]),
            max_side=420,
        )
        save_measurement(database, int(row["id"]), result.parameter_hash, result.metrics)
        gray_available = bool(result.metrics["gray_card"]["available"])
        image_series: dict[str, dict[str, dict[str, object]]] = {"raw": {}, "gray_corrected": {}}
        for region_key, metrics in result.metrics["regions"].items():
            if metrics.get("pixel_count", 0) == 0:
                continue
            base = {
                "image_id": row["id"], "relative_path": row["relative_path"],
                "group": row["group_name"] or "", "region": region_key,
                "region_label": REGION_NAMES[region_key],
                "left_boundary_percent": split_left * 100,
                "right_boundary_percent": split_right * 100,
                "gray_card_available": int(gray_available),
            }
            summary_rows.append({**base, **{k: v for k, v in metrics.items() if not k.startswith("histogram_") and k != "l_percentiles"},
                                 **{f"l_p{k}": v for k, v in metrics["l_percentiles"].items()}})
            raw_hist = metrics["histogram_l_raw"]
            for index, count in enumerate(raw_hist["counts"]):
                legacy_histogram_rows.append({**base, "bin_left_l": raw_hist["edges"][index],
                                              "bin_right_l": raw_hist["edges"][index + 1], "pixel_count": count})
            modes = [("raw", "histogram_l_raw", "median_darkness")]
            if gray_available:
                modes.append(("gray_corrected", "histogram_l_corrected", "median_corrected_darkness"))
            for mode, histogram_key, median_key in modes:
                histogram = metrics[histogram_key]
                frequency = frequencies_from_l_histogram(histogram)
                image_series[mode][region_key] = {
                    "mean": frequency, "median": metrics[median_key], "n": 1,
                }
                for index, (count, relative) in enumerate(zip(histogram["counts"][::-1], frequency)):
                    histogram_rows.append({
                        **base, "mode": mode, "bin_left_depth": index,
                        "bin_right_depth": index + 1, "pixel_count": int(count),
                        "relative_frequency": float(relative),
                    })
        group_name = str(row["group_name"] or "Ungrouped")
        plot_stem = f"{int(row['id']):04d}_{_safe_name(Path(row['filename']).stem)}"
        raw_plot = Path("histogram_plots") / "raw" / "individual" / f"{plot_stem}.png"
        save_distribution_plot(
            output / raw_plot, f"{row['relative_path']} - colour depth distribution",
            image_series["raw"], "Raw",
        )
        corrected_plot: Path | None = None
        if gray_available:
            corrected_plot = Path("histogram_plots") / "gray_corrected" / "individual" / f"{plot_stem}.png"
            save_distribution_plot(
                output / corrected_plot, f"{row['relative_path']} - colour depth distribution",
                image_series["gray_corrected"], "Gray-card corrected",
            )
        for mode in ("raw", "gray_corrected"):
            if not image_series[mode]:
                continue
            group_bucket = group_distributions[mode].setdefault(
                group_name, {"left": [], "middle": [], "right": []}
            )
            for region_key, item in image_series[mode].items():
                group_bucket[region_key].append(item)
        regions = result.metrics["regions"]
        center_index_rows.append(_center_index_row(row, split_left, split_right, regions, "raw"))
        if gray_available:
            center_index_rows.append(
                _center_index_row(row, split_left, split_right, regions, "gray_corrected")
            )
        for mode in ("raw", "gray_corrected"):
            if mode == "gray_corrected" and not gray_available:
                continue
            classification = result.metrics["hair_classification"]["modes"][mode]
            definition = result.metrics["hair_classification"]["definition"]
            for region_key, area in [("overall", classification["overall"]), *classification["regions"].items()]:
                hair_area_rows.append({
                    "image_id": row["id"], "relative_path": row["relative_path"],
                    "group": row["group_name"] or "", "mode": mode,
                    "region": region_key, **definition, **area,
                })
        annotated_name = f"{int(row['id']):04d}_{Path(row['filename']).stem}.png"
        _save_annotated(image, annotation, result.contours, annotated_dir / annotated_name)
        manifest_images.append({
            "image_id": row["id"], "relative_path": row["relative_path"],
            "status": row["status"], "exported": True, "parameter_hash": result.parameter_hash,
            "annotated_image": f"annotated_images/{annotated_name}",
            "raw_histogram_plot": raw_plot.as_posix(),
            "corrected_histogram_plot": None if corrected_plot is None else corrected_plot.as_posix(),
        })

    _write_csv(output / "region_summary.csv", summary_rows)
    _write_csv(output / "lstar_histograms.csv", legacy_histogram_rows)
    _write_csv(output / "color_depth_histograms.csv", histogram_rows)
    _write_csv(output / "center_darkening_index.csv", center_index_rows)
    _write_csv(output / "hair_area_summary.csv", hair_area_rows)
    group_plot_paths = _save_group_plots(output, group_distributions)
    manifest = {
        "project_name": project["name"], "database_path": str(database),
        "export_mode": "project_standard" if project["use_standard_export"] else "per_image",
        "standard_left": project["standard_left"], "standard_right": project["standard_right"],
        "gray_target_l": project["gray_target_l"], "images": manifest_images,
        "group_histogram_plots": group_plot_paths,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def _save_group_plots(
    output: Path,
    grouped: dict[str, dict[str, dict[str, list[dict[str, object]]]]],
) -> list[str]:
    paths: list[str] = []
    for mode, groups in grouped.items():
        for group_name, regions in groups.items():
            series: dict[str, dict[str, object]] = {}
            for region_key, items in regions.items():
                if not items:
                    continue
                values = np.stack([np.asarray(item["mean"], dtype=float) for item in items])
                sem = np.zeros(values.shape[1], dtype=float)
                if values.shape[0] > 1:
                    sem = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
                series[region_key] = {
                    "mean": values.mean(axis=0), "sem": sem,
                    "median": float(np.mean([float(item["median"]) for item in items])),
                    "n": len(items),
                }
            relative = Path("histogram_plots") / mode / "groups" / f"{_safe_name(group_name)}.png"
            save_distribution_plot(
                output / relative, f"{group_name} - mean colour depth distribution",
                series, "Raw" if mode == "raw" else "Gray-card corrected",
            )
            paths.append(relative.as_posix())
    return paths


def _center_index_row(row, split_left: float, split_right: float, regions, mode: str) -> dict[str, object]:
    key = "median_darkness" if mode == "raw" else "median_corrected_darkness"
    left = float(regions["left"][key])
    middle = float(regions["middle"][key])
    right = float(regions["right"][key])
    return {
        "image_id": row["id"], "relative_path": row["relative_path"],
        "group": row["group_name"] or "", "mode": mode,
        "left_boundary_percent": split_left * 100,
        "right_boundary_percent": split_right * 100,
        "left_median_depth": left, "middle_median_depth": middle,
        "right_median_depth": right,
        "middle_minus_left": middle - left,
        "middle_minus_right": middle - right,
        "center_darkening_index": middle - (left + right) / 2,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" .")
    return cleaned[:120] or "unnamed"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_annotated(image, annotation, contours, path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    roi = [tuple(point) for point in annotation["roi"]]
    if len(roi) >= 3:
        draw.line(roi + [roi[0]], fill=(255, 215, 0, 255), width=4)
    for index in annotation["left_edges"]:
        a, b = roi[index % len(roi)], roi[(index + 1) % len(roi)]
        draw.line([a, b], fill=(30, 144, 255, 255), width=7)
    for index in annotation["right_edges"]:
        a, b = roi[index % len(roi)], roi[(index + 1) % len(roi)]
        draw.line([a, b], fill=(255, 80, 80, 255), width=7)
    for polygon in annotation["exclusions"]:
        points = [tuple(point) for point in polygon]
        if len(points) >= 3:
            draw.polygon(points, fill=(255, 0, 255, 55), outline=(255, 0, 255, 255), width=3)
    if len(annotation["gray_card"]) >= 3:
        points = [tuple(point) for point in annotation["gray_card"]]
        draw.polygon(points, fill=(0, 220, 190, 45), outline=(0, 220, 190, 255), width=3)
    for key, color in [("left_middle", (0, 255, 120, 255)), ("middle_right", (255, 140, 0, 255))]:
        for a, b in contours[key]:
            draw.line([tuple(a), tuple(b)], fill=color, width=4)
    canvas.save(path, "PNG")
