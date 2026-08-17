from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    filename: str
    width: int
    height: int
    file_size: int
    modified_time_ns: int


@dataclass(frozen=True)
class ScanIssue:
    relative_path: str
    message: str


def scan_folder(source_root: str | Path) -> tuple[list[ImageRecord], list[ScanIssue]]:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"图片文件夹不存在：{root}")
    records: list[ImageRecord] = []
    issues: list[ScanIssue] = []
    paths = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.casefold() in SUPPORTED_EXTENSIONS),
        key=lambda p: p.relative_to(root).as_posix().casefold(),
    )
    for path in paths:
        rel = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                oriented = ImageOps.exif_transpose(image)
                width, height = oriented.size
            stat = path.stat()
            records.append(ImageRecord(rel, path.name, width, height, stat.st_size, stat.st_mtime_ns))
        except Exception as exc:
            issues.append(ScanIssue(rel, str(exc)))
    return records, issues
