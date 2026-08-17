from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .scanner import scan_folder


SCHEMA_VERSION = 2


class ClosingConnection(sqlite3.Connection):
    """A sqlite connection whose context manager also releases the file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def now_text() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path), factory=ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_project(
    database_path: str | Path | None,
    project_name: str,
    source_root: str | Path,
    standard_left: float = 0.4,
    standard_right: float = 0.6,
) -> Path:
    source = Path(source_root).expanduser().resolve()
    if database_path:
        output = Path(database_path).expanduser().resolve()
    else:
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", project_name.strip()).strip(" .")
        output = source / f"{safe_name or 'coat_analysis_project'}.sqlite"
    if output.suffix.casefold() not in {".sqlite", ".db"}:
        output = output.with_suffix(".sqlite")
    if output.exists():
        raise FileExistsError(f"项目文件已存在：{output}")
    records, issues = scan_folder(source)
    if not records:
        raise ValueError("所选文件夹中没有可读取的图片")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = now_text()
    conn = sqlite3.connect(output)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_SQL)
        with conn:
            conn.execute(
                """INSERT INTO project
                   (id,name,created_at,app_version,schema_version,source_root,
                    standard_left,standard_right,use_standard_export,gray_target_l)
                   VALUES (?,?,?,?,?,?,?,?,1,50.0)""",
                (str(uuid.uuid4()), project_name.strip(), created, __version__, SCHEMA_VERSION,
                 str(source), standard_left, standard_right),
            )
            for record in records:
                cursor = conn.execute(
                    """INSERT INTO image
                       (relative_path,filename,width,height,file_size,modified_time_ns,status,low_quality)
                       VALUES (?,?,?,?,?,?,'not_reviewed',0)""",
                    (record.relative_path, record.filename, record.width, record.height,
                     record.file_size, record.modified_time_ns),
                )
                conn.execute(
                    "INSERT INTO annotation (image_id,split_left,split_right,updated_at) VALUES (?,?,?,?)",
                    (cursor.lastrowid, standard_left, standard_right, created),
                )
            for issue in issues:
                conn.execute("INSERT INTO scan_issue VALUES (?,?)", (issue.relative_path, issue.message))
            _event(conn, "project_created", {"image_count": len(records), "issue_count": len(issues)})
    except Exception:
        conn.close()
        if output.exists():
            output.unlink()
        raise
    finally:
        conn.close()
    return output


def open_project(path: str | Path) -> dict[str, object]:
    database = Path(path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"项目文件不存在：{database}")
    with connect(database) as conn:
        _migrate_schema(conn)
        row = conn.execute("SELECT * FROM project LIMIT 1").fetchone()
        if row is None or row["schema_version"] != SCHEMA_VERSION:
            raise ValueError("不是兼容的 Surgical Coat Analyzer 项目文件")
        data = dict(row)
        data["database_path"] = str(database)
        return data


def list_images(path: str | Path) -> list[dict[str, object]]:
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT image.*, experiment_group.name AS group_name
               FROM image LEFT JOIN experiment_group ON experiment_group.id=image.group_id
               ORDER BY image.relative_path COLLATE NOCASE"""
        ).fetchall()
        return [dict(row) for row in rows]


def get_image(path: str | Path, image_id: int) -> dict[str, object]:
    with connect(path) as conn:
        row = conn.execute(
            """SELECT image.*, annotation.roi_json,annotation.left_edges_json,
                      annotation.right_edges_json,annotation.exclusions_json,
                      annotation.gray_card_json,annotation.split_left,annotation.split_right,
                      annotation.white_depth_min,annotation.white_depth_max,
                      annotation.black_depth_min,annotation.black_depth_max,
                      annotation.updated_at,experiment_group.name AS group_name
               FROM image JOIN annotation ON annotation.image_id=image.id
               LEFT JOIN experiment_group ON experiment_group.id=image.group_id
               WHERE image.id=?""",
            (image_id,),
        ).fetchone()
        if row is None:
            raise KeyError("图片不存在")
        result = dict(row)
        result.update({
            "roi": _loads(result.pop("roi_json"), []),
            "left_edges": _loads(result.pop("left_edges_json"), []),
            "right_edges": _loads(result.pop("right_edges_json"), []),
            "exclusions": _loads(result.pop("exclusions_json"), []),
            "gray_card": _loads(result.pop("gray_card_json"), []),
        })
        return result


def image_path(path: str | Path, image_id: int) -> Path:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT project.source_root,image.relative_path FROM project,image WHERE image.id=?", (image_id,)
        ).fetchone()
        if row is None:
            raise KeyError("图片不存在")
        root = Path(row["source_root"]).resolve()
        target = (root / row["relative_path"]).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("图片路径超出项目根目录") from exc
        return target


def save_annotation(path: str | Path, image_id: int, data: dict[str, object], status: str | None = None) -> None:
    white_min = float(data.get("white_depth_min", 0.0))
    white_max = float(data.get("white_depth_max", 25.0))
    black_min = float(data.get("black_depth_min", 60.0))
    black_max = float(data.get("black_depth_max", 100.0))
    if not (0 <= white_min < white_max <= black_min < black_max <= 100):
        raise ValueError("黑白发范围无效：各范围不能重叠，且必须在 0–100 之间")
    timestamp = now_text()
    with connect(path) as conn:
        conn.execute(
            """UPDATE annotation SET roi_json=?,left_edges_json=?,right_edges_json=?,
                      exclusions_json=?,gray_card_json=?,split_left=?,split_right=?,
                      white_depth_min=?,white_depth_max=?,black_depth_min=?,black_depth_max=?,updated_at=?
               WHERE image_id=?""",
            (json.dumps(data.get("roi", [])), json.dumps(data.get("left_edges", [])),
             json.dumps(data.get("right_edges", [])), json.dumps(data.get("exclusions", [])),
             json.dumps(data.get("gray_card", [])), float(data.get("split_left", .4)),
             float(data.get("split_right", .6)), white_min, white_max, black_min,
             black_max, timestamp, image_id),
        )
        if status:
            if status not in {"not_reviewed", "reviewed", "low_quality", "no_visible_region"}:
                raise ValueError("未知复核状态")
            conn.execute(
                "UPDATE image SET status=?,low_quality=? WHERE id=?",
                (status, int(status in {"low_quality", "no_visible_region"}), image_id),
            )
        conn.execute("DELETE FROM measurement WHERE image_id=?", (image_id,))
        _event(conn, "annotation_saved", {"image_id": image_id, "status": status})


def save_measurement(path: str | Path, image_id: int, parameter_hash: str, metrics: dict[str, object]) -> None:
    with connect(path) as conn:
        conn.execute(
            """INSERT INTO measurement (image_id,parameter_hash,metrics_json,calculated_at)
               VALUES (?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET
               parameter_hash=excluded.parameter_hash,metrics_json=excluded.metrics_json,
               calculated_at=excluded.calculated_at""",
            (image_id, parameter_hash, json.dumps(metrics, ensure_ascii=False), now_text()),
        )


def list_groups(path: str | Path) -> list[dict[str, object]]:
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT experiment_group.id,experiment_group.name,COUNT(image.id) AS image_count
               FROM experiment_group LEFT JOIN image ON image.group_id=experiment_group.id
               GROUP BY experiment_group.id ORDER BY experiment_group.name COLLATE NOCASE"""
        ).fetchall()
        return [dict(row) for row in rows]


def create_group(path: str | Path, name: str) -> dict[str, object]:
    clean = name.strip()
    if not clean:
        raise ValueError("分组名称不能为空")
    with connect(path) as conn:
        cursor = conn.execute(
            "INSERT INTO experiment_group (name,created_at) VALUES (?,?)", (clean, now_text())
        )
        return {"id": cursor.lastrowid, "name": clean, "image_count": 0}


def assign_group(path: str | Path, group_id: int | None, image_ids: list[int]) -> None:
    with connect(path) as conn:
        if group_id is not None and conn.execute(
            "SELECT 1 FROM experiment_group WHERE id=?", (group_id,)
        ).fetchone() is None:
            raise KeyError("实验组不存在")
        conn.executemany("UPDATE image SET group_id=? WHERE id=?", [(group_id, int(i)) for i in image_ids])
        _event(conn, "images_grouped", {"group_id": group_id, "image_ids": image_ids})


def update_settings(path: str | Path, settings: dict[str, object]) -> dict[str, object]:
    left = float(settings["standard_left"])
    right = float(settings["standard_right"])
    target = float(settings.get("gray_target_l", 50.0))
    if not 0 < left < right < 1:
        raise ValueError("统一分界值无效")
    if not 0 < target <= 100:
        raise ValueError("灰卡标准 L* 必须在 0–100 之间")
    with connect(path) as conn:
        conn.execute(
            "UPDATE project SET standard_left=?,standard_right=?,use_standard_export=?,gray_target_l=?",
            (left, right, int(bool(settings.get("use_standard_export", True))), target),
        )
    return open_project(path)


def _event(conn: sqlite3.Connection, event_type: str, payload: dict[str, object]) -> None:
    conn.execute(
        "INSERT INTO event_log (created_at,event_type,payload_json) VALUES (?,?,?)",
        (now_text(), event_type, json.dumps(payload, ensure_ascii=False)),
    )


def _loads(value: str | None, default: object) -> object:
    return default if not value else json.loads(value)


def _migrate_schema(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT schema_version FROM project LIMIT 1").fetchone()
    if row is None:
        return
    version = int(row["schema_version"])
    if version > SCHEMA_VERSION:
        raise ValueError("项目文件由更高版本的软件创建，当前版本无法打开")
    if version < 2:
        conn.execute("ALTER TABLE annotation ADD COLUMN white_depth_min REAL NOT NULL DEFAULT 0.0")
        conn.execute("ALTER TABLE annotation ADD COLUMN white_depth_max REAL NOT NULL DEFAULT 25.0")
        conn.execute("ALTER TABLE annotation ADD COLUMN black_depth_min REAL NOT NULL DEFAULT 60.0")
        conn.execute("ALTER TABLE annotation ADD COLUMN black_depth_max REAL NOT NULL DEFAULT 100.0")
        conn.execute("UPDATE project SET schema_version=2")


SCHEMA_SQL = """
CREATE TABLE project (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, app_version TEXT NOT NULL,
 schema_version INTEGER NOT NULL, source_root TEXT NOT NULL, standard_left REAL NOT NULL,
 standard_right REAL NOT NULL, use_standard_export INTEGER NOT NULL, gray_target_l REAL NOT NULL
);
CREATE TABLE experiment_group (
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
);
CREATE TABLE image (
 id INTEGER PRIMARY KEY AUTOINCREMENT, relative_path TEXT NOT NULL UNIQUE, filename TEXT NOT NULL,
 width INTEGER NOT NULL, height INTEGER NOT NULL, file_size INTEGER NOT NULL,
 modified_time_ns INTEGER NOT NULL, group_id INTEGER REFERENCES experiment_group(id) ON DELETE SET NULL,
 status TEXT NOT NULL, low_quality INTEGER NOT NULL
);
CREATE TABLE annotation (
 image_id INTEGER PRIMARY KEY REFERENCES image(id) ON DELETE CASCADE,
 roi_json TEXT, left_edges_json TEXT, right_edges_json TEXT, exclusions_json TEXT,
 gray_card_json TEXT, split_left REAL NOT NULL, split_right REAL NOT NULL,
 white_depth_min REAL NOT NULL DEFAULT 0.0, white_depth_max REAL NOT NULL DEFAULT 25.0,
 black_depth_min REAL NOT NULL DEFAULT 60.0, black_depth_max REAL NOT NULL DEFAULT 100.0,
 updated_at TEXT NOT NULL
);
CREATE TABLE measurement (
 image_id INTEGER PRIMARY KEY REFERENCES image(id) ON DELETE CASCADE,
 parameter_hash TEXT NOT NULL, metrics_json TEXT NOT NULL, calculated_at TEXT NOT NULL
);
CREATE TABLE scan_issue (relative_path TEXT NOT NULL, message TEXT NOT NULL);
CREATE TABLE event_log (
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
 event_type TEXT NOT NULL, payload_json TEXT NOT NULL
);
"""
