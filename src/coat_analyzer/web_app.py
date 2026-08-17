from __future__ import annotations

import io
import base64
import json
import mimetypes
import os
import subprocess
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .analysis import AnalysisOutput, analyze_image, load_rgb
from .exporter import export_project
from .project_store import (
    assign_group,
    create_group,
    create_project,
    get_image,
    image_path,
    list_groups,
    list_images,
    open_project,
    save_annotation,
    save_measurement,
    update_settings,
)


class AppState:
    def __init__(self) -> None:
        self.database_path: Path | None = None
        self.server: ThreadingHTTPServer | None = None


STATE = AppState()


class Handler(BaseHTTPRequestHandler):
    server_version = "SurgicalCoatAnalyzer/0.1"

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/project":
                self._json(None if STATE.database_path is None else _project_payload())
            elif parsed.path == "/api/images":
                self._require_project()
                self._json(list_images(STATE.database_path))
            elif parsed.path == "/api/groups":
                self._require_project()
                self._json(list_groups(STATE.database_path))
            elif parsed.path == "/api/image-meta":
                self._require_project()
                image_id = int(parse_qs(parsed.query)["id"][0])
                self._json(get_image(STATE.database_path, image_id))
            elif parsed.path == "/api/image-data":
                self._require_project()
                image_id = int(parse_qs(parsed.query)["id"][0])
                image = load_rgb(image_path(STATE.database_path, image_id))
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=94, subsampling=0)
                self._bytes(buffer.getvalue(), "image/jpeg")
            else:
                self._static(parsed.path)
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            data = self._read_json()
            if parsed.path == "/api/dialog/folder":
                self._json({"path": _choose_directory(data.get("initial", ""))})
            elif parsed.path == "/api/dialog/save-project":
                self._json({"path": _choose_project_file(True, data.get("initial", ""))})
            elif parsed.path == "/api/dialog/open-project":
                self._json({"path": _choose_project_file(False, data.get("initial", ""))})
            elif parsed.path == "/api/project/create":
                path = create_project(data.get("database_path") or None, data["name"], data["source_root"])
                STATE.database_path = path
                self._json(_project_payload(), HTTPStatus.CREATED)
            elif parsed.path == "/api/project/open":
                info = open_project(data["database_path"])
                STATE.database_path = Path(info["database_path"])
                self._json(_project_payload())
            elif parsed.path == "/api/analyze":
                self._require_project()
                image_id = int(data["image_id"])
                image = load_rgb(image_path(STATE.database_path, image_id))
                project = open_project(STATE.database_path)
                output = analyze_image(
                    image, data["roi"], data["left_edges"], data["right_edges"],
                    data.get("exclusions", []), data.get("gray_card", []),
                    float(data["split_left"]), float(data["split_right"]),
                    float(project["gray_target_l"]),
                    float(data.get("white_depth_min", 0.0)),
                    float(data.get("white_depth_max", 25.0)),
                    float(data.get("black_depth_min", 60.0)),
                    float(data.get("black_depth_max", 100.0)),
                )
                self._json({"metrics": output.metrics, "contours": output.contours,
                            "solver": output.solver, "parameter_hash": output.parameter_hash,
                            "classification_overlays": _classification_overlays(output)})
            elif parsed.path == "/api/annotation/save":
                self._require_project()
                image_id = int(data["image_id"])
                save_annotation(STATE.database_path, image_id, data, data.get("status"))
                if data.get("metrics") and data.get("parameter_hash"):
                    save_measurement(STATE.database_path, image_id, data["parameter_hash"], data["metrics"])
                self._json(get_image(STATE.database_path, image_id))
            elif parsed.path == "/api/groups/create":
                self._require_project()
                self._json(create_group(STATE.database_path, data["name"]), HTTPStatus.CREATED)
            elif parsed.path == "/api/groups/assign":
                self._require_project()
                assign_group(STATE.database_path, data.get("group_id"), data.get("image_ids", []))
                self._json({"ok": True})
            elif parsed.path == "/api/settings":
                self._require_project()
                self._json(update_settings(STATE.database_path, data))
            elif parsed.path == "/api/export":
                self._require_project()
                output = export_project(STATE.database_path, data.get("output_root") or None)
                self._json({"output_path": str(output)})
            elif parsed.path == "/api/shutdown":
                self._json({"ok": True})
                if STATE.server:
                    threading.Thread(target=STATE.server.shutdown, daemon=True).start()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._error(exc)

    def _project_payload(self) -> dict[str, object]:
        return _project_payload()

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _static(self, path: str) -> None:
        name = "index.html" if path in {"", "/"} else path.lstrip("/")
        if "/" in name or "\\" in name or ".." in name:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        resource = files("coat_analyzer.web").joinpath(name)
        if not resource.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._bytes(resource.read_bytes(), mimetypes.guess_type(name)[0] or "application/octet-stream")

    def _require_project(self) -> None:
        if STATE.database_path is None:
            raise RuntimeError("请先新建或打开项目")

    def _error(self, exc: Exception) -> None:
        status = HTTPStatus.BAD_REQUEST if isinstance(exc, (ValueError, KeyError, FileNotFoundError)) else HTTPStatus.INTERNAL_SERVER_ERROR
        self._json({"error": str(exc)}, status)

    def log_message(self, format: str, *args: object) -> None:
        return


def _project_payload() -> dict[str, object]:
    assert STATE.database_path is not None
    project = open_project(STATE.database_path)
    images = list_images(STATE.database_path)
    project["image_count"] = len(images)
    project["reviewed_count"] = sum(row["status"] == "reviewed" for row in images)
    return project


def _classification_overlays(output: AnalysisOutput, max_side: int = 1200) -> dict[str, str]:
    overlays: dict[str, str] = {}
    for mode, classes in output.classification_maps.items():
        height, width = classes.shape
        scale = min(1.0, max_side / max(width, height))
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        indexed = Image.fromarray(classes, mode="L")
        if target != (width, height):
            indexed = indexed.resize(target, Image.Resampling.NEAREST)
        values = np.asarray(indexed)
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[values == 1] = (65, 205, 255, 115)
        rgba[values == 2] = (132, 70, 255, 125)
        buffer = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buffer, "PNG", optimize=True)
        overlays[mode] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return overlays


def _choose_directory(initial: str) -> str:
    script = r"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$owner = New-Object System.Windows.Forms.Form
$owner.Text = 'Surgical Coat Analyzer'
$owner.StartPosition = 'CenterScreen'
$owner.Size = [System.Drawing.Size]::new(1, 1)
$owner.ShowInTaskbar = $false
$owner.TopMost = $true
$owner.Opacity = 0
$owner.Show()
$owner.Activate()
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select the image folder'
$dialog.ShowNewFolderButton = $true
$initial = [Environment]::GetEnvironmentVariable('COAT_DIALOG_INITIAL')
if ($initial -and (Test-Path -LiteralPath $initial)) { $dialog.SelectedPath = $initial }
$result = $dialog.ShowDialog($owner)
$owner.Close()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Write($dialog.SelectedPath)
}
"""
    return _run_windows_dialog(script, initial)


def _choose_project_file(save: bool, initial: str) -> str:
    dialog_type = "SaveFileDialog" if save else "OpenFileDialog"
    extra = "$dialog.DefaultExt = 'sqlite'; $dialog.AddExtension = $true" if save else "$dialog.CheckFileExists = $true"
    script = rf"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$owner = New-Object System.Windows.Forms.Form
$owner.Text = 'Surgical Coat Analyzer'
$owner.StartPosition = 'CenterScreen'
$owner.Size = [System.Drawing.Size]::new(1, 1)
$owner.ShowInTaskbar = $false
$owner.TopMost = $true
$owner.Opacity = 0
$owner.Show()
$owner.Activate()
$dialog = New-Object System.Windows.Forms.{dialog_type}
$dialog.Filter = 'SQLite project (*.sqlite)|*.sqlite|SQLite database (*.db)|*.db'
$dialog.Title = 'Select a Surgical Coat Analyzer project file'
{extra}
$initial = [Environment]::GetEnvironmentVariable('COAT_DIALOG_INITIAL')
if ($initial) {{
    if (Test-Path -LiteralPath $initial -PathType Container) {{ $dialog.InitialDirectory = $initial }}
    else {{
        $dialog.InitialDirectory = [System.IO.Path]::GetDirectoryName($initial)
        $dialog.FileName = [System.IO.Path]::GetFileName($initial)
    }}
}}
$result = $dialog.ShowDialog($owner)
$owner.Close()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::Write($dialog.FileName)
}}
"""
    return _run_windows_dialog(script, initial)


def _run_windows_dialog(script: str, initial: str) -> str:
    environment = os.environ.copy()
    environment["COAT_DIALOG_INITIAL"] = initial or ""
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Windows 文件选择窗口启动失败")
    return completed.stdout.strip()


def run() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    STATE.server = server
    url = f"http://127.0.0.1:{server.server_address[1]}"
    if os.environ.get("COAT_ANALYZER_NO_BROWSER") != "1":
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"Surgical Coat Analyzer 正在运行：{url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
