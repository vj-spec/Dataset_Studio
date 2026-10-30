"""Small, user-profile state files used to restore the studio after restart."""
from __future__ import annotations

import json
import os
from config import app_state_file


def load_state(name: str, default=None):
    path = app_state_file(name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {} if default is None else default


def save_state(name: str, data):
    path = app_state_file(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def clear_saved_state():
    """Clear app preferences/session snapshots, never dataset or model files."""
    for name in ("session.json", "video_annotator.json", "cameras.json", "last_detection.png"):
        for suffix in ("", ".tmp"):
            try:
                app_state_file(name + suffix).unlink(missing_ok=True)
            except OSError:
                pass
