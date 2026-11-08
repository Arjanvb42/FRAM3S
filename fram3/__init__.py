"""FRAM3: video/photos to 3D, edit, export."""
__version__ = "0.1.0"

import os as _os
from pathlib import Path as _Path

_cfg = _Path.home() / ".config/fram3/env"
if _cfg.exists():
    for _line in _cfg.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
