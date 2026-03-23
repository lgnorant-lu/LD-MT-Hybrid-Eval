from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_CLI_PATH = SRC / "local_poll_mqm" / "cli.py"
_SPEC = importlib.util.spec_from_file_location("local_poll_mqm.cli", _CLI_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load cli module from {_CLI_PATH}")

_CLI_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLI_MODULE)
main = _CLI_MODULE.main


if __name__ == "__main__":
    raise SystemExit(main())
