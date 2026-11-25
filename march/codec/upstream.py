"""Load the pinned NextHAM sources without editing their absolute imports."""
from pathlib import Path
import sys

def activateNextHam():
    root = Path(__file__).resolve().parents[2] / "thirdParty" / "nextHam"
    if not (root / "tg_src" / "e3modules.py").is_file():
        raise RuntimeError(f"Missing vendored NextHAM source: {root}")
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
    for name in ("nets", "tg_src"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            locations = list(getattr(loaded, "__path__", []))
            if locations and not all(Path(p).resolve().is_relative_to(root) for p in locations):
                raise RuntimeError(f"{name} was imported from outside the pinned NextHAM copy")
    return root
