"""Shared CLI boilerplate: put the repo on sys.path and parse --config."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args(description: str, argv=None, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/ablation.yaml", help="YAML config path")
    if extra:
        extra(p)
    return p.parse_args(argv)
