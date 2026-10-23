"""Single entry point: run every stage in order with one config.

    python scripts/run_all.py --config configs/ablation.yaml
    python scripts/run_all.py --config configs/smoke.yaml      # ~3 min CPU check
"""
import importlib
import sys
import time

import _bootstrap  # noqa: F401
from _bootstrap import parse_args

STAGES = ["01_make_data", "02_pretrain", "03_run_ablation", "04_analyze", "05_render_report"]


def main(argv=None):
    args = parse_args(__doc__, argv)
    t0 = time.perf_counter()
    for stage in STAGES:
        print(f"\n===== {stage} =====", flush=True)
        importlib.import_module(stage).main(["--config", args.config])
    print(f"\nall stages finished in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
