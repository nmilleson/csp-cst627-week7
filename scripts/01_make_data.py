"""Stage 1 - generate the synthetic co-op demand dataset and its provenance manifest.

Deterministic: the same config seed always produces byte-identical files. If a
manifest already exists, the new hashes are checked against it and the result
is reported (MATCH / MISMATCH) before it is overwritten.
"""
import json
import time

import _bootstrap  # noqa: F401
from _bootstrap import parse_args

from lora_ablation import data as gen
from lora_ablation import plots
from lora_ablation.utils import config_hash, load_config, resolve, sha256_file, write_json


def main(argv=None):
    args = parse_args(__doc__, argv)
    cfg = load_config(args.config)
    data_dir, results_dir = resolve(cfg, "data_dir"), resolve(cfg, "results_dir")
    t0 = time.perf_counter()
    source, target, card = gen.generate(cfg)

    files = {"source_coops.csv.gz": source, "target_coop.csv.gz": target}
    manifest_path = data_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest = {
        "generator": "lora_ablation/data.py (synthetic, no external download)",
        "generator_sha256": sha256_file(gen.__file__),
        "seed": cfg["data"]["seed"],
        "date_range": [cfg["data"]["start"], cfg["data"]["end"]],
        "frequency": "hourly",
        "units": {"demand_mw": "MW", "temperature_c": "deg C", "temperature_forecast_c": "deg C"},
        "files": {},
    }
    for name, df in files.items():
        path = data_dir / name
        df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        manifest["files"][name] = {
            "sha256": sha256_file(path), "rows": len(df), "coops": int(df.coop_id.nunique()),
        }
        print(f"wrote {path.relative_to(data_dir.parent)}  rows={len(df):,}  sha256={manifest['files'][name]['sha256'][:16]}")
    write_json(manifest, manifest_path)
    write_json(card, data_dir / "data_card.json")

    if previous:
        same = all(previous["files"].get(k, {}).get("sha256") == v["sha256"] for k, v in manifest["files"].items())
        print("Reproducibility check vs previous manifest:", "MATCH" if same else "MISMATCH")
    plots.data_overview(source, target, results_dir / "fig_data_overview.png", cfg["data"]["target_train_start"])
    print(f"config {config_hash(cfg)}  done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
