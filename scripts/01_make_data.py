"""Stage 1 - generate the synthetic co-op demand dataset and its provenance manifest.

Deterministic: the same config seed always produces the same CSV content. The
manifest records two hashes per file: `content_sha256` (the uncompressed CSV
text, comparable across platforms) and `sha256` (the .gz file, which can differ
between zlib builds even for identical content). If a manifest already exists,
the content hashes are checked against it and MATCH / MISMATCH is reported
before it is overwritten.
"""
import hashlib
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
        content = hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
        manifest["files"][name] = {
            "content_sha256": content, "sha256": sha256_file(path),
            "rows": len(df), "coops": int(df.coop_id.nunique()),
        }
        print(f"wrote {path.relative_to(data_dir.parent)}  rows={len(df):,}  content_sha256={content[:16]}")
    write_json(manifest, manifest_path)
    write_json(card, data_dir / "data_card.json")

    if previous:
        key = "content_sha256" if all("content_sha256" in f for f in previous["files"].values()) else "sha256"
        same = all(previous["files"].get(k, {}).get(key) == v[key] for k, v in manifest["files"].items())
        print(f"Reproducibility check vs previous manifest ({key}):", "MATCH" if same else "MISMATCH")
    plots.data_overview(source, target, results_dir / "fig_data_overview.png", cfg["data"]["target_train_start"])
    print(f"config {config_hash(cfg)}  done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
