#!/usr/bin/env python
"""Command-line entry point.

    python run_pipeline.py                    # all stages
    python run_pipeline.py --stages 1 2 3     # selected stages
    python run_pipeline.py --stages 5 --config my_config.yaml
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

# conda numpy/MKL (libiomp5md) and pip torch (libomp) both ship an OpenMP runtime on Windows; without this
# the process aborts with "OMP: Error #15" as soon as both are loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sjogren_pipeline.context import load_context
from sjogren_pipeline.utils import set_seed

STAGES = {
    1: "stage01_download",
    2: "stage02_preprocess",
    3: "stage03_diffexpr",
    4: "stage04_pathways",
    5: "stage05_models",
    6: "stage06_explain",
    7: "stage07_ranking",
    8: "stage08_validation",
    9: "stage09_figures",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml)")
    ap.add_argument("--stages", nargs="*", type=int, default=sorted(STAGES), help="stage numbers 1-9")
    args = ap.parse_args()

    ctx = load_context(args.config)
    set_seed(ctx.seed)
    for n in args.stages:
        name = STAGES[n]
        ctx.log.info("=" * 20 + f" {name} " + "=" * 20)
        t0 = time.time()
        importlib.import_module(f"sjogren_pipeline.{name}").run(ctx)
        ctx.log.info("%s finished in %.1fs", name, time.time() - t0)


if __name__ == "__main__":
    main()
