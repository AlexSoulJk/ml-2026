#!/usr/bin/env python3
"""Полный консольный отчёт по метрикам из готового ML run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag_nav.ml.metrics import print_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Путь к каталогу эксперимента")
    parser.add_argument(
        "--bins",
        type=int,
        default=6,
        metavar="N",
        help="Число бинов расстояния (по умолчанию 6)",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"Каталог не найден: {args.run_dir}")

    print_report(args.run_dir, n_bins=args.bins)


if __name__ == "__main__":
    main()
