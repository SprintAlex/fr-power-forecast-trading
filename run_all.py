"""End-to-end pipeline runner.

  python run_all.py                 # synthetic, walk-forward, full report
  python run_all.py --single        # single-fit instead of walk-forward
  python run_all.py --real          # use real ENTSO-E parquet (needs token+pull)

Steps: build data -> forecast (LEAR/LGBM quantile + conformal) -> battery LP
verdict -> FR-DE spread verdict -> dashboard figure.
"""
from __future__ import annotations

import sys

from src.data import synthetic
from src.eval import dashboard, strategy_comparison
from src.forecast import train
from src.trading import spread_strategy

REAL = "--real" in sys.argv
WALK = "--single" not in sys.argv


def main():
    if not REAL:
        print("== 1/5 build synthetic FR + DE ==")
        synthetic.build("FR")
        synthetic.build("DE")

    print("\n== 2/5 forecast (LEAR + LightGBM quantile + conformal) ==")
    train.run(synthetic=not REAL, walkforward=WALK)

    print("\n== 3/5 battery arbitrage — desk verdict ==")
    strategy_comparison.run()

    if not REAL:                     # spread needs the DE zone (synthetic only for now)
        print("\n== 4/5 FR-DE location spread — sign accuracy & P&L ==")
        spread_strategy.run()

    print("\n== 5/5 dashboard ==")
    dashboard.run()
    print("\nDone. See results/dashboard.png")


if __name__ == "__main__":
    main()
