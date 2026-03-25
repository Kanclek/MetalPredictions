from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _workspace_root() -> Path:
    # app/main.py -> repo_root
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    # Чтобы можно было запускать `python app/main.py ...` с любой CWD
    repo_root = _workspace_root()
    sys.path.insert(0, str(repo_root))

    from app.inference.pipeline import build_pipeline  # local import after sys.path tweak

    parser = argparse.ArgumentParser(description="NN inference for metal prediction")
    parser.add_argument("--params", required=True, help="Path to params Excel")
    parser.add_argument("--target", required=True, help="Path to target Excel")
    parser.add_argument("--nz", type=float, default=None, help="Optional filter by `nz` (default: None -> no filter)")
    parser.add_argument(
        "--weights",
        default=str(repo_root / "app" / "inference" / "model" / "repository" / "model2_weights.pth"),
        help="Path to model2_weights.pth",
    )
    parser.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    parser.add_argument("--out", default=None, help="Optional output CSV path")

    args = parser.parse_args(argv)

    pipe = build_pipeline(
        weights_path=args.weights,
        device=args.device,
        nz_value=args.nz,
    )

    df = pipe.predict(params_path=args.params, target_path=args.target, nz_value=args.nz)

    # минимальный человекочитаемый вывод
    cols = ["nz", "pred_class", "pred_proba_class_1"] if "pred_proba_class_1" in df.columns else ["pred_class"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(20).to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

