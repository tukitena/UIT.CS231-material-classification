"""Dòng lệnh ``minc``. Chạy qua uv: ``uv run minc <lệnh>``.

uv run minc download [--force]
uv run minc check
uv run minc extract [--split train validation test]
uv run minc run configs/e5_hsv_bow_multi.yaml [--allow-retest]
uv run minc summarize [--out reports]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import REPO_ROOT, load_config, paths

BASE_CONFIG = REPO_ROOT / "configs" / "_base.yaml"


def main(argv: list[str] | None = None) -> None:
    # Console Windows (cp1252) không in được tiếng Việt khi output bị pipe/redirect.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="minc", description="HSV + BoW-HOG trên MINC-2500")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Tải dataset đúng revision ghim trong config")
    p_dl.add_argument("--force", action="store_true", help="Xóa bản local và tải lại")

    sub.add_parser("check", help="Kiểm tra bản local khớp config (số mẫu, lớp, revision)")

    p_ex = sub.add_parser("extract", help="Trích và cache HOG + HSV hai góc nhìn")
    p_ex.add_argument(
        "--split", nargs="+", default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )  # fmt: skip

    p_run = sub.add_parser("run", help="Chạy một config thí nghiệm (E1–E8)")
    p_run.add_argument("config", type=Path)
    p_run.add_argument(
        "--allow-retest", action="store_true",
        help="Cho phép đánh giá test lần nữa (bị ghi vào metadata)",
    )  # fmt: skip

    p_sum = sub.add_parser("summarize", help="Gộp kết quả thành bảng cho báo cáo")
    p_sum.add_argument("--out", type=Path, default=REPO_ROOT / "reports")

    args = parser.parse_args(argv)
    p = paths()

    if args.cmd in ("download", "check", "extract"):
        base = load_config(BASE_CONFIG)
        from . import data

        if args.cmd == "download":
            data.download(base["dataset"], p.dataset, force=args.force)
        elif args.cmd == "check":
            data.check_dataset(p.dataset, base["dataset"])
            print(f"OK: {p.dataset}")
        else:
            from .pipeline import extract

            extract(base, p, tuple(args.split))
    elif args.cmd == "run":
        from .pipeline import run

        run_dir = run(args.config, allow_retest=args.allow_retest)
        print(f"Xong: {run_dir}")
    elif args.cmd == "summarize":
        from .evaluation.report import write_all

        write_all(p.runs, args.out)


if __name__ == "__main__":
    main()
