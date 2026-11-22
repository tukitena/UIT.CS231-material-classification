"""Đánh dấu hàm chưa cài đặt. Test gặp ``NotImplementedError`` sẽ được báo SKIPPED (xem
``tests/conftest.py``), nên ``uv run pytest`` vẫn xanh trong lúc các module đang được viết."""

from typing import NoReturn


def todo(owner: str, what: str) -> NoReturn:
    raise NotImplementedError(f"TODO [{owner}] {what}")
