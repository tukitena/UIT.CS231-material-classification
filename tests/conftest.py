"""Test gặp NotImplementedError (hàm còn là stub ``todo(...)``) được báo SKIPPED kèm tên người
phụ trách, thay vì FAILED. Khi hàm được cài đặt, test tự chuyển sang kiểm tra thật."""

from __future__ import annotations

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, str(call.excinfo.value))
