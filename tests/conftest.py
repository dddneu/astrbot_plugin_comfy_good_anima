"""Pytest 配置：共享 fixtures 和全局 DEBUG 日志设置。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# 开启所有第三方库的 DEBUG 日志，方便排查网络问题
_DEBUG_LOGGERS = [
    "httpx",
    "httpcore",
    "httpcore2",
    "openai",
    "openai._base_client",
    "openai.resources",
    "anima_agent",
]


def pytest_configure(config) -> None:
    for name in _DEBUG_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)
    # 同时开启 root logger 确保日志能输出
    logging.getLogger().setLevel(logging.DEBUG)
