"""Unit tests for voxlib.log."""

from __future__ import annotations

import logging

from voxlib.log import LOGGER_NAME, configure_logging, get_logger


def test_get_logger_root():
    """get_logger() with no name returns the voxlib root logger."""
    logger = get_logger()
    assert logger.name == LOGGER_NAME


def test_get_logger_child():
    """get_logger('x') returns a child under the voxlib namespace."""
    logger = get_logger("x")
    assert logger.name == f"{LOGGER_NAME}.x"


def test_configure_logging_is_idempotent():
    """Calling configure_logging twice does not duplicate handlers."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()

    configure_logging(logging.DEBUG)
    configure_logging(logging.DEBUG)

    stream_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1
    assert logger.level == logging.DEBUG

    logger.handlers.clear()
