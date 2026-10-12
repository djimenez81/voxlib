"""Logging configuration for voxlib.

Library code never calls :func:`logging.basicConfig` itself; that is
the application's job. What this module provides is a helper that
applications (including our own tests and future CLI) can call to get
a sensible default handler on the ``voxlib`` logger without touching
the root logger.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "voxlib"

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``voxlib`` namespace.

    Args:
        name: Optional sub-logger name. If given, it is appended to
            the ``voxlib`` prefix with a dot. If ``None``, the root
            ``voxlib`` logger is returned.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a default stream handler to the ``voxlib`` logger.

    This is a convenience for scripts, tests, and the future CLI. It
    is idempotent: calling it twice does not duplicate handlers. It
    never modifies the root logger, so applications that configure
    logging themselves are unaffected.

    Args:
        level: Minimum level for the ``voxlib`` logger. Accepts either
            a numeric level or its string name (case-insensitive).
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
    )
    logger.addHandler(handler)
    logger.propagate = False
