"""Logging helpers for the bathymetry package."""

from __future__ import annotations

import logging


def default_logger(name: str = "bathymetry") -> logging.Logger:
    """Create a configured logger when no handlers are attached.

    Parameters
    ----------
    name : str, default="bathymetry"
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(handler)
    return logger
