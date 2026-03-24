"""Command-line entry point for the bathymetry package."""

from __future__ import annotations

import logging

from . import Bathymetry


def main() -> None:
    """Log the main public object for a minimal import sanity check."""

    logging.basicConfig(level=logging.INFO)
    logging.info("Bathymetry package entry point loaded: %s", Bathymetry)


if __name__ == "__main__":
    main()
