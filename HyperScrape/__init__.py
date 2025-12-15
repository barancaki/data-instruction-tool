"""
HyperScrape Engine v2
A modular, high-performance fair scraping system.
"""

from .core.engine import HyperScrapeEngine
from .config.loader import load_fair_config, load_all_configs
from .models.fair import Company, FairConfig, ScrapeResult

__version__ = "2.0.0"
__all__ = [
    "HyperScrapeEngine",
    "load_fair_config",
    "load_all_configs",
    "Company",
    "FairConfig",
    "ScrapeResult",
]
