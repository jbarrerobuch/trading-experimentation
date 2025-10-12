"""
GammaNeutral - A gamma neutral trading strategy for crypto markets
using options and perpetual futures.
"""

__version__ = "0.1.0"
__author__ = "jbarrerobuch"

from .core.greeks import OptionsGreeksCalculator
from .core.hedging import PerpetualFuturesHedger
from .strategies.gamma_neutral import GammaNeutralStrategy

__all__ = [
    "OptionsGreeksCalculator",
    "PerpetualFuturesHedger",
    "GammaNeutralStrategy",
]
