"""
Core/__init__.py
================
Package initializer for the Core engine.
Exposes the three central engines cleanly so any module
can import directly from Core instead of deep paths.

Usage:
    from Core import Logic_Engine
    from Core import DataCollector
    from Core import Solver
    from Core import Logic_Engine, DataCollector, Solver  # all at once
"""

from Core.logic_engine import Logic_Engine
from Core.data_collector import DataCollector
from Core.solver import Solver

__all__ = [
    "Logic_Engine",
    "DataCollector",
    "Solver",
]