from typing import Optional

import numpy as np
import pandas as pd
from sympy import lambdify, Symbol
from sympy.core.expr import Expr

from Core.logic_engine import Logic_Engine


class Solver:
    """
    Numerical computation engine.

    Receives a symbolic expression from LogicEngine and known values
    from DataCollector (via main.py), substitutes them numerically
    using NumPy, and returns a structured result.

    Flow:
        1. _receive_expression()  — validates and stores the symbolic expression
        2. _receive_known_values() — validates and stores the known variable map
        3. _compute()             — numerically evaluates and builds the result
    """

    def __init__(self, engine: Logic_Engine) -> None:
        self._engine = engine
        self._expression: Optional[Expr] = None
        self._known: Optional[dict[str, float]] = None
        self._target: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        expression: Expr,
        known: dict[str, float],
        target: str,
        time_steps: Optional[int] = None,
    ) -> dict:
        """
        Orchestrates the three-step solving pipeline.

        Args:
            expression:  Rearranged symbolic expression from LogicEngine.
            known:       Dict of {symbol_name: numeric_value} from DataCollector.
            target:      Name of the variable being solved for.
            time_steps:  If provided, generates a time-series DataFrame
                         over [0, t_max] with this many steps (requires 't' in known).

        Returns:
            {
                "target":      str,
                "result":      float,
                "expression":  str,
                "dataframe":   pd.DataFrame or None,
            }
        """
        self._receive_expression(expression, target)
        self._receive_known_values(known)
        return self._compute(time_steps)

    # ------------------------------------------------------------------
    # Step 1 — Receive and validate the symbolic expression
    # ------------------------------------------------------------------

    def _receive_expression(self, expression: Expr, target: str) -> None:
        if expression is None:
            raise ValueError("Solver received a None expression from LogicEngine.")

        free_symbols = {str(s) for s in expression.free_symbols}
        if target in free_symbols:
            raise ValueError(
                f"Target '{target}' still appears in the expression — "
                "LogicEngine did not rearrange it fully."
            )

        self._expression = expression
        self._target = target

    # ------------------------------------------------------------------
    # Step 2 — Receive and validate the known values
    # ------------------------------------------------------------------

    def _receive_known_values(self, known: dict[str, float]) -> None:
        if not known:
            raise ValueError("Solver received an empty known-values dictionary.")

        if self._expression is None:
            raise RuntimeError("Expression was not received before validating known values.")

        required_symbols = {str(s) for s in self._expression.free_symbols}
        provided_symbols = set(str(k) for k in known.keys()) 
        missing = required_symbols - provided_symbols

        if missing:
            raise ValueError(
                f"Missing symbols: {missing}. "
                f"Provided: {provided_symbols}, Required: {required_symbols}"
            )

        self._known = {str(k): v for k, v in known.items()}  

    # ------------------------------------------------------------------
    # Step 3 — Compute the numerical result
    # ------------------------------------------------------------------

    def _compute(self, time_steps: Optional[int]) -> dict:
        if self._expression is None:
            raise RuntimeError("Expression was not received before compute was called.")
        if self._known is None:
            raise RuntimeError("Known values were not received before compute was called.")
        
        free_symbols = [Symbol(str(s)) for s in self._expression.free_symbols]
        symbol_names = [str(s) for s in free_symbols]       
        ordered_values = [self._known[name] for name in symbol_names]

        numeric_fn = lambdify(free_symbols, self._expression, modules=["numpy"])

        try:
            raw_result = numeric_fn(*ordered_values)
            result = float(np.squeeze(raw_result))
        except Exception as exc:
            raise RuntimeError(f"Numerical evaluation failed: {exc}") from exc

        dataframe = None
        if time_steps is not None:
            dataframe = self._build_time_series(numeric_fn, free_symbols, time_steps)

        self._print_result(result)

        return {
            "target": self._target,
            "result": result,
            "expression": str(self._expression),
            "dataframe": dataframe,
        }

    # ------------------------------------------------------------------
    # Time-series DataFrame (feeds into plotter.py)
    # ------------------------------------------------------------------

    def _build_time_series(
        self,
        numeric_fn,
        free_symbols: list[Symbol],
        time_steps: int,
    ) -> pd.DataFrame:
        
        if self._known is None or "t" not in self._known:
            raise ValueError(
                "Time-series generation requires 't' in known values as the upper bound."
            )

        t_max = self._known["t"]
        t_array = np.linspace(0, t_max, time_steps)

        symbol_names = [str(s) for s in free_symbols]
        values = []
        for name in symbol_names:
            if name == "t":
                values.append(t_array)
            else:
                values.append(np.full_like(t_array, self._known[name]))

        try:
            result_array = numeric_fn(*values)
        except Exception as exc:
            raise RuntimeError(f"Time-series computation failed: {exc}") from exc

        return pd.DataFrame({
            "t": t_array,
            self._target: np.squeeze(result_array),
        })

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    def _print_result(self, result: float) -> None:
        print("\n" + "=" * 60)
        print("  Solver Result")
        print("=" * 60)
        print(f"  Expression : {self._expression}")
        print(f"  Known      : {self._known}")
        print(f"  {self._target} = {result:.4f}")
        print("=" * 60)