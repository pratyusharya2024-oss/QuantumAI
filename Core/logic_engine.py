from ast import Import
from typing import Optional
from sympy import im, symbols, Eq, solve, diff, integrate, simplify, latex, sympify, Symbol
from sympy import Function, Symbol
from sympy.core.expr import Expr

class Logic_Engine:

    """
    Central symbolic computation engine.
    Receives formula definitions from physics modules, rearranges and solves
    them for any target variable using SymPy.
    """

    def __init__(self) -> None:
        self._formula_registry: dict[str, Eq] = {}
        self._symbol_registry: dict[str, Symbol] = {}

    def register_formula(self, name: str, equation: Eq) -> None:
        if not isinstance(equation, Eq):
            raise TypeError(f"Expected sympy.Eq for '{name}', got {type(equation)}")
        self._formula_registry[name] = equation

    def register_symbol(self, name: str, symbol: Symbol) -> None:
        self._symbol_registry[name] = symbol

    def get_symbol(self, name: str) -> Symbol:
        if name not in self._symbol_registry:
            self._symbol_registry[name] = symbols(name)
        return self._symbol_registry[name]

    def get_formula(self, name: str) -> Eq:
        if name not in self._formula_registry:
            raise KeyError(f"Formula '{name}' is not registered.")
        return self._formula_registry[name]

    def list_formulas(self) -> list[str]:
        return list(self._formula_registry.keys())
    
    def select_formula(self, known_symbols: set[str], target: str) -> str:
        all_vars = known_symbols | {target}
        for name, equation in self._formula_registry.items():
            formula_symbols = {str(s) for s in equation.free_symbols}
            if formula_symbols.issubset(all_vars):
                return name
        raise ValueError(f"No registered formula matches variables: {all_vars}")

    # ------------------------------------------------------------------
    # Core symbolic operations
    # ------------------------------------------------------------------

    def rearrange(self, formula_name: str, target: Symbol) -> list[Expr]:

        """
        Rearrange a registered formula to isolate `target` on the left-hand side.
        Returns a list of symbolic solutions (may be multiple due to non-linearity).
        """

        equation = self.get_formula(formula_name)
        solutions = solve(equation, target)
        if not solutions:
            raise ValueError(
                f"Could not isolate '{target}' from formula '{formula_name}'."
            )
        return solutions

    def rearrange_expr(self, equation: Eq, target: Symbol) -> list[Expr]:
        solutions = solve(equation, target)
        if not solutions:
            raise ValueError(f"Could not isolate '{target}' from the given equation.")
        return solutions

    def substitute(
        self, formula_name: str, substitutions: dict[Symbol, float | Expr]
    ) -> Expr:
        equation = self.get_formula(formula_name)
        if not isinstance(equation, Eq):
            raise TypeError(f"Formula '{formula_name}' is not a SymPy Eq")

        lhs = equation.lhs.subs(substitutions)
        rhs = equation.rhs.subs(substitutions)

        if not isinstance(lhs, Expr) or not isinstance(rhs, Expr):
            raise TypeError("Substitution produced non-expression equation sides")

        result = simplify(lhs - rhs)

        free = result.free_symbols
        if len(free) == 1:
            (unknown,) = free
            solutions = solve(result, unknown)
            if not solutions:
                raise ValueError(
                    f"No solution found after substitution in '{formula_name}'."
                )
            return solutions[0]

        return simplify(rhs)

    def differentiate(
        self, formula_name: str, variable: Symbol, order: int = 1
    ) -> Expr:
        """
        Differentiate the RHS of a registered formula with respect to `variable`.

        Example: derive velocity v = ds/dt from a position expression s(t).
        """
        equation = self.get_formula(formula_name)
        result = equation.rhs
        for _ in range(order):
            result = diff(result, variable)
        return simplify(result)

    def integrate_formula(
        self,
        formula_name: str,
        variable: Symbol,
        limits: Optional[tuple[float, float]] = None,
    ) -> Expr:
        """
        Integrate the RHS of a registered formula with respect to `variable`.

        Pass `limits=(a, b)` for a definite integral, or omit for indefinite.
        """
        equation = self.get_formula(formula_name)
        expr = equation.rhs
        if limits is not None:
            result = integrate(expr, (variable, limits[0], limits[1]))
        else:
            result = integrate(expr, variable)
        return simplify(result)

    def derive_formula(self, expression: Expr, variable: Symbol, order: int = 1) -> Expr:
        """
        Differentiate an arbitrary expression (not from registry).
        Intended for inline use by module files.
        """
        result = expression
        for _ in range(order):
            result = diff(result, variable)
        return simplify(result)

    def simplify_expression(self, expression: Expr) -> Expr:
        return simplify(expression)

    def to_latex(self, expression: Expr) -> str:
        """Return the LaTeX representation of a symbolic expression."""
        return latex(expression)

    # ------------------------------------------------------------------
    # Utility: build an equation from a raw string (for REPL / notebook use)
    # ------------------------------------------------------------------

    def parse_equation(self, lhs_str: str, rhs_str: str) -> Eq:
        """
        Parse two string expressions into a SymPy Eq using registered symbols
        as the local namespace.

        Example:
            engine.parse_equation("v", "u + a*t")
        """
        namespace = dict(self._symbol_registry)
        try:
            lhs = sympify(lhs_str, locals=namespace)
            rhs = sympify(rhs_str, locals=namespace)
        except Exception as exc:
            raise ValueError(f"Failed to parse equation: {exc}") from exc
        return Eq(lhs, rhs)