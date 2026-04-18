from sympy import symbols

from Core.logic_engine import Logic_Engine
from Core.data_collector import DataCollector
from Core.solver import Solver
from Visualization.animations import Animator
from Visualization.plotter import Plotter
import Modules


MODULE_REGISTRY: dict = {
    "kinematics":        Modules.kinematics,
    "gravitation":       Modules.gravitation,
    "energetics":        Modules.energetics,
    "thermodynamics":    Modules.thermodynamics,
    "quantum_mechanics": Modules.quantum_mechanics,
}

TIME_STEPS = 500


class PhysicsModeler:
    """
    Application entry point and pipeline orchestrator.

    Coordinates the full pipeline:
        DataCollector → LogicEngine → Solver → Terminal Output
                                             → Plotter (optional)
                                             → Animator (optional)
    """

    def __init__(self) -> None:
        self._engine    = Logic_Engine()
        self._collector = DataCollector()
        self._solver    = Solver(self._engine)
        self._plotter   = Plotter()
        self._animator  = Animator()

    def run(self) -> None:
        self._print_welcome()
        while True:
            try:
                self._run_pipeline()
            except KeyboardInterrupt:
                self._print_exit()
                break
            except (ValueError, KeyError, RuntimeError) as exc:
                print(f"\n  [Error] {exc}")
                print("  Please try again with a different problem.\n")

            if not self._ask_continue():
                self._print_exit()
                break

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        # Stage 1 — Collect problem data from the user
        data = self._collect_problem()

        # Stage 2 — Register the relevant physics module into the engine
        self._load_physics_module(data["physics_field"])

        # Stage 3 — Pull the rearranged formula from the engine
        expression = self._resolve_formula(data)

        # Stage 4 — Push expression + known values into the solver
        result = self._compute_result(expression, data)

        # Stage 5 — Display the final answer
        self._display_result(result)

        # Stage 6 — Optionally plot a graph
        if self._ask_graph():
            result_with_series = self._compute_with_time_series(expression, data)
            self._plotter.plot(
                dataframe=result_with_series["dataframe"],
                target=data["target"],
            )

            # Stage 7 — Optionally animate
            if self._ask_animation():
                self._animator.animate(
                    dataframe=result_with_series["dataframe"],
                    target=data["target"],
                    known=data["known"],
                )

    def _collect_problem(self) -> dict:
        data = self._collector.collect()
        if not data.get("known"):
            raise ValueError("No known values were extracted from the problem.")
        if not data.get("target"):
            raise ValueError("Could not determine the target variable.")
        return data

    def _load_physics_module(self, physics_field: str) -> None:
        if physics_field not in MODULE_REGISTRY:
            raise KeyError(
                f"Physics field '{physics_field}' has no registered module. "
                f"Available: {list(MODULE_REGISTRY.keys())}"
            )
        MODULE_REGISTRY[physics_field].register(self._engine)
        print(f"\n  Module loaded: {physics_field.upper()}")

    def _resolve_formula(self, data: dict):
        known_symbols = set(data["known"].keys())
        target = data["target"]

        formula_name = self._engine.select_formula(known_symbols, target)
        print(f"  Formula selected: {formula_name}")

        target_symbol = self._engine.get_symbol(target)
        solutions = self._engine.rearrange(formula_name, target_symbol)

        if not solutions:
            raise ValueError(
                f"LogicEngine could not rearrange '{formula_name}' for '{target}'."
            )

        print(f"  Rearranged expression: {target} = {solutions[0]}")
        return solutions[0]

    def _compute_result(self, expression, data: dict) -> dict:
        """Single-point solve — no time series."""
        return self._solver.solve(
            expression=expression,
            known=data["known"],
            target=data["target"],
        )

    def _compute_with_time_series(self, expression, data: dict) -> dict:
        """Re-runs the solver with time_steps to produce the DataFrame for visuals."""
        if "t" not in data["known"]:
            raise ValueError(
                "Cannot generate a graph — 't' (time) is not in the known values."
            )
        return self._solver.solve(
            expression=expression,
            known=data["known"],
            target=data["target"],
            time_steps=TIME_STEPS,
        )

    def _display_result(self, result: dict) -> None:
        print("\n" + "=" * 60)
        print("  FINAL ANSWER")
        print("=" * 60)
        print(f"  {result['target']} = {result['result']:.4f}")
        print(f"  (Expression used: {result['expression']})")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Terminal UI helpers
    # ------------------------------------------------------------------

    def _print_welcome(self) -> None:
        print("\n" + "=" * 60)
        print("   Automated Physics Modeler")
        print("   Computational Research Engine v1.0")
        print("=" * 60)
        print("  Supported fields: kinematics, gravitation,")
        print("  energetics, thermodynamics, quantum_mechanics")
        print("=" * 60 + "\n")

    def _print_exit(self) -> None:
        print("\n  Exiting Physics Modeler. Goodbye!\n")

    def _ask_continue(self) -> bool:
        answer = input("\n  Solve another problem? (y/n): ").strip().lower()
        return answer == "y"

    def _ask_graph(self) -> bool:
        answer = input("  Would you like to see a graph? (y/n): ").strip().lower()
        return answer == "y"

    def _ask_animation(self) -> bool:
        answer = input("  Would you like to see an animation? (y/n): ").strip().lower()
        return answer == "y"


if __name__ == "__main__":
    app = PhysicsModeler()
    app.run()