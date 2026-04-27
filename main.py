import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.logic_engine import Logic_Engine
from Core.data_collector import DataCollector
from Core.solver import Solver
from Visualization.plotter import Plotter
from Visualization.animator import Animator
from Modules import kinematics, gravitation, energetics


MODULE_REGISTRY = {
    "kinematics":        kinematics,
    "gravitation":       gravitation,
    "energetics":        energetics,
}

TIME_STEPS = 500


class PhysicsModeler:
    """
    Application entry point and pipeline orchestrator.

    Pipeline:
        DataCollector → LogicEngine → Solver → Terminal Output
                                             → Plotter   (optional)
                                             → Animator  (optional)
    """

    def __init__(self) -> None:
        self._engine    = Logic_Engine()
        self._collector = DataCollector()
        self._solver    = Solver(self._engine)
        self._plotter   = Plotter()
        self._animator  = Animator()

    # ------------------------------------------------------------------
    # Application loop
    # ------------------------------------------------------------------

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

            if not self._ask_yes_no("\n  Solve another problem? (y/n): "):
                self._print_exit()
                break

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        # Stage 1 — DataCollector gathers known values, target, physics field
        data = self._collect_problem()

        # Stage 2 — Load matching module → pushes formulas into LogicEngine
        self._load_module(data["physics_field"])

        # Stage 3 — LogicEngine selects and rearranges the correct formula
        expression = self._resolve_formula(data)

        # Stage 4 — Solver receives expression + known values → computes answer
        result = self._compute(expression, data)

        # Stage 5 — Print the answer in terminal
        self._display_result(result)

        # Stage 6 — Ask graph → ask animation
        self._handle_visualizations(expression, data)

    def _collect_problem(self) -> dict:
        """Delegates to DataCollector to parse the word problem from terminal."""
        data = self._collector.collect()
        if not data.get("known"):
            raise ValueError("No known values were extracted from the problem.")
        if not data.get("target"):
            raise ValueError("Could not determine the target variable.")
        return data

    def _load_module(self, physics_field: str) -> None:
        """
        Looks up the matching module from the registry and calls register(engine),
        which pushes all its formulas into LogicEngine.
        """
        if physics_field not in MODULE_REGISTRY:
            raise KeyError(
                f"No module found for field '{physics_field}'. "
                f"Available: {list(MODULE_REGISTRY.keys())}"
            )
        MODULE_REGISTRY[physics_field].register(self._engine)
        print(f"\n  Module loaded : {physics_field.upper()}")

    def _resolve_formula(self, data: dict):
        """
        Asks LogicEngine to:
          1. Select the right formula based on known symbols + target
          2. Rearrange it to isolate the target variable
        Returns the rearranged symbolic expression.
        """
        known_symbols = set(data["known"].keys())
        target        = data["target"]

        formula_name   = self._engine.select_formula(known_symbols, target)
        target_symbol  = self._engine.get_symbol(target)
        solutions      = self._engine.rearrange(formula_name, target_symbol)

        if not solutions:
            raise ValueError(
                f"LogicEngine could not rearrange '{formula_name}' for '{target}'."
            )

        print(f"  Formula       : {formula_name}")
        print(f"  Rearranged    : {target} = {solutions[0]}")
        return solutions[0]

    def _compute(self, expression, data: dict) -> dict:
        """Single-point solve — passes expression + known values to Solver."""
        return self._solver.solve(
            expression=expression,
            known=data["known"],
            target=data["target"],
        )

    def _compute_time_series(self, expression, data: dict) -> dict:
        """
        Re-runs Solver with time_steps to produce a DataFrame for visualizations.
        Requires 't' to be present in known values.
        """
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
        print(f"  Expression used : {result['expression']}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Visualization flow
    # ------------------------------------------------------------------

    def _handle_visualizations(self, expression, data: dict) -> None:
        """
        Asks the user sequentially:
          1. Graph?    → plotter.py
          2. Animation? → animator.py
        Both share the same time-series DataFrame so Solver runs only once.
        """
        if not self._ask_yes_no("  Would you like to see a graph? (y/n): "):
            return

        try:
            result_series = self._compute_time_series(expression, data)
        except ValueError as exc:
            print(f"\n  [Visualization skipped] {exc}\n")
            return

        dataframe = result_series["dataframe"]
        target    = data["target"]

        # Send to plotter
        self._plotter.plot(dataframe=dataframe, target=target)

        # Ask animation only after graph is confirmed
        if self._ask_yes_no("  Would you like to see an animation? (y/n): "):
            self._animator.animate(
                dataframe=dataframe,
                target=target,
                known=data["known"],
            )

    # ------------------------------------------------------------------
    # Terminal UI helpers
    # ------------------------------------------------------------------

    def _print_welcome(self) -> None:
        print("\n" + "=" * 60)
        print("   Automated Physics Modeler")
        print("   Computational Research Engine v1.0")
        print("=" * 60)
        print("  Supported fields : kinematics, gravitation,")
        print("                     energetics, thermodynamics,")
        print("                     quantum_mechanics")
        print("=" * 60 + "\n")

    def _print_exit(self) -> None:
        print("\n  Exiting Physics Modeler. Goodbye!\n")

    def _ask_yes_no(self, prompt: str) -> bool:
        """Reusable y/n prompt — returns True for 'y', False for anything else."""
        return input(prompt).strip().lower() == "y"


if __name__ == "__main__":
    app = PhysicsModeler()
    app.run()