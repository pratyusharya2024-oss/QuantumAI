"""
main.py
=======
Central Entry Point — Automated Physics Modeler
Coordinates: DataCollector → Logic_Engine → Solver → Plotter → Animator
Author: [Your Name]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Core import Logic_Engine, DataCollector, Solver

# ─────────────────────────────────────────────
# MODULE REGISTRY — loads register() from each module
# ─────────────────────────────────────────────

def _load_module(physics_field: str, engine: Logic_Engine) -> None:
    """
    Dynamically loads the correct physics module and
    registers its formulas into the Logic_Engine.
    """
    module_map = {
        "kinematics":      "Modules.kinematics",
        "gravitation":     "Modules.gravitation",
        "energetics":      "Modules.energetics",
        "thermodynamics":  "Modules.thermodynamics",
        "quantum_mechanics": "Modules.quantum_mechanics",
    }

    module_path = module_map.get(physics_field)
    if not module_path:
        raise ValueError(f"Unknown physics field: '{physics_field}'")

    import importlib
    mod = importlib.import_module(module_path)

    if hasattr(mod, "register"):
        mod.register(engine)
        print(f"\n  ✅ Module loaded: {physics_field.upper()}")
    else:
        raise AttributeError(
            f"Module '{module_path}' has no register() function."
        )


# ─────────────────────────────────────────────
# FORMULA DISPLAY — prints formula, steps, answer
# ─────────────────────────────────────────────

def _display_result(
    physics_field: str,
    formula_name: str,
    formula_latex: str,
    known: dict,
    target: str,
    result: float,
    expression: str,
) -> None:
    """Prints a clean formatted result block to terminal."""
    print("\n" + "═" * 60)
    print(f"  📐 PHYSICS FIELD  : {physics_field.upper()}")
    print(f"  📋 FORMULA USED   : {formula_name}")
    print("═" * 60)
    print(f"  Symbolic Form  : {formula_latex}")
    print(f"  Expression     : {expression}")
    print("─" * 60)
    print("  KNOWN VALUES:")
    for sym, val in known.items():
        print(f"    {sym} = {val}")
    print("─" * 60)
    print(f"  ✅ {target} = {result:.6f}")
    print("═" * 60)


# ─────────────────────────────────────────────
# LOGGER — writes to Data/logs.csv
# ─────────────────────────────────────────────

def _log_result(
    physics_field: str,
    formula_name: str,
    known: dict,
    target: str,
    result: float,
    status: str = "success",
    notes: str = "",
) -> None:
    """Appends a result row to Data/logs.csv."""
    import csv
    from datetime import datetime

    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Data", "logs.csv"
    )

    inputs_str = "  ".join(f"{k}={v}" for k, v in known.items())
    result_str = f"{target}={result:.6f}"
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [timestamp, physics_field, formula_name,
           inputs_str, result_str, status, notes]

    try:
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"\n  📁 Result logged to Data/logs.csv")
    except Exception as e:
        print(f"\n  ⚠️  Could not write to logs.csv: {e}")


# ─────────────────────────────────────────────
# PLOT TRIGGER
# ─────────────────────────────────────────────

def _trigger_plot(physics_field: str, result_data: dict) -> None:
    """Calls the correct plotter function based on physics field."""
    try:
        from Visualization.plotter import (
            plot_projectile_trajectory,
            plot_1d_motion,
            plot_orbital_path,
            plot_gravitational_field,
            plot_shm_energy,
            plot_projectile_energy,
            plot_radioactive_decay,
            plot_hydrogen_energy_levels,
            plot_carnot_cycle,
            plot_pv_diagram,
        )

        df = result_data.get("dataframe")

        print("\n  📊 Generating graph...")

        if physics_field == "kinematics":
            if df is not None:
                plot_1d_motion(df)
            else:
                print("  ⚠️  No simulation data for plot.")

        elif physics_field == "gravitation":
            if df is not None:
                plot_orbital_path(df)
            else:
                print("  ⚠️  No simulation data for plot.")

        elif physics_field == "energetics":
            if df is not None:
                plot_shm_energy(df)
            else:
                print("  ⚠️  No simulation data for plot.")

        elif physics_field == "quantum_mechanics":
            if df is not None:
                plot_radioactive_decay(df)
            else:
                print("  ⚠️  No simulation data for plot.")

        elif physics_field == "thermodynamics":
            print("  ℹ️  For thermodynamics, use plot_carnot_cycle() or plot_pv_diagram() directly.")

    except Exception as e:
        print(f"\n  ⚠️  Could not generate plot: {e}")


# ─────────────────────────────────────────────
# ANIMATION TRIGGER
# ─────────────────────────────────────────────

def _trigger_animation(
    physics_field: str,
    formula_latex: str,
    steps: list,
    answer: str,
    known: dict,
) -> None:
    """Asks user if they want animation, then launches it."""
    print("\n" + "─" * 60)
    choice = input("  🎬 Want to see an animation? (yes/no): ").strip().lower()
    if choice not in ("yes", "y"):
        print("  Skipping animation.")
        return

    try:
        from Visualization.animations import (
            animate_projectile,
            animate_1d_motion,
            animate_orbit,
            animate_shm,
            animate_decay,
            animate_bohr,
            animate_carnot,
        )

        print("\n  🎬 Launching animation...")

        if physics_field == "kinematics":
            v = known.get("v", known.get("u", 20))
            angle = known.get("theta", 45)
            animate_projectile(v, angle, formula_latex, steps, answer)

        elif physics_field == "gravitation":
            M = known.get("M", 5.972e24)
            r = known.get("r", 6.771e6)
            animate_orbit(M, r, formula_latex, steps, answer)

        elif physics_field == "energetics":
            k = known.get("k", 100)
            A = known.get("x", 0.5)
            m = known.get("m", 1.0)
            animate_shm(k, A, m, formula_latex, steps, answer)

        elif physics_field == "quantum_mechanics":
            N0      = known.get("N", 1000)
            t_half  = known.get("t", 5)
            total_t = t_half * 6
            animate_decay(N0, t_half, total_t, formula_latex, steps, answer)

        elif physics_field == "thermodynamics":
            T_hot  = known.get("T1", 500)
            T_cold = known.get("T2", 300)
            V1     = known.get("V", 0.001)
            V2     = V1 * 3
            animate_carnot(T_hot, T_cold, V1, V2,
                           formula=formula_latex,
                           steps=steps, answer=answer)

    except Exception as e:
        print(f"\n  ⚠️  Animation failed: {e}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run() -> None:
    """
    Full pipeline:
    DataCollector → Logic_Engine → Solver → Display → Log → Plot → Animate
    """
    print("\n" + "═" * 60)
    print("   ⚛️   AUTOMATED PHYSICS MODELER   ⚛️")
    print("═" * 60)

    # ── Step 1: Collect problem
    collector = DataCollector()
    try:
        problem_data = collector.collect()
    except ValueError as e:
        print(f"\n  ❌ Input error: {e}")
        return

    physics_field = problem_data["physics_field"]
    known         = problem_data["known"]
    target        = problem_data["target"]

    # ── Step 2: Load module + register formulas
    engine = Logic_Engine()
    try:
        _load_module(physics_field, engine)
    except (ValueError, AttributeError) as e:
        print(f"\n  ❌ Module error: {e}")
        return

    # ── Step 3: Select best formula
    try:
        formula_name = engine.select_formula(set(known.keys()), target)
        formula_eq   = engine.get_formula(formula_name)
        print(f"\n  🔍 Selected formula: {formula_name}")
        print(f"     {formula_eq}")
    except ValueError as e:
        print(f"\n  ❌ Formula error: {e}")
        _log_result(physics_field, "unknown", known,
                    target, 0.0, "error", str(e))
        return

    # ── Step 4: Rearrange for target
    from sympy import symbols
    target_sym = engine.get_symbol(target)
    try:
        solutions  = engine.rearrange(formula_name, target_sym)
        expression = solutions[0]
        formula_latex = engine.to_latex(formula_eq)
        print(f"\n  🔄 Rearranged: {target} = {expression}")
    except ValueError as e:
        print(f"\n  ❌ Rearrangement error: {e}")
        _log_result(physics_field, formula_name, known,
                    target, 0.0, "error", str(e))
        return

    # ── Step 5: Solve numerically
    solver = Solver(engine)
    try:
        result_data = solver.solve(
            expression = expression,
            known      = known,
            target     = target,
            time_steps = 500,
        )
    except (ValueError, RuntimeError) as e:
        print(f"\n  ❌ Solver error: {e}")
        _log_result(physics_field, formula_name, known,
                    target, 0.0, "error", str(e))
        return

    result = result_data["result"]

    # ── Step 6: Build steps for display + animation
    steps = [
        f"Field      : {physics_field}",
        f"Formula    : {formula_name}",
        f"Equation   : {formula_eq}",
        f"Rearranged : {target} = {expression}",
        f"Known      : {known}",
        f"Result     : {target} = {result:.6f}",
    ]
    answer = f"{target} = {result:.6f}"

    # ── Step 7: Display result
    _display_result(
        physics_field = physics_field,
        formula_name  = formula_name,
        formula_latex = formula_latex,
        known         = known,
        target        = target,
        result        = result,
        expression    = str(expression),
    )

    # ── Step 8: Log to CSV
    _log_result(
        physics_field = physics_field,
        formula_name  = formula_name,
        known         = known,
        target        = target,
        result        = result,
        status        = "success",
    )

    # ── Step 9: Plot
    _trigger_plot(physics_field, result_data)

    # ── Step 10: Animation
    _trigger_animation(
        physics_field = physics_field,
        formula_latex = formula_latex,
        steps         = steps,
        answer        = answer,
        known         = known,
    )

    print("\n  ✅ Session complete.\n")


# ─────────────────────────────────────────────
# REPEAT LOOP — run multiple problems
# ─────────────────────────────────────────────

if __name__ == "__main__":
    while True:
        run()
        print("\n" + "─" * 60)
        again = input("  🔁 Solve another problem? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  👋 Exiting Physics Modeler. Goodbye!\n")
            break