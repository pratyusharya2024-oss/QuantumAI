import re
from typing import Optional


PHYSICS_KEYWORDS: dict[str, list[str]] = {
    "kinematics": [
        "velocity", "acceleration", "displacement", "projectile",
        "motion", "speed", "distance", "time", "deceleration", "retardation"
    ],
    "gravitation": [
        "gravity", "gravitational", "orbit", "orbital", "planet",
        "satellite", "mass", "weight", "free fall", "escape velocity"
    ],
    "energetics": [
        "energy", "kinetic", "potential", "work", "power",
        "joule", "watt", "conservation", "momentum", "collision"
    ],
    "thermodynamics": [
        "temperature", "heat", "entropy", "pressure", "volume",
        "gas", "thermal", "boiling", "celsius", "kelvin", "carnot"
    ],
    "quantum_mechanics": [
        "quantum", "photon", "wavelength", "frequency", "electron",
        "planck", "wave", "particle", "uncertainty", "spin"
    ],
}

VARIABLE_PATTERNS: dict[str, list[str]] = {
    "u":     ["initial velocity", "initial speed"],
    "v":     ["final velocity", "final speed"],
    "a":     ["acceleration", "deceleration", "retardation"],
    "t":     ["time", "duration", "seconds", "second"],
    "s":     ["displacement", "distance", "travelled", "traveled"],
    "g":     ["gravity", "gravitational acceleration"],
    "m":     ["mass"],
    "F":     ["force"],
    "KE":    ["kinetic energy"],
    "PE":    ["potential energy"],
    "W":     ["work"],
    "P":     ["power"],
    "T":     ["temperature"],
    "h":     ["height", "altitude"],
    "r":     ["radius", "orbital radius"],
}


class DataCollector:

    def __init__(self) -> None:
        self._raw_problem: str = ""

    def collect(self) -> dict:
        self._raw_problem = self._prompt_user()
        known = self._extract_known_values()
        target = self._extract_target(known)
        physics_field = self._identify_physics_field()

        return {
            "raw_problem": self._raw_problem,
            "physics_field": physics_field,
            "known": known,
            "target": target,
        }

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _prompt_user(self) -> str:
        print("\n" + "=" * 60)
        print("  Automated Physics Modeler — Problem Input")
        print("=" * 60)
        print("Enter your physics word problem below.")
        print("Example: A car starts from rest and accelerates at")
        print("         3 m/s² for 10 seconds. Find the final velocity.")
        print("-" * 60)
        problem = input("Problem: ").strip()
        if not problem:
            raise ValueError("No problem was entered.")
        return problem.lower()

    # ------------------------------------------------------------------
    # Known value extraction
    # ------------------------------------------------------------------

    def _extract_known_values(self) -> dict[str, float]:
        known: dict[str, float] = {}
        numbers = self._extract_all_numbers()

        for symbol, phrases in VARIABLE_PATTERNS.items():
            for phrase in phrases:
                value = self._find_value_near_phrase(phrase, numbers)
                if value is not None and symbol not in known:
                    known[symbol] = value

        if not known:
            known = self._fallback_manual_input()

        return known

    def _extract_all_numbers(self) -> list[tuple[int, float]]:
        return [
            (m.start(), float(m.group()))
            for m in re.finditer(r"-?\d+\.?\d*", self._raw_problem)
        ]

    def _find_value_near_phrase(
        self, phrase: str, numbers: list[tuple[int, float]]
    ) -> Optional[float]:
        
        idx = self._raw_problem.find(phrase)
        if idx == -1:
            return None
        search_start = idx
        search_end = idx + len(phrase) + 60
        for pos, value in numbers:
            if search_start <= pos <= search_end:
                return value
        return None

    def _fallback_manual_input(self) -> dict[str, float]:
        print("\n  Could not extract values automatically.")
        print("  Please enter known values manually.")
        print("  Available symbols:", ", ".join(VARIABLE_PATTERNS.keys()))
        print("  Press Enter with no symbol to stop.\n")

        known: dict[str, float] = {}
        while True:
            symbol = input("  Symbol (e.g. u, a, t): ").strip()
            if not symbol:
                break
            if symbol not in VARIABLE_PATTERNS:
                print(f"  '{symbol}' is not a recognised symbol. Try again.")
                continue
            try:
                value = float(input(f"  Value for '{symbol}': ").strip())
                known[symbol] = value
            except ValueError:
                print("  Invalid number. Try again.")

        if not known:
            raise ValueError("No known values were provided.")
        return known

    # ------------------------------------------------------------------
    # Target variable extraction
    # ------------------------------------------------------------------

    def _extract_target(self, known: dict[str, float]) -> str:
        trigger_phrases = ["find", "calculate", "determine", "what is", "solve for"]
        for phrase in trigger_phrases:
            idx = self._raw_problem.find(phrase)
            if idx == -1:
                continue
            window = self._raw_problem[idx: idx + 80]
            for symbol, phrases in VARIABLE_PATTERNS.items():
                if symbol in known:
                    continue
                for var_phrase in phrases:
                    if var_phrase in window:
                        return symbol

        return self._ask_target(known)

    def _ask_target(self, known: dict[str, float]) -> str:
        unknown_symbols = [s for s in VARIABLE_PATTERNS if s not in known]
        print(f"\n  Could not detect the target variable automatically.")
        print(f"  Unknown symbols available: {', '.join(unknown_symbols)}")
        while True:
            target = input("  What are you solving for? ").strip()
            if target in VARIABLE_PATTERNS:
                return target
            print(f"  '{target}' not recognised. Choose from: {', '.join(unknown_symbols)}")

    # ------------------------------------------------------------------
    # Physics field identification
    # ------------------------------------------------------------------

    def _identify_physics_field(self) -> str:
        scores: dict[str, int] = {field: 0 for field in PHYSICS_KEYWORDS}
        for field, keywords in PHYSICS_KEYWORDS.items():
            for keyword in keywords:
                if keyword in self._raw_problem:
                    scores[field] += 1

        best_field = max(scores, key=lambda f: scores[f])

        if scores[best_field] == 0:
            best_field = self._ask_physics_field()

        print(f"\n  Detected physics field: {best_field.upper()}")
        return best_field

    def _ask_physics_field(self) -> str:
        fields = list(PHYSICS_KEYWORDS.keys())
        print("\n  Could not detect the physics field automatically.")
        print(f"  Available fields: {', '.join(fields)}")
        while True:
            field = input("  Enter the physics field: ").strip().lower()
            if field in fields:
                return field
            print(f"  '{field}' not recognised. Choose from: {', '.join(fields)}")