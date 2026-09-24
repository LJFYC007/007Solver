"""Writes catalog-derived scenarios for the daily GPU parity and race checks.

Ranges come from the tested revision's captured catalog through the derive() of its
scripts/sync-preflop-fixtures.py (never custom or inferred ranges). Boards, reduced stacks and
betting trees are harness choices recorded in rangeSource. Both 6-max BTN/BB ranges exceed 256
combos, which exercises CUDA's multi-tile Backup grid that no repository fixture reaches.

Usage: python3 tests/daily/scenarios.py <source-root> <output-dir>
"""

import json
from pathlib import Path
import sys

source_root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
script = source_root / "scripts/sync-preflop-fixtures.py"
text = script.read_text()
marker = "\nfixtures = ROOT"
if marker not in text:
    sys.exit(f"{script} no longer has '{marker.strip()}'; update tests/daily/scenarios.py")
namespace = {"__file__": str(script)}
exec(compile(text.split(marker)[0], str(script), "exec"), namespace)

SRP_6MAX = [{"actor": p, "action": "Fold"} for p in ("UTG", "HJ", "CO")] + [
    {"actor": "BTN", "action": "Raise 2.5"},
    {"actor": "SB", "action": "Fold"},
    {"actor": "BB", "action": "Call"},
]


def srp_scenario(board, updates):
    ranges = namespace["derive"]("6max", SRP_6MAX)
    return {
        "board": board,
        "heroPosition": "BTN",
        "villainPosition": "BB",
        "heroActsFirst": False,
        "initialPot": 5.5,
        "heroStack": 10.0,
        "villainStack": 10.0,
        "iterations": updates,
        "bettingTree": {**{street: {"bet": [50], "raise": []} for street in ("flop", "turn", "river")}, "maxRaises": 0, "allInSpr": 0.0},
        "ranges": {"BTN": ranges["BTN"], "BB": ranges["BB"]},
        "rangeSource": {
            "catalog": "resources/gtowizard-preflop",
            "sha256": namespace["digest"],
            "solution": "6max",
            "history": SRP_6MAX,
            "retainedClasses": None,
            "reduction": "Full captured 6-max BTN raise 2.5 / SB fold / BB call ranges; the 5.5bb pot includes the "
            "folded SB's 0.5bb. Stacks are reduced to 10bb and raises disabled for emulated-GPU runtime. Daily harness only.",
        },
    }


SCENARIOS = {
    "btn-bb-srp-dry": srp_scenario("Ks 9s 2d", 4),
    "btn-bb-srp-paired": srp_scenario("7h 7d 2c", 4),
    "btn-bb-srp-monotone": srp_scenario("Ah 8h 3h", 4),
}

output.mkdir(parents=True, exist_ok=True)
for name, scenario in SCENARIOS.items():
    (output / f"{name}.json").write_text(json.dumps(scenario, indent=1) + "\n")
    print(name, {position: len(weights) for position, weights in scenario["ranges"].items()}, "hand classes")
