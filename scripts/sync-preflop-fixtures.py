"""Derive bundled test scenarios from the captured GTO Wizard catalog.

This copies input weights only. Independent reference answers must be regenerated
with tests/oracle after changing a referenced scenario; this script never creates answers.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "resources/gtowizard-preflop"
catalog = json.loads((CATALOG_PATH / "catalog.json").read_text())
# Hash the directory deterministically, including names so moves are visible in provenance.
catalog_hash = hashlib.sha256()
for path in sorted(CATALOG_PATH.rglob("*.json")):
    catalog_hash.update(path.relative_to(CATALOG_PATH).as_posix().encode() + b"\0")
    catalog_hash.update(path.read_bytes() + b"\0")
digest = catalog_hash.hexdigest()
solutions = {solution["id"]: solution for solution in catalog["solutions"]}


def history_key(history):
    return tuple((choice["actor"], choice["action"]) for choice in history)


nodes = {}
for path in sorted(CATALOG_PATH.glob("*/*.json")):
    format_name, actor = path.parent.name, path.stem
    solution = solutions[format_name]
    for node in json.loads(path.read_text()):
        game_type = parse_qs(urlparse(node["sourceUrl"]).query)["gametype"][0]
        assert game_type == solution["gameType"] and actor == node["actor"].lower(), path
        assert node["actor"] in solution["positions"], path
        key = (game_type, history_key(node["history"]))
        assert key not in nodes, f"Duplicate source history: {path}"
        nodes[key] = node


def derive(format_name, history):
    solution = solutions[format_name]
    full_range = nodes[solution["gameType"], ()]["hands"]
    ranges = {position: dict.fromkeys(full_range, 1.0) for position in solution["positions"]}
    for step, choice in enumerate(history):
        node = nodes[solution["gameType"], history_key(history[:step])]
        assert node["actor"] == choice["actor"]
        action = next(i for i, candidate in enumerate(node["actions"]) if candidate["label"] == choice["action"])
        ranges[choice["actor"]] = {
            hand: weight * node["hands"][hand][action] / sum(node["hands"][hand])
            for hand, weight in ranges[choice["actor"]].items()
            if hand in node["hands"] and weight > 0 and node["hands"][hand][action] > 0
        }
    return ranges


def line(format_name, bet_level):
    solution = solutions[format_name]
    history = [{"actor": position, "action": "Raise 2.5" if position == "UTG" else "Fold"}
               for position in solution["positions"][:-1]]
    assert bet_level in (3, 4)
    history.append({"actor": "BB", "action": "Raise 13" if format_name == "8max" else "Raise 12.5"})
    if bet_level == 3:
        return history + [{"actor": "UTG", "action": "Call"}]
    return history + [{"actor": "UTG", "action": "Raise 23.5" if format_name == "8max" else "Raise 26.5"},
                      {"actor": "BB", "action": "Call"}]


def scenario(format_name, bet_level, selected, stack, board="Ks 9s 2d", pot=2.0, iterations=200):
    history = line(format_name, bet_level)
    ranges = derive(format_name, history)
    retained = {position: {hand: ranges[position][hand] for hand in hands}
                for position, hands in selected.items()} if selected else {position: ranges[position] for position in ("UTG", "BB")}
    assert all(weight > 0 for weights in retained.values() for weight in weights.values())
    return {
        "board": board, "heroPosition": "UTG", "villainPosition": "BB", "heroActsFirst": False,
        "initialPot": pot, "heroStack": stack, "villainStack": stack, "iterations": iterations,
        # Half-pot fixture sizes avoid cross-engine chip-rounding differences.
        "bettingTree": {**{street: {"bet": [50], "raise": [50]} for street in ("flop", "turn", "river")},
                        "maxRaises": 2, "allInSpr": 0.15},
        "ranges": retained,
        "rangeSource": {"catalog": "resources/gtowizard-preflop", "sha256": digest, "solution": format_name,
                        "history": history, "retainedClasses": selected,
                        "reduction": "Postflop pot/stacks are reduced for runtime; retained hand classes, when listed, are a test subset. Source weights are not rounded or rescaled."},
    }


fixtures = ROOT / "tests/fixtures"
weighted = scenario("8max", 4, {"UTG": ["AKs", "QQ"], "BB": ["KK", "A5s"]}, 4.0)
raised = scenario("6max", 4, {"UTG": ["KJs"], "BB": ["AQs"]}, 8.0)
wide = scenario("8max", 3, None, 87.0, "Ac Kh Qs", 26.5, 2000)
for street in ("flop", "turn", "river"):
    wide["bettingTree"][street]["bet"] = [33, 125]
# Keep small bets as distinct branches instead of replacing them with all-ins.
wide["bettingTree"]["allInSpr"] = 0.0
wide["rangeSource"]["reduction"] = (
    "Full captured 8-max UTG raise 2.5 / BB raise 13 / UTG call ranges. "
    "100bb starting stacks leave 87bb each; the 26.5bb pot includes the folded SB's 0.5bb. "
    "No range subsets or weight rescaling."
)
# Lockstep CPU/GPU updates; iterations counts compared updates.
parity = scenario("8max", 3, None, 30.0, "Ac Kh Qs", 26.5, 12)
parity["rangeSource"]["reduction"] = (
    "Full captured 8-max UTG raise 2.5 / BB raise 13 / UTG call ranges. "
    "Stacks are reduced to 30bb for runtime while GPU street regions still split into batches; "
    "the 26.5bb pot includes the folded SB's 0.5bb. No range subsets or weight rescaling. "
    "The CPU backend is the reference; there is no independent answer."
)
for name, value in [("weighted-flop", weighted), ("raise-flop", raised), ("utg-bb-wide", wide), ("backend-parity", parity)]:
    (fixtures / f"{name}.json").write_bytes((json.dumps(value, indent=4) + "\n").encode())
print("Updated four input fixtures from GTO Wizard; regenerate independent references for changed referenced inputs next.")
