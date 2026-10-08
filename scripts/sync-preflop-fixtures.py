"""Derive bundled test scenarios from the captured GTO Wizard cases.

This copies input weights only. Independent reference answers must be regenerated
with tests/oracle after changing a referenced scenario; this script never creates answers.
"""
import hashlib
import json

from gtowizard_case import CASES, ROOT, history_key

RAKED = "Cash6mSimple_6mGGrcR25_100"
RAKE_FREE = "Cash6mSimple_6mcEVR25_100"


class Case:
    def __init__(self, case):
        self.directory = CASES / case
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        assert self.manifest["id"] == case and self.manifest["schemaVersion"] == 3
        self.index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.chunks = {}

    def node(self, history):
        """The saved node at a history, or None after an action that ends preflop."""
        file = self.index.get(history_key(history))
        if file is None:
            return None
        if file not in self.chunks:
            nodes = json.loads((self.directory / file).read_text(encoding="utf-8"))
            self.chunks[file] = {history_key(node["history"]): node for node in nodes}
        return self.chunks[file][history_key(history)]


def derive(case, history):
    """Mirror the desktop's replayPreflop: full ranges, each saved node's incoming ranges, then the actor's action
    frequencies, and finally the incoming ranges of the node the history reaches, if saved."""
    ranges = {position: dict.fromkeys(case.manifest["handOrder"], 1) for position in case.manifest["positions"]}
    used = []

    def source_ranges(node):
        used.append(node)
        for position, weights in node["incomingRanges"].items():
            ranges[position] = dict(zip(case.manifest["handOrder"], weights, strict=True))

    for step, choice in enumerate(history):
        node = case.node(history[:step])
        assert node and node["actor"] == choice["actor"] and not node["sourceWarning"], history[:step + 1]
        source_ranges(node)
        action = next(i for i, candidate in enumerate(node["actions"]) if candidate["label"] == choice["action"])
        ranges[choice["actor"]] = {
            hand: weight * node["hands"][hand][action] / sum(node["hands"][hand])
            for hand, weight in ranges[choice["actor"]].items()
            if hand in node["hands"] and weight > 0 and node["hands"][hand][action] > 0
        }
    assert not history or node["actions"][action]["next"]["kind"] == "flop", history
    if reached := case.node(history):
        source_ranges(reached)
    digest = hashlib.sha256(json.dumps(used, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {position: {hand: weight for hand, weight in weights.items() if weight > 0}
            for position, weights in ranges.items()}, digest


def only_raise(node):
    labels = [action["label"] for action in node["actions"] if action["label"].startswith("Raise ")]
    assert len(labels) == 1, labels
    return labels[0]


def line(case, bet_level):
    positions = case.manifest["positions"]
    history = [{"actor": "UTG", "action": "Raise 2.5"}] + [{"actor": position, "action": "Fold"} for position in positions[1:-1]]
    assert positions[0] == "UTG" and positions[-1] == "BB" and bet_level in (2, 3, 4)
    if bet_level == 2:
        return history + [{"actor": "BB", "action": "Call"}]
    history.append({"actor": "BB", "action": only_raise(case.node(history))})
    if bet_level == 3:
        return history + [{"actor": "UTG", "action": "Call"}]
    history.append({"actor": "UTG", "action": only_raise(case.node(history))})
    return history + [{"actor": "BB", "action": "Call"}]


def scenario(case_id, bet_level, selected, stack, board="Ks 9s 2d", pot=2.0, iterations=200):
    """bet_level None takes the root's ranges."""
    case = Case(case_id)
    history = line(case, bet_level) if bet_level else []
    ranges, digest = derive(case, history)
    retained = {position: {hand: ranges[position][hand] for hand in hands}
                for position, hands in selected.items()} if selected else {position: ranges[position] for position in ("UTG", "BB")}
    assert all(weight > 0 for weights in retained.values() for weight in weights.values())
    return {
        "board": board, "heroPosition": "UTG", "villainPosition": "BB", "heroActsFirst": False,
        "initialPot": pot, "heroStack": stack, "villainStack": stack, "iterations": iterations,
        "rakePercent": case.manifest["rakePercent"], "rakeCap": case.manifest["rakeCap"],
        # Half-pot fixture sizes avoid cross-engine chip-rounding differences.
        "bettingTree": {**{position: {street: {"bet": [50], "raise": [50]} for street in ("flop", "turn", "river")}
                           for position in ("oop", "ip")},
                        "maxRaises": 2, "allInSpr": 0.15},
        "ranges": retained,
        # sha256 covers the canonical JSON of the saved nodes the history reads.
        "rangeSource": {"catalog": f"resources/gtowizard-preflop/cases/{case_id}", "sha256": digest, "solution": case_id,
                        "history": history, "retainedClasses": selected,
                        "reduction": "Postflop pot/stacks are reduced for runtime; retained hand classes, when listed, are a test subset. Source weights are not rounded or rescaled."},
    }


fixtures = ROOT / "tests/fixtures"
weighted = scenario(RAKED, 4, {"UTG": ["AKs", "TT"], "BB": ["KK", "A5s"]}, 4.0)
raised = scenario(RAKE_FREE, 4, {"UTG": ["KJs"], "BB": ["AQs"]}, 8.0)
wide = scenario(RAKED, 2, None, 97.5, "Ac Kh Qs", 5.5, 1000)
for sizes in wide["bettingTree"]["oop"], wide["bettingTree"]["ip"]:
    for street in ("flop", "turn", "river"):
        sizes[street]["bet"] = [33, 125]
# Keep small bets as distinct branches instead of replacing them with all-ins.
wide["bettingTree"]["allInSpr"] = 0.0
wide["rangeSource"]["reduction"] = (
    "Full captured 6-max GG R&C UTG raise 2.5 / BB call ranges. "
    "100bb starting stacks leave 97.5bb each; the 5.5bb pot includes the folded SB's 0.5bb. "
    "No range subsets or weight rescaling."
)
# Lockstep CPU/GPU updates; iterations counts compared updates.
parity = scenario(RAKED, 3, None, 30.0, "Ac Kh Qs", 22.5, 12)
parity["rangeSource"]["reduction"] = (
    "Full captured 6-max GG R&C UTG raise 2.5 / BB raise 11 / UTG call ranges. "
    "Stacks are reduced to 30bb for runtime while GPU street regions still split into batches; "
    "the 22.5bb pot includes the folded SB's 0.5bb. No range subsets or weight rescaling. "
    "The CPU backend is the reference; there is no independent answer."
)
# The same lockstep comparison over full ranges, which take the GPU's wide-range kernel paths.
wide_parity = scenario(RAKED, None, None, 4.0, iterations=6)
wide_parity["rangeSource"]["reduction"] = (
    "Both players' full ranges at the preflop root, which no postflop line of the source reaches: UTG's captured incoming "
    "range and BB's full starting range, as the desktop replays the root, which records only its actor's range. Pot and "
    "stacks are reduced to a shallow tree for runtime. The CPU backend is the reference; there is no independent answer."
)
for name, value in [("weighted-flop", weighted), ("raise-flop", raised), ("utg-bb-wide", wide), ("backend-parity", parity),
                    ("wide-parity", wide_parity)]:
    (fixtures / f"{name}.json").write_bytes((json.dumps(value, indent=4) + "\n").encode())
print("Updated five input fixtures from GTO Wizard; regenerate independent references for changed referenced inputs next.")
