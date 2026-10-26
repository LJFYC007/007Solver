"""Derive bundled test scenarios from the captured GTO Wizard cases.

This copies input weights only. Independent reference answers must be regenerated
with tests/oracle after changing a referenced scenario; this script never creates answers.
"""
import hashlib
import json

from gtowizard_case import CASES, ROOT, history_key

RAKED = "Cash6mSimple_6mGGrcR25_100"
RAKE_FREE = "Cash6mSimple_6mcEVR25_100"
RANKS, SUITS = "AKQJT98765432", "shdc"


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


def combos(hand_class):
    first, second, kind = hand_class[0], hand_class[1], hand_class[2:]
    if first == second:
        suits = [(a, b) for i, a in enumerate(SUITS) for b in SUITS[i + 1:]]
    else:
        suits = [(a, b) for a in SUITS for b in SUITS if (a == b) == (kind == "s")]
    return [(first + a, second + b) for a, b in suits]


def lock(scenario, actor, path, patterns, positions=(0, 1, 4), period=7):
    """Locks the actor's hands at the given positions of every period hands in the solver's hand order (by card index:
    rank, then suit shdc). The default mixes locked and free hands within 8-hand AVX2 chunks and in every combination
    across adjacent hands, which GPU threads move together; an odd period keeps the latter. Weights cycle through
    patterns; hands holding a dealt card stay free."""
    flop = set(scenario["board"].split())
    dealt = flop | {step for step in path if isinstance(step, str)}
    index = lambda hand: sorted(RANKS.index(card[0]) * 4 + SUITS.index(card[1]) for card in hand)
    hands = sorted((hand for hand_class in scenario["ranges"][actor] for hand in combos(hand_class) if not flop & set(hand)), key=index)
    locked = [hand for i, hand in enumerate(hands) if i % period in positions and not dealt & set(hand)]
    return {"path": path, "hands": {"".join(hand): patterns[i % len(patterns)] for i, hand in enumerate(locked)}}


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
# The same comparison on flops whose suit group swaps two suits or permutes three, so training runs suit-isomorphic.
two_tone_parity = {**parity, "board": "Ah Kh Qs"}
monotone_parity = {**parity, "board": "Ah Kh Qh"}
# The same lockstep comparison over full ranges, which take the GPU's wide-range kernel paths.
wide_parity = scenario(RAKED, None, None, 4.0, iterations=6)
wide_parity["rangeSource"]["reduction"] = (
    "Both players' full ranges at the preflop root, which no postflop line of the source reaches: UTG's captured incoming "
    "range and BB's full starting range, as the desktop replays the root, which records only its actor's range. Pot and "
    "stacks are reduced to a shallow tree for runtime. The CPU backend is the reference; there is no independent answer."
)
# Node locks (hand -> one weight per action, at an action-index and dealt-card path) are test inputs, not ranges.
# A rainbow unpaired flop keeps the oracle from merging suit-isomorphic runouts, so its locks may sit on any street.
locked = scenario(RAKE_FREE, 4, {"UTG": ["KJs", "TT"], "BB": ["AQs", "QQ"]}, 4.0, "Ks 9h 2d")
locked["locks"] = [
    # BB's whole range at the root (check, bet); zero weights take actions out of some hands' play.
    {"path": [], "hands": {"AsQs": [1, 3], "AhQh": [3, 1], "AdQd": [0, 1], "AcQc": [1, 1], "QsQh": [1, 0],
                           "QsQd": [1, 0], "QsQc": [3, 1], "QhQd": [2, 0], "QhQc": [1, 0], "QdQc": [1, 1]}},
    # Partial locks of both players: UTG after a check (check, bet) and facing a bet (fold, call, all-in), BB facing a
    # bet after checking, and BB on a concrete turn.
    {"path": [0], "hands": {"KhJh": [0, 1], "ThTd": [1, 1]}},
    {"path": [1], "hands": {"KdJd": [0, 1, 1], "TsTc": [1, 3, 0]}},
    {"path": [0, 1], "hands": {"QhQd": [0, 1, 0], "AhQh": [2, 1, 1]}},
    {"path": [0, 0, "7c"], "hands": {"AsQs": [1, 1]}},
]
# Locked lockstep updates on the parity tree, at a decision of each GPU lock site: the root, a decision inlined into its
# parent's Backup (flop and river), street roots on a concrete runout, a river decision fused into its fold and
# showdown's Terminal block, and a whole-range lock whose zero weight prunes the opponent's subtree.
locked_parity = {**parity, "iterations": 4}
river = [0, 0, "7d", 0, 0, "2c"]
two, three = [[1, 0], [1, 3], [0, 1], [2, 1]], [[0, 1, 0], [1, 1, 2], [1, 0, 0], [0, 2, 1]]
locked_parity["locks"] = [
    lock(parity, "BB", [], two),
    lock(parity, "UTG", [1], three),
    lock(parity, "BB", [0, 0, "7d"], two),
    lock(parity, "UTG", [0, 0, "7d", 0], [[1, 0]], range(7)),
    lock(parity, "BB", river, two),
    lock(parity, "UTG", river + [1], three),
    lock(parity, "BB", river + [1, 2], two),
]
# The benchmark's locked comparison over the wide parity input's full ranges, whose Reach and Backup items span several
# 512-hand tiles and whose Terminal groups double: sparse partial locks reach every tile at the root, an inlined child,
# street roots on a concrete runout and a fused river fold/call. Two updates run each locked site for both players.
locked_wide_parity = {**wide_parity, "iterations": 2}
locked_wide_parity["locks"] = [
    lock(wide_parity, "BB", [], two, period=49),
    lock(wide_parity, "UTG", [1], three, period=49),
    lock(wide_parity, "BB", [0, 0, "7d"], two, period=49),
    lock(wide_parity, "BB", river, two, period=49),
    lock(wide_parity, "BB", river + [1, 2], two, period=49),
]
for name, value in [("weighted-flop", weighted), ("raise-flop", raised), ("utg-bb-wide", wide), ("backend-parity", parity),
                    ("two-tone-parity", two_tone_parity), ("monotone-parity", monotone_parity), ("wide-parity", wide_parity),
                    ("locked-flop", locked), ("locked-parity", locked_parity), ("locked-wide-parity", locked_wide_parity)]:
    (fixtures / f"{name}.json").write_bytes((json.dumps(value, indent=4) + "\n").encode())
print("Updated the input fixtures from GTO Wizard; regenerate independent references for changed referenced inputs next.")
