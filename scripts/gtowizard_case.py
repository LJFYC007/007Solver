"""Case setup and source rules shared by the GTO Wizard receiver and exporter."""
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
# Raw capture checkpoints and verified case metadata. Checkpoints are only ever appended.
RAW = ROOT / "resources/gtowizard-preflop/raw"
# Exported cases in catalog layout.
CASES = ROOT / "resources/gtowizard-preflop/cases"
# The receiver's daily ledger.
WORK = ROOT / "build/gtowizard-capture"
plan = json.loads((ROOT / "resources/gtowizard-preflop/capture-plan.json").read_text(encoding="utf-8"))


def case_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--case", choices=plan["targetCases"], default=plan["currentCase"])
    return parser


def load_case(case):
    """Return the case's unlocked library listing and its verified capture metadata."""
    library = json.loads((ROOT / "resources/gtowizard-preflop/library.json").read_text(encoding="utf-8"))
    listing = next(entry for entry in library["solutions"] if entry["id"] == case)
    assert listing["availability"] == "unlocked", "Source solution is locked"
    meta = json.loads(metadata_path(case).read_text(encoding="utf-8"))
    assert meta["caseId"] == case and f'{meta["mode"]["name"]}_{listing["stack"]}' == case
    assert f'{meta["mode"]["players"]}max' == plan["scope"]["players"] and meta["mode"]["info"]["rake"] == listing["rake"]
    assert f'{meta["openingSize"]}x' == plan["scope"]["openingSize"]
    return listing, meta


def metadata_path(case):
    return RAW / (case + ".metadata.json")


def checkpoint_path(case):
    """The receiver's durable JSONL of acknowledged source responses, which the exporter reads."""
    return RAW / (case + ".raw.jsonl")


def history_key(history):
    """A history's index.json key; the desktop's choiceKey must produce the same string."""
    return json.dumps([[choice["actor"], choice["action"]] for choice in history], separators=(",", ":"))


def check_source(node, meta, listing):
    """Assert that a captured record is this case's preflop node at its path."""
    query = parse_qs(urlparse(node["requestUrl"]).query, keep_blank_values=True)
    assert query["gametype"] == [meta["mode"]["name"]] and query["depth"] == [str(listing["stack"])]
    assert query["preflop_actions"] == [node["path"]]
    assert not any(query.get(key, [""])[0] for key in ("stacks", "board", "flop_actions", "turn_actions", "river_actions"))
    response = node["response"]
    assert response["game"]["current_street"]["type"] == "PREFLOP"
    assert response["warning"] in (None, "ZERO_RANGE") and not response["hands_locked"], node["path"]


def label(action):
    """History label of a source action; histories built by the receiver must match the exporter's."""
    if action["type"] in ("RAISE", "BET"):
        size = format(Decimal(action["betsize"]), "f")
        if "." in size:
            size = size.rstrip("0").rstrip(".")
        return ("Allin " if action["allin"] else "Raise ") + size
    return action["type"].capitalize()


def terminal(action):
    """The street or hand end an action reaches, or None when it leads to another preflop node."""
    assert all(type(action[name]) is bool for name in ("next_street", "is_hand_end", "is_showdown"))
    if action["is_showdown"]:
        return "showdown"
    if action["is_hand_end"]:
        return "hand-end"
    if action["next_street"]:
        return "flop"
    return None


def child_path(path, code):
    return path + "-" + code if path else code


def child_jobs(node):
    """Capture jobs for a saved node's actions that lead to further preflop nodes."""
    response = node["response"]
    if response["warning"] == "ZERO_RANGE":
        return []
    children = []
    for entry in response["action_solutions"]:
        action = entry["action"]
        if terminal(action):
            continue
        children.append({"path": child_path(node["path"], action["code"]),
                         "history": node["history"] + [{"actor": response["game"]["active_position"], "action": label(action)}],
                         "expectedActor": action["next_position"]})
    return children
