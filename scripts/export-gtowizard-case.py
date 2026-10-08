"""Validate the captured source tree, then write one case in catalog layout."""
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import gzip
import hashlib
import json
import math
import os

from gtowizard_case import (CASES, RAW, WORK, case_parser, check_source, checkpoint_path, child_jobs, copy_synced, history_key, label, load_case,
                            plan, terminal)

parser = case_parser(__doc__)
parser.add_argument("--status", action="store_true")
parser.add_argument("--checkpoint", action="store_true")
args = parser.parse_args()
CASE = args.case
listing, meta = load_case(CASE)
order = meta["handOrder"]
assert len(order) == len(set(order)) == 169
records = {}
source_path = checkpoint_path(CASE)
source_stream = source_path.open("rb")
source_stat = os.fstat(source_stream.fileno())
while True:
    offset = source_stream.tell()
    line = source_stream.readline()
    if not line:
        break
    assert line.endswith(b"\n"), "Interrupted final checkpoint record; start the local receiver to back up and repair its tail before exporting"
    row = json.loads(line)
    assert row["path"] not in records, row["path"]
    records[row["path"]] = offset


def read_line(path):
    source_stream.seek(records[path])
    return source_stream.readline()


def check_unchanged(stage):
    stat = os.fstat(source_stream.fileno())
    assert (stat.st_size, stat.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns), f"Checkpoint changed during {stage}; stop the receiver before exporting"


def read_record(path):
    return json.loads(read_line(path))


resume_jobs = []
terminals = Counter()
warnings = Counter()
missing_evs = 0
nonstandard_responses = 0
visited = set()
zero_paths = set()


def walk(job):
    global missing_evs, nonstandard_responses
    path = job["path"]
    if path not in records:
        resume_jobs.append(job)
        return
    assert path not in visited, path
    visited.add(path)
    row = read_record(path)
    assert row["history"] == job["history"], (path, row["history"], job["history"])
    check_source(row, meta, listing)
    data = row["response"]
    actor = data["game"]["active_position"]
    assert actor == job["expectedActor"], (path, actor, job["expectedActor"])
    if data["warning"]:
        warnings[data["warning"]] += 1
        zero_paths.add(path)
        return
    nonstandard_responses += "rawText" in row
    active = next(player for player in data["players_info"] if player["player"]["position"] == actor)
    assert sorted(active["simple_hand_counters"]) == order, path
    assert len(active["hand_evs"]) == 169 and all(len(player["range"]) == 169 for player in data["players_info"]), path
    for i, hand in enumerate(order):
        counter_ev = active["simple_hand_counters"][hand]["hand_ev"]
        ev = active["hand_evs"][i]
        # The source counters report zero for excluded hands; the arrays still
        # contain their strategy and EV. Only compare counters with positive reach.
        if active["range"][i] > 0:
            assert counter_ev == ev or (counter_ev is not None and ev is not None and math.isclose(counter_ev, ev, abs_tol=1e-12)), (path, hand)
        assert math.isclose(active["simple_hand_counters"][hand]["total_frequency"], active["range"][i], abs_tol=1e-12), (path, hand)
        if ev is None:
            missing_evs += 1
    actions = data["action_solutions"]
    assert actions and len({entry["action"]["code"] for entry in actions}) == len(actions)
    for entry in actions:
        action = entry["action"]
        assert action["position"] == actor
        assert len(entry["strategy"]) == len(entry["evs"]) == 169
        assert all(value is not None and math.isfinite(value) and 0 <= value <= 1 for value in entry["strategy"])
        missing_evs += sum(value is None for value in entry["evs"])
        end = terminal(action)
        if end:
            terminals[end] += 1
    for child in child_jobs(row):
        walk(child)


walk({"path": "", "history": [], "expectedActor": meta["rootActor"]})
missing = [job["path"] for job in resume_jobs]
excluded = set(records) - visited
assert all(any(path.startswith(zero + "-") for zero in zero_paths) for path in excluded), "Orphaned nodes outside ZERO_RANGE subtrees"
report = {"case": CASE, "nodes": len(visited), "strategyNodes": len(visited) - len(zero_paths),
          "zeroRangeMarkers": len(zero_paths), "excludedDescendants": len(excluded), "terminalEdges": dict(terminals),
          "unresolvedEdges": len(missing), "nextMissing": missing[:12],
          "sourceWarnings": dict(warnings), "unavailableEvValues": missing_evs,
          "minimumNodeCount": len(visited) + len(missing),
          "nonstandardJsonResponses": nonstandard_responses}
print(json.dumps(report, indent=2))
if args.status:
    raise SystemExit(0)
assert not missing or args.checkpoint, "The source tree is incomplete; use --checkpoint to save an explicitly incomplete archive"

# Source action groups the desktop colors (catalog.ts).
groups = {"FOLD", "CALL", "CHECK", "BET_SMALL", "BET_OVERBET"}
destination = CASES / CASE
destination.mkdir(parents=True, exist_ok=True)
nodes_by_actor = Counter()
node_files = []
node_index = {}
block = []
(destination / "chunks").mkdir(exist_ok=True)
maximum_rounding_error = Decimal(0)
maximum_row_difference = 0
normalized_hands = 0


def units(value):
    return int((Decimal(str(value)) * 10000).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def flush_block():
    if not block:
        return
    filename = f"chunks/{len(node_files):05d}.json"
    node_file = destination / filename
    node_file.write_text(json.dumps(block, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    node_files.append({"file": filename, "nodeCount": len(block), "sha256": hashlib.sha256(node_file.read_bytes()).hexdigest()})
    for node in block:
        node_index[history_key(node["history"])] = filename
    block.clear()


def store_node(node):
    nodes_by_actor[node["actor"]] += 1
    block.append(node)
    if len(block) == plan["nodesPerChunk"]:
        flush_block()


raw_archive = destination / "source.jsonl.gz"
with raw_archive.open("wb") as output, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
    for path in sorted(visited):
        line = read_line(path)
        compressed.write(line)
        row = json.loads(line)
        data = row["response"]
        actor = data["game"]["active_position"]
        if path in zero_paths:
            store_node({"sourceUrl": row["sourceUrl"], "actor": actor, "history": row["history"],
                        "sourceWarning": "ZERO_RANGE", "actions": [], "hands": {}})
            continue
        active = next(player for player in data["players_info"] if player["player"]["position"] == actor)
        entries = list(reversed(data["action_solutions"]))
        hands = {}
        for i, hand in enumerate(order):
            if active["range"][i] <= 0:
                continue
            raw_frequencies = [entry["strategy"][i] for entry in entries]
            frequencies = [units(value) for value in raw_frequencies]
            assert sum(frequencies) > 0, (path, hand)
            maximum_row_difference = max(maximum_row_difference, abs(sum(frequencies) - 10000))
            assert abs(sum(frequencies) - 10000) <= 2, (path, hand, frequencies)
            maximum_rounding_error = max(maximum_rounding_error, *(abs(Decimal(value) / 10000 - Decimal(str(raw))) for value, raw in zip(frequencies, raw_frequencies)))
            hands[hand] = frequencies
        normalized_hands += len(hands)
        actions = []
        for entry in entries:
            action = entry["action"]
            assert action["advanced_group"] in groups, (path, action["advanced_group"])
            actions.append({"code": action["code"], "label": label(action), "group": action["advanced_group"],
                            "next": {"kind": terminal(action) or "node"}})
        # EVs and incoming ranges keep the source's 169-class arrays in handOrder; action EV
        # arrays follow the actions' order.
        node = {"sourceUrl": row["sourceUrl"], "actor": actor,
                "history": row["history"], "actions": actions, "hands": hands,
                "nodeEvs": active["hand_evs"], "actionEvs": [entry["evs"] for entry in entries],
                "incomingRanges": {player["player"]["position"]: player["range"] for player in data["players_info"]},
                "sourceWarning": data["warning"]}
        store_node(node)

flush_block()
index_file = destination / "index.json"
index_file.write_text(json.dumps(node_index, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
source_mode = meta["mode"]
root_players = read_record("")["response"]["game"]["players"]
with raw_archive.open("rb") as stream:
    source_hash = hashlib.file_digest(stream, "sha256")
manifest = {
    "schemaVersion": 3, "id": CASE, "source": "GTO Wizard", "complete": not missing,
    "captureStopReason": meta.get("captureStopReason"),
    "listing": listing,
    "scope": {**meta["scope"], "stacks": [listing["stack"]]},
    "requestedLibraryScope": plan["scope"],
    "gameType": source_mode["name"], "stack": listing["stack"],
    "positions": [player["position"] for player in root_players],
    "smallBlind": meta["smallBlind"], "bigBlind": meta["bigBlind"], "ante": meta["ante"],
    "rake": source_mode["info"]["rake"], "rakePercent": source_mode["info"]["rake_pct"],
    "rakeCap": source_mode["info"]["rake_cap"], "openingSize": meta["openingSize"],
    "sourceMode": {**source_mode, "game_modes": [mode for mode in source_mode["game_modes"] if Decimal(mode["depth"]) == listing["stack"]]},
    "handOrder": order,
    "frequencyUnits": "0.01 percentage points; normalize each retained row by its sum when applying",
    "handCoverage": "Strategy rows (hands) cover positive incoming hands; nodeEvs, actionEvs (per action) and incomingRanges are source 169-class arrays in handOrder. ZERO_RANGE nodes contain only markers.",
    "zeroRangePolicy": "Store a ZERO_RANGE marker, omit strategy and EV, and do not traverse descendants. Complete means all other preflop branches are captured.",
    "evUnits": "bb; source values for the acting player, unchanged; null denotes unavailable source values",
    "evCoverage": "complete for captured normal nodes" if not missing_evs else "source contains unavailable values",
    "nodeCount": len(visited), "strategyNodeCount": len(visited) - len(zero_paths), "zeroRangeNodeCount": len(zero_paths),
    "nodesByActor": dict(nodes_by_actor),
    "normalizedHandRows": normalized_hands,
    # The report's other fields; its case and node counts are the fields above.
    "validation": {**{key: value for key, value in report.items() if key not in ("case", "nodes", "strategyNodes", "zeroRangeMarkers")},
                   "maximumFrequencyRoundingError": str(maximum_rounding_error), "maximumRowSumDifference": maximum_row_difference},
    "sourceArchive": {"file": raw_archive.name, "sha256": source_hash.hexdigest()},
    "nodeFiles": node_files,
    "index": {"file": index_file.name, "sha256": hashlib.sha256(index_file.read_bytes()).hexdigest()},
    "resume": {"file": "resume.json"},
    "desktopIntegration": "Rebuild the app after exporting. Saved branches load on demand; missing branches are locked.",
}
(destination / "resume.json").write_text(json.dumps(resume_jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
# Publish only at a stopped capture milestone, never for each appended response.
# Stage outside Git/LFS so a file watcher cannot cache partially copied snapshots.
snapshot = RAW / source_path.name
check_unchanged("export")
# The checkpoint only grows from the snapshot it was seeded with, so an equal size means equal bytes.
if source_path != snapshot and not (snapshot.exists() and snapshot.stat().st_size == source_stat.st_size):
    temporary = WORK / (source_path.name + ".publish-tmp")
    source_stream.seek(0)
    copy_synced(source_stream, temporary)
    check_unchanged("publication")
    temporary.replace(snapshot)
source_stream.close()
(destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
print("Saved " + str(destination))
