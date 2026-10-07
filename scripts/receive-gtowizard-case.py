"""Receive one serial case capture on localhost, with durable daily request counts."""
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import hashlib
import heapq
import json
import os

from gtowizard_case import ROOT, WORK, case_parser, check_source, child_jobs, load_case, metadata_path, plan, prepare_checkpoint

CASE = case_parser(__doc__).parse_args().case
listing, meta = load_case(CASE)
DATA = prepare_checkpoint(CASE)
WORK.mkdir(parents=True, exist_ok=True)
LEDGER = WORK / "daily-capture.json"


def capture_date():
    return datetime.now(timezone(timedelta(hours=8))).date().isoformat()


today = capture_date()
ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
day = ledger.setdefault(today, {"requests": 0, "strategyNodes": 0, "zeroRangeMarkers": 0, "stopReason": None})
case_stops = day.setdefault("caseStopReasons", {})
browser_stops = day.setdefault("browserStopReasons", {})
browser_stop_times = day.setdefault("browserStopObservedAt", {})
AUTH_ERRORS = {"Source HTTP 401", "Error: Source HTTP 401"}
QUOTA_STOP = "Source daily browsing limit reached"


def browser_for(case_id):
    return next(browser for browser, cases in plan["browsers"].items() if case_id in cases)


def record_stop(case_id, reason, restoring=False):
    if reason in AUTH_ERRORS:
        case_stops[case_id] = reason
    elif reason == QUOTA_STOP:
        browser = browser_for(case_id)
        browser_stops[browser] = reason
        if not restoring:
            browser_stop_times[browser] = datetime.now(timezone.utc).isoformat()
    elif not day["stopReason"]:
        day["stopReason"] = reason


browser_resets = {}
for case_id in plan["targetCases"]:
    metadata_file = metadata_path(case_id)
    if metadata_file.exists():
        case_meta = json.loads(metadata_file.read_text(encoding="utf-8"))
        stop = case_meta.get("captureStopReason") or {}
        if stop.get("observedAt") == today:
            record_stop(case_id, stop["message"], restoring=True)
        usage = case_meta.get("sourceUsage") or {}
        if usage.get("limit", 0) > 0 and usage.get("count", 0) >= usage["limit"] and usage.get("reset_date"):
            reset = datetime.fromisoformat(usage["reset_date"].replace("Z", "+00:00"))
            assert reset.tzinfo is not None, "Source reset time must include its timezone"
            browser = browser_for(case_id)
            browser_resets[browser] = max(reset, browser_resets.get(browser, reset))


def blocked_reason():
    if day["stopReason"]:
        return day["stopReason"]
    browser = browser_for(CASE)
    reset = browser_resets.get(browser)
    if reset and datetime.now(timezone.utc) < reset:
        return QUOTA_STOP
    reason = browser_stops.get(browser)
    observed = browser_stop_times.get(browser)
    # A reset still in the future returned above.
    expired = reason == QUOTA_STOP and reset and (not observed or reset > datetime.fromisoformat(observed))
    if reason and not expired:
        return reason
    reason = case_stops.get(CASE)
    return None if reason in AUTH_ERRORS else reason


def save_ledger():
    temporary = LEDGER.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(ledger, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(LEDGER)


def check_date():
    if capture_date() != today:
        raise ValueError("Capture date changed; stop this batch and restart the receiver for the next daily run")


def signature(response):
    if response["warning"] == "ZERO_RANGE":
        value = {"warning": "ZERO_RANGE", "actor": response["game"]["active_position"]}
    else:
        value = {**response, "players_info": [
            {**{k: v for k, v in player.items() if k != "simple_hand_counters"},
             "hand_keys": sorted(player["simple_hand_counters"])} for player in response["players_info"]]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


known = {}
if DATA.exists():
    with DATA.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # An interrupted append was never acknowledged. Keep its bytes
                # before removing only that unfinished final record.
                backup = DATA.with_name(DATA.name + ".incomplete-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                with backup.open("xb") as saved:
                    saved.write(line)
                    saved.flush()
                    os.fsync(saved.fileno())
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
                print(f"Recovered interrupted final record; saved bytes to {backup}", flush=True)
                break
            node = json.loads(line)
            known[node["path"]] = (signature(node["response"]), child_jobs(node))
queue = []
queued = set()


def enqueue(job):
    if job["path"] in known:
        for child in known[job["path"]][1]:
            enqueue(child)
    elif job["path"] not in queued:
        queued.add(job["path"])
        raises = sum(choice["action"].startswith(("Raise ", "Allin ")) for choice in job["history"])
        heapq.heappush(queue, (len(job["history"]), raises, job["path"], job))


enqueue({"path": "", "history": [], "expectedActor": meta["rootActor"]})
save_ledger()


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, content_type="application/json"):
        body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            check_date()
        except ValueError as error:
            self.send_error(409, str(error))
            return
        blocked = blocked_reason()
        if self.path == "/capture.js":
            self.reply((ROOT / "scripts/gtowizard-capture.js").read_text(), "text/javascript")
        elif self.path == "/config":
            self.reply({"caseId": CASE, "gameType": meta["mode"]["name"], "depth": listing["stack"],
                        "requestsUsed": day["requests"],
                        "blockedReason": blocked})
        elif self.path == "/next":
            self.reply({"job": queue[0][3] if queue and not blocked else None, "remainingEntrances": len(queue), "blockedReason": blocked})
        elif self.path == "/status":
            self.reply({"caseId": CASE, "knownResponses": len(known), "remainingEntrances": len(queue), "day": day})
        else:
            self.reply("<!doctype html><title>GTO case capture</title><h1>Serial case capture</h1>", "text/html; charset=utf-8")

    def do_POST(self):
        try:
            check_date()
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["case"] == CASE
            count = payload["requestCount"]
            assert type(count) is int and count >= day["requests"]
            day["requests"] = count
            if payload.get("stopReason"):
                record_stop(CASE, payload["stopReason"])
            save_ledger()
            node = payload.get("node")
            if node is None:
                self.reply({"saved": False, "day": day})
                return
            check_source(node, meta, listing)
            response = node["response"]
            fingerprint = signature(response)
            if node["path"] in known:
                assert known[node["path"]][0] == fingerprint, "Conflicting duplicate response"
            else:
                assert queue and node["path"] == queue[0][3]["path"], "Node is not the next serial job"
                job = queue[0][3]
                assert node["history"] == job["history"] and response["game"]["active_position"] == job["expectedActor"]
                children = child_jobs(node)
                with DATA.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(node, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                known[node["path"]] = (fingerprint, children)
                heapq.heappop(queue)
                queued.remove(node["path"])
                for child in children:
                    enqueue(child)
                day["zeroRangeMarkers" if response["warning"] == "ZERO_RANGE" else "strategyNodes"] += 1
                save_ledger()
            self.reply({"saved": True, "remainingEntrances": len(queue), "day": day})
        except (AssertionError, KeyError, ValueError) as error:
            self.send_error(409, str(error))

    def log_message(self, *_):
        pass


HTTPServer(("127.0.0.1", 8767), Handler).serve_forever()
