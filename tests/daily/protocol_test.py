"""Robustness of the JSON-lines service protocol after a solve (solver/ARCHITECTURE.md, "Service
and desktop lifecycle"): every request line gets exactly one reply, errors never end the session,
node reports satisfy their documented invariants, and EOF drains queued work.

Usage: python3 tests/daily/protocol_test.py <solver_service> <fixtures-dir> [unittest args]
"""

import json
import math
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import unittest

SERVICE = str(Path(sys.argv.pop(1)).resolve())
FIXTURES = Path(sys.argv.pop(1)).resolve()
TIMEOUT = 600  # generous for sanitizer builds


class Session:
    """A solved raise-flop session; stops at a loose accuracy target to keep setup short."""

    def __init__(self):
        scenario = json.loads((FIXTURES / "raise-flop.json").read_text())
        scenario["accuracyPercent"] = 100
        self.process = subprocess.Popen(
            [SERVICE, "--stdin", "--device=cpu"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.lines = queue.Queue()
        self.stderr = []
        self.readers = [
            threading.Thread(target=self._read, daemon=True),
            threading.Thread(target=lambda: self.stderr.extend(self.process.stderr), daemon=True),
        ]
        for reader in self.readers:
            reader.start()
        self.write(json.dumps(scenario).encode())
        while True:
            message = self.receive()
            if message.get("event") == "ready":
                self.ready = message
                break
            assert message.get("event") in ("building_tree", "solving"), message

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def write(self, raw):
        self.process.stdin.write(raw + b"\n")
        self.process.stdin.flush()

    def receive(self):
        line = self.lines.get(timeout=TIMEOUT)
        assert line is not None, "service closed stdout; stderr: " + b"".join(self.stderr).decode(errors="replace")[-2000:]
        return json.loads(line)

    def request(self, request_id, command, node_id):
        self.write(json.dumps({"requestId": request_id, "command": command, "nodeId": node_id}).encode())

    def finish(self):
        """Closes stdin and returns (remaining replies, exit code)."""
        self.process.stdin.close()
        self.process.wait(timeout=TIMEOUT)
        replies = []
        while True:
            line = self.lines.get(timeout=TIMEOUT)
            if line is None:
                return replies, self.process.returncode
            replies.append(json.loads(line))

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        for reader in self.readers:
            reader.join(timeout=TIMEOUT)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if not stream.closed:
                stream.close()


class ProtocolRobustnessTest(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        self.addCleanup(self.session.kill)

    def query_ok(self, request_id, command, node_id):
        self.session.request(request_id, command, node_id)
        reply = self.session.receive()
        self.assertEqual(reply["requestId"], request_id)
        self.assertTrue(reply["ok"], reply)
        return reply

    def assert_error_reply(self, raw, request_id=0):
        self.session.write(raw)
        reply = self.session.receive()
        self.assertFalse(reply["ok"], (raw[:200], reply))
        self.assertTrue(reply["error"], reply)
        self.assertEqual(reply["requestId"], request_id, (raw[:200], reply))

    def test_malformed_requests_each_get_one_error_reply(self):
        root = self.query_ok(1, "query_node", self.session.ready["rootNodeId"])["node"]
        node_count = self.session.ready["nodeCount"]
        # Lines without a usable requestId are answered with requestId 0.
        for raw in [
            b"not json",
            b"",
            b"   ",
            b"{}",
            b"[1, 2]",
            b"5",
            b"null",
            b"[" * 100000 + b"]" * 100000,
            b'{"requestId": -1, "command": "query_node", "nodeId": 0}',
            b'{"requestId": 1.5, "command": "query_node", "nodeId": 0}',
            b'{"requestId": "7", "command": "query_node", "nodeId": 0}',
            b'{"requestId": 18446744073709551616, "command": "query_node", "nodeId": 0}',
            b'{"requestId": 3, "command": "query_node", "nodeId": 0} trailing',
        ]:
            with self.subTest(raw=raw[:60]):
                self.assert_error_reply(raw)
        for request in [{"command": "query_node", "nodeId": 0}, {"requestId": None, "command": "query_node", "nodeId": 0}]:
            with self.subTest(request=request):
                self.assert_error_reply(json.dumps(request).encode())
        # A valid requestId is echoed even when the rest of the request is invalid.
        for request_id, request in enumerate(
            [
                {"nodeId": 0},
                {"command": 123, "nodeId": 0},
                {"command": "unknown", "nodeId": 0},
                {"command": "query_node"},
                {"command": "query_node", "nodeId": -1},
                {"command": "query_node", "nodeId": 1.0},
                {"command": "query_node", "nodeId": "0"},
                {"command": "query_node", "nodeId": 2**31},
                {"command": "query_node", "nodeId": 2**31 - 1},
                {"command": "query_node", "nodeId": node_count},
                {"command": "query_node_evs", "nodeId": node_count},
                {"command": "query_equity", "nodeId": node_count},
            ],
            start=100,
        ):
            request = {"requestId": request_id, **request}
            with self.subTest(request=request):
                self.assert_error_reply(json.dumps(request).encode(), request_id)
        # Unknown fields are ignored and the session still answers valid queries identically.
        self.session.write(json.dumps({"requestId": 200, "command": "query_node", "nodeId": 0, "extra": [1, {"x": None}]}).encode())
        reply = self.session.receive()
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(self.query_ok(201, "query_node", self.session.ready["rootNodeId"])["node"], root)
        self.assertTrue(self.query_ok(202, "query_node", node_count - 1)["node"])
        replies, code = self.session.finish()
        self.assertEqual((replies, code), ([], 0))

    def test_sampled_nodes_satisfy_report_invariants(self):
        node_count = self.session.ready["nodeCount"]
        ids = sorted(set(random.Random(20260924).sample(range(node_count), 300)) | {0, node_count - 1})
        for request_id, node_id in enumerate(ids):
            self.session.request(request_id, "query_node", node_id)
        nodes = {}
        for _ in ids:
            reply = self.session.receive()
            self.assertTrue(reply["ok"], reply)
            nodes[ids[reply["requestId"]]] = reply["node"]
        kinds = {}
        for node_id, node in nodes.items():
            kinds[node["kind"]] = kinds.get(node["kind"], 0) + 1
            with self.subTest(node=node_id, kind=node["kind"]):
                board = set(node["state"]["board"])
                if node["kind"] == "decision":
                    self.check_decision(node, board)
                elif node["kind"] == "chance":
                    # AnalysisSession omits cards that every live hand pair blocks, so only a subset is guaranteed.
                    cards = [outcome["card"] for outcome in node["outcomes"]]
                    self.assertEqual(len(set(cards)), len(cards))
                    self.assertFalse(set(cards) & board)
                    self.assertTrue(cards)
                    self.assertLessEqual(len(cards), 52 - len(board))
                else:
                    self.assertEqual(node["kind"], "terminal")
                    self.assertIn(node["result"]["reason"], ("fold", "showdown"))
        self.assertGreater(kinds.get("decision", 0), 0, kinds)
        # EVs (asynchronous) and equities on a subset of decision nodes.
        decisions = [node_id for node_id, node in nodes.items() if node["kind"] == "decision"][:40]
        for request_id, node_id in enumerate(decisions):
            self.session.request(1000 + request_id, "query_node_evs", node_id)
            self.session.request(2000 + request_id, "query_equity", node_id)
        for _ in range(2 * len(decisions)):
            reply = self.session.receive()
            self.assertTrue(reply["ok"], reply)
            if reply["requestId"] < 2000:
                self.check_decision(reply["node"], set(reply["node"]["state"]["board"]), evs_ready=True)
            else:
                values = [reply["equity"]["players"][player]["equity"] for player in ("hero", "villain")]
                if all(value is not None for value in values):
                    self.assertTrue(all(0 <= value <= 1 for value in values), values)
                    self.assertAlmostEqual(sum(values), 1, delta=1e-5)

    def check_decision(self, node, board, evs_ready=False):
        self.assertEqual(node["evsReady"], evs_ready)
        self.assertTrue(node["actions"])
        self.assertLessEqual(sum(hand["marginalReachMass"] for hand in node["hands"]), 1 + 1e-5)
        for hand in node["hands"]:
            self.assertFalse(set(hand["cards"]) & board)
            self.assertGreater(hand["inputRangeWeight"], 0)
            self.assertGreaterEqual(hand["ownReachWeight"], 0)
            self.assertGreaterEqual(hand["marginalReachMass"], 0)
            if hand["marginalReachMass"] > 0:
                self.assertEqual(len(hand["strategy"]), len(node["actions"]))
                self.assertAlmostEqual(sum(hand["strategy"]), 1, delta=1e-5)
                self.assertTrue(all(0 <= p <= 1 for p in hand["strategy"]))
                if evs_ready:
                    self.assertTrue(math.isfinite(hand["nodeStrategyEv"]))
                    continue
            self.assertIsNone(hand["nodeStrategyEv"])

    def test_eof_drains_many_queued_ev_requests(self):
        root = self.session.ready["rootNodeId"]
        count = 200
        for request_id in range(count):
            self.session.request(request_id, "query_node_evs" if request_id % 2 == 0 else "query_node", root)
        replies, code = self.session.finish()
        self.assertEqual(code, 0, b"".join(self.session.stderr).decode(errors="replace")[-2000:])
        self.assertEqual(sorted(reply["requestId"] for reply in replies), list(range(count)))
        self.assertTrue(all(reply["ok"] for reply in replies))

    # KI-1 (tests/daily/README.md): nlohmann's parse error quotes the invalid bytes, the reply
    # cannot be serialized, and the exception escapes the per-request handler, so the service
    # reports "failed" and exits 1. These tests pin that exact shape; once they fail with
    # "KI-1 appears fixed", replace them with tests expecting an error reply and a live session.
    def assert_ki1_ends_session(self, raw):
        self.session.write(raw)
        message = self.session.receive()
        self.assertEqual(message.get("event"), "failed", f"KI-1 appears fixed: {message}")
        self.assertIn("UTF-8", message["message"])
        replies, code = self.session.finish()
        self.assertEqual((replies, code), ([], 1))

    def test_ki1_invalid_utf8_inside_string_ends_session(self):
        self.assert_ki1_ends_session(b'{"requestId": 3, "command": "query_node", "nodeId": 0, "x": "\xff\xfe"}')

    def test_ki1_invalid_utf8_outside_json_ends_session(self):
        self.assert_ki1_ends_session(b"\xff\xfe not json")

if __name__ == "__main__":
    unittest.main()
