"""Exercise the shipped JSON-lines CLI in real child processes, without external packages."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest


SERVICE = str(Path(sys.argv.pop(1)).resolve())
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class CliEndToEndTest(unittest.TestCase):
    def start(self, name, *, stdin=False, accuracy=None):
        scenario = json.loads((FIXTURES / f"{name}.json").read_text())
        process = subprocess.Popen(
            [SERVICE, "--stdin" if stdin else str(FIXTURES / f"{name}.json"),
             "--device=auto" if stdin else "--device=cpu"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env={**os.environ, "OMP_NUM_THREADS": "4"},
        )
        self.addCleanup(self.stop, process)
        if stdin:
            if accuracy is not None:
                scenario["accuracyPercent"] = accuracy
            self.send(process, scenario)
        first = self.receive(process)
        self.assertEqual(first["event"], "building_tree")
        self.assertEqual(first["totalIterations"], scenario["iterations"])
        previous_iterations = 0
        previous_elapsed = 0
        phases = set()
        while True:
            message = self.receive(process)
            self.assertIn(message["event"], ("solving", "ready"), message)
            phases.add(message["phase"])
            completed = message.get("completedIterations", message.get("iterations"))
            self.assertGreaterEqual(completed, previous_iterations)
            self.assertLessEqual(completed, scenario["iterations"])
            self.assertGreaterEqual(message["elapsedSeconds"], previous_elapsed)
            previous_iterations, previous_elapsed = completed, message["elapsedSeconds"]
            if message["event"] == "ready":
                break
        self.assertTrue({"training", "checking", "finalizing", "complete"} <= phases)
        self.assertEqual(message["estimatedRemainingSeconds"], 0)
        self.assertTrue(math.isfinite(message["accuracyPercent"]))
        self.assertGreaterEqual(message["accuracyPercent"], -0.0005)
        self.assertEqual(message["targetAccuracyPercent"], scenario.get("accuracyPercent", 0.01))
        self.assertGreater(message["nodeCount"], 0)
        return process, message

    @staticmethod
    def stop(process):
        if process.poll() is None:
            process.kill()
        process.communicate()

    @staticmethod
    def send(process, message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def receive(self, process):
        line = process.stdout.readline()
        self.assertTrue(line, "CLI closed stdout before completing the protocol")
        return json.loads(line)

    def query(self, process, node_id, request_id=1, command="query_node"):
        self.send(process, {"requestId": request_id, "command": command, "nodeId": node_id})
        reply = self.receive(process)
        self.assertEqual(reply["requestId"], request_id)
        self.assertTrue(reply["ok"], reply)
        return reply["node"]

    def follow(self, process, node, kind):
        action = next(action for action in node["actions"] if action["kind"] == kind)
        return self.query(process, action["nextNodeId"])

    def check_decision(self, node):
        self.assertEqual(node["kind"], "decision")
        self.assertTrue(node["hands"])
        for hand in node["hands"]:
            self.assertFalse(set(hand["cards"]) & set(node["state"]["board"]))
            self.assertGreater(hand["inputRangeWeight"], 0)
            self.assertGreaterEqual(hand["ownReachWeight"], 0)
            self.assertGreaterEqual(hand["marginalReachMass"], 0)
            if hand["marginalReachMass"] > 0:
                self.assertEqual(len(hand["strategy"]), len(node["actions"]))
                self.assertAlmostEqual(sum(hand["strategy"]), 1, delta=1e-5)
                self.assertTrue(all(0 <= p <= 1 for p in hand["strategy"]))
                self.assertTrue(math.isfinite(hand["nodeStrategyEv"]))
            else:
                self.assertIsNone(hand["nodeStrategyEv"])

    def test_file_solve_navigation_and_eof_drains_queries(self):
        process, ready = self.start("weighted-flop")
        self.assertLessEqual(ready["accuracyPercent"], 0.5)  # Existing solve tolerance: 0.01 chips / pot 2.
        self.assertEqual(ready["stopReason"], "iterationLimit")
        self.assertEqual(ready["iterations"], 200)
        root = self.query(process, ready["rootNodeId"])
        self.check_decision(root)
        self.assertEqual(root["actor"], "villain")
        self.assertEqual(root["state"]["board"], ["Ks", "9s", "2d"])
        self.assertEqual(root["state"]["pot"], 2)
        self.assertEqual(root["state"]["stacks"], {"hero": 4, "villain": 4})
        self.assertEqual([(a["kind"], a["amountTo"]) for a in root["actions"]], [("check", 0), ("bet", 1)])
        self.assertAlmostEqual(sum(h["marginalReachMass"] for h in root["hands"]), 1, delta=1e-5)
        for hand in root["hands"]:
            self.assertEqual(hand["ownReachWeight"], hand["inputRangeWeight"])

        facing = self.follow(process, root, "bet")
        folded = self.follow(process, facing, "fold")
        self.assertEqual(folded["kind"], "terminal")
        self.assertEqual(folded["result"], {"reason": "fold", "foldedBy": "hero"})
        chance = self.follow(process, facing, "call")
        for card, street, count in [("Qc", "turn", 49), ("Th", "river", 48)]:
            self.assertEqual(chance["kind"], "chance")
            outcomes = chance["outcomes"]
            self.assertEqual(len({o["card"] for o in outcomes}), count)
            self.assertFalse({o["card"] for o in outcomes} & set(chance["state"]["board"]))
            outcome = next(o for o in outcomes if o["card"] == card)
            node = self.query(process, outcome["nextNodeId"])
            self.check_decision(node)
            self.assertEqual(node["state"]["street"], street)
            self.assertEqual(node["state"]["board"][-1], card)
            self.assertEqual(node["state"]["pot"], 4)
            self.assertEqual(node["state"]["stacks"], {"hero": 3, "villain": 3})
            chance = self.follow(process, self.follow(process, node, "check"), "check")
        self.assertEqual(chance["kind"], "terminal")
        self.assertEqual(chance["result"], {"reason": "showdown"})

        # An invalid request fails alone among queued queries; EOF still answers every line.
        commands = {11: "unknown", 12: "query_equity", 13: "query_node"}
        for request_id, command in commands.items():
            self.send(process, {"requestId": request_id, "command": command, "nodeId": ready["rootNodeId"]})
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn("Device: CPU", stderr)
        replies = [json.loads(line) for line in stdout.splitlines()]
        self.assertEqual(len(replies), len(commands))
        responses = {reply["requestId"]: reply for reply in replies}
        self.assertEqual(responses.keys(), commands.keys())
        self.assertFalse(responses[11]["ok"])
        self.assertTrue(responses[11]["error"])
        for request_id in (12, 13):
            self.assertTrue(responses[request_id]["ok"], responses[request_id])
        self.assertEqual(responses[13]["node"], root)
        equity = responses[12]["equity"]
        self.assertEqual(equity["nodeId"], ready["rootNodeId"])
        values = [equity["players"][player]["equity"] for player in ("hero", "villain")]
        self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertAlmostEqual(sum(values), 1, delta=1e-5)

    def test_stdin_auto_device_stops_at_accuracy(self):
        process, ready = self.start("raise-flop", stdin=True, accuracy=100)
        self.assertEqual(ready["stopReason"], "accuracy")
        self.assertLess(ready["iterations"], 200)
        self.assertLessEqual(ready["accuracyPercent"], ready["targetAccuracyPercent"])
        root = self.query(process, ready["rootNodeId"])
        self.check_decision(root)
        raised = self.follow(process, self.follow(process, root, "bet"), "raise")
        self.assertEqual(raised["actor"], "villain")
        self.assertEqual(raised["state"]["pot"], 6)
        self.assertEqual(raised["state"]["stacks"], {"hero": 5, "villain": 7})
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(stdout, "")
        self.assertIn("Device:", stderr)

    def test_invalid_invocation_and_scenario_fail_as_json(self):
        for arguments, stdin in [([], ""), (["--stdin", "--device=invalid"], ""), (["--stdin"], "{}\n")]:
            with self.subTest(arguments=arguments):
                result = subprocess.run([SERVICE, *arguments], input=stdin, capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                message = json.loads(result.stdout)
                self.assertEqual(message["event"], "failed")
                self.assertTrue(message["message"])


if __name__ == "__main__":
    unittest.main()
