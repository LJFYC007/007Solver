# Solver contracts

## Boundaries and ownership

Dependencies flow `analysis -> engine -> game -> core`. JSON and presentation stay outside core, game and engine.

A result and its snapshot must refer to the same compiled game. Sparse snapshots use uniform play for omitted exact hands. Node and strategy views borrow their owners' storage.

## Training and memory

Each [DcfrSession](engine/DcfrSession.h) iteration updates one player, alternating across successive `Run` calls. Checkpoints preserve training state; export consumes it. Reach propagation, value backup and strategy accumulation use the policy at entry to each decision node. Parallel subtree updates must retain ancestor entry policies, use separate scratch, and preserve tree/action backup order and the alternating-player boundary.

CPU and GPU use float arithmetic and can differ with accumulation and traversal order. `DcfrSession::EvaluateCheckpoint` uses CPU evaluation for final metrics and GPU checkpoints that reach the supplied stopping target. Callers must supply that target whenever a checkpoint can end training. Final metrics must describe the exported strategy and use the same cumulative-strategy normalization.

The [GPU plan](engine/gpu/GpuPlan.h) may reuse descendant scratch only after backup, retaining live ancestor reaches and child-root values. Each player's reach is stored in the slot written by the nearest ancestor that changed it, so a decision rewrites only its actor's side. Terminal evaluation reads only the opponent's reach, which may remain in an ancestor's slot; the updating player's reach need not be stored for terminals. Its batching target is not a hard memory limit: individual street regions and retained ancestors can exceed it, and the whole tree's regrets and cumulative strategies remain resident.

Allocation estimates include checkpoint scratch and GPU readback while training remains resident; later navigation and EV caches are excluded. See [CPU sizing](engine/MemoryEstimate.cpp) and [GPU sizing](engine/GpuDcfrSession.cpp).

## Service and desktop lifecycle

Each service process owns one solve, stopping at the accuracy target or update limit. Only `Ready` marks completion; elapsed time includes preparation, checks and export. Remaining-time estimates require a measured decreasing convergence trend and include future checks; insufficient or unstable trends have no estimate.

Desktop requests and caches belong to a solution generation. Replacing or cancelling a solve invalidates pending requests and reports; stale responses must not update the view. Line and board edits invalidate before applying changes. Solver settings are drafts for the next solve and preserve the current result and submitted scenario.

Navigation returns strategy and reach without waiting for EVs. Decision reports have `evsReady: false` until `query_node_evs` returns; pending null EVs must not be aggregated as zero. A single EV worker owns access to the [node evaluator](engine/StrategyEvaluator.h), borrowing the analysis session's immutable result. Navigation's [reach cache](analysis/ReachCalculator.h) stays on the input thread; later queries may invalidate borrowed reaches. EOF drains accepted EV requests before destroying the session.

Service stdout is one JSON message per line; diagnostics go to stderr. Responses may arrive out of order and must be matched by request ID. Protocol changes must agree across the [C++ serializer](service/JsonAdapter.cpp), [Rust envelope](../desktop/src-tauri/src/solver_protocol.rs) and [TypeScript types](../desktop/src/solver/types.ts); Rust passes node and equity payloads through. For `iterations` and `accuracyPercent`, see [CLI usage](../README.md#setup-and-development).

## Identity and amounts

- A `NodeId` identifies one complete action and concrete-card history within one tree, not a topology index or an identity across solves. Shared betting topology and unordered turn/river rank tables must not merge ordered strategy histories.
- An `InfoSetKey` pairs a node with an exact private hand. Preserve original suits and input flop order with [Card.h](core/Card.h) helpers.
- `Chips` stores integer tenths of a chip. External amounts and EVs use chip units. Core has no big-blind conversion.
- Player 0 is hero and player 1 is villain on the wire, not persistent seat identities.
- `amountTo` is the player's total commitment on the current street; `chipsCommitted` is the additional payment from the parent state.

## Values and reach

| Quantity | Meaning |
|---|---|
| Solver utility | Root net payoff minus half the initial pot; the two utilities sum to zero |
| Node strategy EV | Expected pot share minus contributions after the queried node; the two EVs sum to that node's pot |
| Showdown equity | Win probability plus half the tie probability over legal runouts from current ranges, independent of future betting |

| [Reach field](analysis/NodeReport.h) | Meaning |
|---|---|
| `inputRangeWeight` | Original exact-combo weight |
| `ownReachWeight` | Input weight times only this player's action probabilities |
| `marginalReachMass` | Marginal of legal joint hand-pair mass, including both players' actions, chance and blockers |

Joint reach is normalized at the root, not per node: multiply both own reaches and history chance probability, divide by the root's legal-pair mass, and apply private/public blockers at the queried board. Own reach excludes chance. This assumes independent input ranges.

A hand can have positive own reach and zero joint reach; its report then has no strategy and a null node EV. Conditional hand EV uses compatible opponent mass, not the hand's own reach. A chance node without joint support still lists all public-board-compatible cards.

Each chance outcome has probability `1 / (52 - boardCardCount - 4)` for a compatible private-hand pair. Called all-ins still include every legal remaining runout. Exploitability measures the chosen betting tree and does not account for omitted action sizes.

## Betting tree

The scenario's [`bettingTree`](io/ScenarioLoader.cpp) requires all three streets and applies to both players. Empty `bet` or `raise` arrays disable that aggression.

Bet percentages use the current pot. Raise percentages specify the additional raise above a call, as a fraction of the pot after calling. Amounts round half up to tenths of a chip, then clamp to the legal minimum and effective-stack maximum.

`maxRaises` counts raises per street, excluding the opening bet; at the cap only call/fold remain. `allInSpr` replaces a configured bet/raise with the effective-stack maximum when the remaining effective stack divided by the pot **after the opponent calls** is at most the threshold. Zero disables replacement; equal resulting sizes are deduplicated.

All-in is a property of a bet, raise or call; no separate action kind or size is added. `isAllIn` requires exhausting the acting player's stack, even when covering the opponent.

## Preflop input

The [preflop adapter](../desktop/src/solver/preflop.ts) multiplies each player's own captured action frequencies, normalized for display rounding. Missing branches or continuation hands have no fallback. It maps IP to hero, OOP to villain, and one source bb to one scenario chip. Folded contributions remain in the pot; folded-player card-removal correlations are excluded.
