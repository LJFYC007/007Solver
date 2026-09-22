# Solver contracts

This document records semantics shared across modules. Implementation details live with the code; setup and verification commands are in the [README](../README.md).

## Boundaries and ownership

The calculation layers are `analysis -> engine -> game -> core`. The `io` adapter parses scenarios; `service` composes the solve and query lifecycle. JSON and presentation concerns stay outside core, game and engine.

Each service process owns one solve, stopping at the accuracy target or update limit. Checkpoints do not consume training state; export releases it. Final metrics must describe the exported strategy, using the same normalization. A result and its snapshot must refer to the same compiled game. Sparse snapshots use uniform play for omitted exact-hand entries. Node views and borrowed strategy data must not outlive their owners.

Each training iteration updates one player, alternating across successive `Run` calls. [DcfrSession](engine/DcfrSession.h) owns that schedule, DCFR discounts, progress timing and the consuming export boundary; CPU and GPU backends execute one player update at a time. Reach propagation, value backup and strategy accumulation use the policy at entry to each decision node. Parallel subtree updates must retain ancestor entry policies, use separate scratch, and preserve tree/action backup order and the alternating-player boundary.

[HandBoardData](engine/HandBoardData.h) owns the range-specific hand indices and rank tables for one exact public board. [HandTraversalData](engine/HandTraversalData.h) adds the query or training root's nodes and utility baseline; mutable training state belongs to each session. Rank tables may share an unordered turn/river pair, but strategy state remains distinct for every ordered history. Both backends share snapshot normalization and compaction.

The [GPU plan](engine/gpu/GpuPlan.h) defines buffer upload sources, allocation sizes, initialization passes and player-update passes for both executors and the memory estimate; binding indices and pass stages are shared with the kernels in [GpuTypes.h](engine/gpu/GpuTypes.h). Initialization completes before player updates begin. Upload sources borrow the plan and must be copied before it is destroyed. The plan may reuse descendant scratch only after backup, retaining live ancestor reaches and child-root values. Its batching target is not a hard memory limit: individual street regions and retained ancestors can exceed it, and the whole tree's regrets and cumulative strategies remain resident.

CPU and GPU use float arithmetic; their accumulation and traversal orders can produce different results. CUDA and Metal share kernel arithmetic. `DcfrSession::EvaluateCheckpoint` owns certification: final checkpoints always use CPU evaluation, and provisional GPU checkpoints reaching the supplied stopping target receive independent CPU evaluation before returning. Callers supply a stopping target whenever a checkpoint can end training. CPU evaluation uses the same cumulative-strategy normalization as snapshot export.

Desktop requests and caches belong to a solution generation. Replacing or cancelling a solve invalidates pending requests and cached reports; old responses must not update the current view. Line and board edits must invalidate before applying changes. Solver settings are drafts for the next solve and preserve the current result and submitted scenario.

Node navigation returns strategy and reach without waiting for EV evaluation. Decision reports use `evsReady: false` until a separate `query_node_evs` response supplies the complete report. While pending, null hand EVs must not be aggregated as zero. A single EV worker exclusively accesses the analysis session's node evaluator, which borrows that session's immutable solve result and retains at most one exact board's hand tables per street. Changing boards replaces that street's cached table; each query rebuilds its nodes, utility baseline, history reach and traversal scratch. The input thread owns a separate reach cache containing only the root-to-current-query path. Moving to another branch releases the abandoned suffix before computing new reaches; a later reach query may invalidate borrowed reach references. Responses may arrive out of request order and are matched by request ID. EOF drains accepted EV requests before destroying the analysis session.

Service stdout is one protocol JSON message per line; diagnostics go to stderr. Protocol changes must agree across the [C++ serializer](service/JsonAdapter.cpp), [Rust envelope](../desktop/src-tauri/src/solver_protocol.rs) and [TypeScript types](../desktop/src/solver/types.ts). Rust passes node and equity payloads through.

The service accuracy target is exploitability as a percentage of the initial pot (`0.01` means `0.01%`). Ready reports actual completed updates and the stop reason. Elapsed time includes preparation, checks and export. Remaining time requires a measured decreasing convergence trend and includes future checks; an unstable or insufficient trend has no estimate. Only Ready marks completion.

## Identity and amounts

- A `NodeId` identifies one complete action and concrete-card history within one tree. Shared betting topology does not merge strategy nodes. IDs are neither stored-topology indices nor persistent identities across solves.
- An `InfoSetKey` is a node plus an exact private hand. Preserve original suits and input flop order; use [Card.h](core/Card.h) helpers.
- `Chips` stores integer tenths of a chip. External amounts and EVs use chip units. Core has no big-blind conversion.
- Player 0 is hero and player 1 is villain on the wire. These are solver indices, not persistent seat identities.
- `amountTo` is the player's total commitment on the current street; `chipsCommitted` is the additional payment from the parent state. An effective-stack maximum need not empty the acting player's stack.

## Values and reach

Solver utility and displayed EV have different baselines:

| Quantity | Meaning |
|---|---|
| Solver utility | Root net payoff minus half the initial pot; the two utilities sum to zero |
| Node strategy EV | Expected pot share minus contributions after the queried node; the two EVs sum to that node's pot |
| Showdown equity | Win probability plus half the tie probability over legal runouts from current ranges, independent of future betting |

Reach fields in [NodeReport.h](analysis/NodeReport.h) are also distinct:

| Field | Meaning |
|---|---|
| `inputRangeWeight` | Original exact-combo weight |
| `ownReachWeight` | Input weight times only this player's action probabilities |
| `marginalReachMass` | Marginal of legal joint hand-pair mass, including both players' actions, chance and blockers |

Joint reach is normalized at the root, not at every node. A hand can have positive own reach and zero joint reach; its report then has no strategy and a null node EV. Conditional hand EV uses compatible opponent mass, not the hand's own reach.

Each chance outcome has probability `1 / (52 - boardCardCount - 4)` for a compatible private-hand pair. Called all-ins still include every legal remaining runout. Exploitability measures the chosen betting tree and does not account for omitted action sizes.

## Betting tree

Each scenario supplies all three streets of `bettingTree`; both players use the same settings. Empty `bet` or `raise` arrays disable that aggression. See [the input fixture](../tests/fixtures/weighted-flop.json) and [parser](io/ScenarioLoader.cpp) for required fields and validation.

Bet percentages use the current pot. Raise percentages specify the additional raise above a call, as a fraction of the pot after calling. Amounts round half up to tenths of a chip, then clamp to the legal minimum and effective-stack maximum.

`maxRaises` counts raises per street, excluding the opening bet. At the cap only call/fold remain; reaching the cap never forces a shove. `allInSpr` replaces a configured bet/raise with the effective-stack maximum when the remaining effective stack divided by the pot **after the opponent calls** is at most the threshold. Zero disables this replacement. Equal resulting sizes are deduplicated; no separate all-in size is added.

All-in remains a property of a bet, raise or call, not a separate action kind. Covering the opponent's stack does not necessarily set `isAllIn` for the acting player.

## Preflop input

The [catalog](../resources/gtowizard-preflop/) is the source of application and test ranges. Missing branches and missing continuation hands have no inferred fallback.

The [preflop adapter](../desktop/src/solver/preflop.ts) forms ranges by multiplying each player's own captured action frequencies, normalized for display rounding. Folded contributions remain in the pot. It maps IP to hero, OOP to villain, and one source bb to one scenario chip; it does not carry folded-player card-removal correlations into the heads-up solve.

Fixture subsets and reduced pot/stacks are recorded in each input's `rangeSource`. Follow [the reference update workflow](../tests/README.md#updating-inputs-and-references) when changing them.
