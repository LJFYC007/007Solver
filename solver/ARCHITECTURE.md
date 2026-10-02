# Solver contracts

## Boundaries and ownership

Dependencies flow `analysis -> engine -> game -> core`. JSON and presentation stay outside core, game and engine.

A result and its snapshot must refer to the same compiled game. Sparse snapshots use uniform play for omitted exact hands. Node and strategy views borrow their owners' storage.

## Training

Each [DcfrSession](engine/DcfrSession.h) iteration updates one player, alternating across successive `Run` calls. Checkpoints preserve training state; export consumes it.

CPU and GPU run the same algorithm and may differ only in summation order and FMA rounding:

- **Reach.** An update propagates only the opponent's reach (range weight, the opponent's action probabilities and chance), using the policy at entry to each decision node.
- **Pruning.** Subtrees without opponent reach are skipped exactly: their values and increments are zero, and an acting decision keeps its stamp, the index of its actor's last unpruned update.
- **State.** Both devices share the 16-bit layout of [HandTraversalData.h](engine/HandTraversalData.h) and the stochastic rounding of [GpuQuantize.h](engine/gpu/GpuQuantize.h); a hand an update does not touch keeps its values exactly.
- **Discounts.** Both DCFR discounts apply lazily ([UpdateWeights](engine/HandTraversalData.h)). Cumulative strategies sum t² · reach · policy during the opponent's updates (Burch et al.'s alternating average), so only per-hand normalization gives the average policy; a hand without reach at a decision is left untouched there.
- **Terminals.** Both devices use one formulation ([HandEvaluation.h](engine/HandEvaluation.h)). Training and GPU checkpoint evaluation take all-in runouts from precomputed win/loss rows; CPU evaluation walks them exactly.
- **CPU parallelism.** CPU walks run chance subtrees in parallel but must retain ancestor entry policies, use separate scratch and preserve backup order, so results do not depend on the worker count; training must also keep the alternating-player boundary.

Near-tied regrets flip regret matching under rounding differences, so independently trained CPU and GPU trajectories diverge within a few updates. Compare devices one update at a time from a state shared through `DcfrSession::ReadTrainingState` and `WriteTrainingState`, between sessions with equal completed iterations, by decoded values (`HandTraversalData::Decode`): equal values can have different encodings.

`DcfrSession::EvaluateCheckpoint` uses CPU evaluation for final metrics and for GPU checkpoints that reach the supplied stopping target, which callers must supply whenever a checkpoint can end training. Final metrics must describe the exported strategy with the same cumulative-strategy normalization.

## GPU plan

The [GPU plan](engine/gpu/GpuPlan.h) supports at most 16 actions per decision (`kMaxActions`); the CPU has no such limit. [GpuTypes.h](engine/gpu/GpuTypes.h), [GpuPlan.cpp](engine/gpu/GpuPlan.cpp) and [GpuKernels.inc](engine/gpu/GpuKernels.inc) document the scratch slots, records, passes, Backup inlining and stream pipelining. Changes must keep these invariants:

- A node's children occupy consecutive slots and its record holds only the first (`Node::link`); every kernel and Backup inlining rely on it.
- Descendant scratch is reused only after backup, keeping live ancestor reaches and child-root values.
- Slot intervals order passes but cannot detect overlap within one. A Terminal block stages every row it reads before writing, and writes only rows that no other block of its pass reads.
- Children without opponent reach are unflagged and their values are stale: parents substitute zero, and acting decisions read the flag before touching regrets.
- Node stamps live in two halves selected by update parity: a pass reads the half its player's previous update wrote and writes the other, so tiles of one node never race.
- Cross-stream predecessors derive from the slot intervals each pass reads and writes (`Plan::predecessors`), and CUDA captures them as graph edges. `Executor::Update` may return before the device finishes; `Synchronize`, downloads and root values wait.
- The batching target keeps leaf lanes' reach and values mostly in the GPU's L2; it is not a memory limit. Street regions and retained ancestors can exceed it, and the whole tree's regrets and cumulative strategies stay resident.

## Memory

Allocation estimates combine host and device memory and impose no limit. They include checkpoint scratch and GPU readback while training remains resident; later queries are excluded. After training, the [analysis session](analysis/AnalysisSession.h) keeps the traversal tables of the last decision whose EVs it evaluated, about 96 bytes per node of its subtree, for later queries of that decision. See [CPU sizing](engine/MemoryEstimate.cpp) and [GPU sizing](engine/GpuDcfrSession.cpp).

## Service and desktop lifecycle

Each service process owns one solve and stops at the accuracy target or update limit, or when a message write fails because its client has closed stdout. Only `Ready` marks completion.

Interface requests and caches belong to a solution generation. Replacing or cancelling a solve invalidates pending requests and reports; stale responses must not update the view. Line and board edits invalidate before applying changes. Solver settings are drafts for the next solve and preserve the current result and submitted scenario.

A [bridge](../desktop/src-tauri/src/solver_bridge.rs) owns one solve at a time: one in the desktop app, and one per page session in the [web server](../desktop/src-tauri/src/server.rs), whose routes mirror the Tauri commands. Cancelling keeps the session in both. [client.ts](../desktop/src/solver/client.ts) must keep both transports equivalent; it serializes solves and cancels because HTTP can reorder them.

Server solves wait as `queued` for the server's single GPU permit and hold it until `Ready`, when export has freed the training memory but not the CUDA context, or until their service exits. A session ends when its page unloads (a `pagehide` that does not keep the page in the back/forward cache, after which a late solve request is refused) or when the server releases it for idleness or memory; its later requests fail.

Service stdout carries one JSON message per line; diagnostics go to stderr. Protocol changes must agree across the [C++ serializer](service/JsonAdapter.cpp), [Rust envelope](../desktop/src-tauri/src/solver_protocol.rs) and [TypeScript types](../desktop/src/solver/types.ts). Rust forwards a successful query response's only field besides `requestId` and `ok`; each report kind is a `QueryKind` in the envelope and the same key of `SolverReports` in [client.ts](../desktop/src/solver/client.ts).

## Identity and amounts

- A `NodeId` identifies one complete action and concrete-card history within one tree, not a topology index or an identity across solves. Shared betting topology and unordered turn/river rank tables must not merge ordered strategy histories.
- An `InfoSetKey` pairs a node with an exact private hand. Preserve original suits and input flop order with [Card.h](core/Card.h) helpers.
- `Chips` stores integer tenths of a chip. External amounts and EVs use chip units. Core has no big-blind conversion.
- Player 0 is hero and player 1 is villain on the wire, not persistent seat identities.
- `amountTo` is the player's total commitment on the current street; `chipsCommitted` is the additional payment from the parent state.

## Values and reach

| Quantity | Meaning |
|---|---|
| Solver utility | Root net payoff minus half the initial pot; the two utilities sum to minus the rake paid |
| Node strategy EV | Expected pot share after rake minus contributions after the queried node; the two range-averaged EVs sum to that node's pot minus expected rake |
| Action EV | The actor's node strategy EV had it taken that action (its contribution included) and then followed the strategy; their strategy-weighted mean is the node strategy EV |
| Range EV | A player's node strategy EVs averaged over joint reach (a decision report's `rangeEvs`); the waiting player's is the node's pot less the actor's and the rake the actor's evaluation expects from the node on |
| Showdown equity | Win probability plus half the tie probability over legal runouts from current ranges, independent of future betting and rake |

| [Reach field](analysis/NodeReport.h) | Meaning |
|---|---|
| `inputRangeWeight` | Original exact-combo weight |
| `ownReachWeight` | Input weight times only this player's action probabilities |
| `marginalReachMass` | Marginal of legal joint hand-pair mass, including both players' actions, chance and blockers |

Joint reach is normalized at the root, not per node: multiply both own reaches and the history's chance probability, divide by the root's legal-pair mass, and apply private/public blockers at the queried board. Own reach excludes chance. This assumes independent input ranges.

A hand can have positive own reach and zero joint reach; its report then has no strategy or action EVs and a null [node EV](engine/StrategyEvaluator.h), which must not be aggregated as zero. Conditional hand EV uses compatible opponent mass, not the hand's own reach. Node reports carry the actor's hand and action EVs and both players' range EVs; the waiting player's hand EVs come from `query_opponent_ev`, which costs another subtree evaluation. A chance node without joint support still lists all public-board-compatible cards.

Each chance outcome has probability `1 / (52 - boardCardCount - 4)` for a compatible private-hand pair. Called all-ins still include every legal remaining runout. Exploitability measures the chosen betting tree and does not account for omitted action sizes.

For raked games, `exploitability` is NashConv / 2: `((BR0 - V0) + (BR1 - V1)) / 2`, where each best response and fixed-policy EV use the same average strategy and legal joint range weights. The zero-rake shortcut `(BR0 + BR1) / 2` is valid only when policy utilities sum to zero. The reported value measures unilateral improvement; DCFR's zero-sum equilibrium guarantee does not extend to raked games.

## Rake

The scenario's `rakePercent` and `rakeCap` (in scenario chips) default to zero; either zero disables rake. Every postflop terminal pays `min(matched pot × rakePercent / 100, rakeCap)`. Matched pot excludes the difference between current-street contributions, so an uncalled bet is returned without rake. The winner pays the rake, or both players split it in a tie. Rake stays floating point rather than rounding to the tree's tenth-chip betting increment. Terminal reports, and chance reports of forced runouts, carry the `rake` they pay ([TerminalRake](game/TerminalSettlement.h)); clients must take payouts and showdown pots after it.

The initial pot must be gross of this hand's rake: settlement applies the cap once across the whole pot, including preflop contributions. Jackpot fees, cash drops, rakeback and per-street deductions need separate source rules and are not encoded by these two fields.

## Betting tree

In the scenario's [`bettingTree`](io/ScenarioLoader.cpp), the `oop` and `ip` sizes apply to that player's own bets and raises. `heroActsFirst` determines which player is OOP. Empty `bet` or `raise` arrays disable that player's aggression. `maxRaises` and `allInSpr` are shared.

Bet percentages use the current pot. Raise percentages specify the additional raise above a call, as a fraction of the pot after calling. Amounts round half up to tenths of a chip, then clamp to the legal minimum and effective-stack maximum.

`maxRaises` counts raises per street, excluding the opening bet; at the cap only call/fold remain. `allInSpr` replaces a configured bet/raise with the effective-stack maximum when the remaining effective stack divided by the pot **after the opponent calls** is at most the threshold. Zero disables replacement; equal resulting sizes are deduplicated.

All-in is a property of a bet, raise or call; no separate action kind or size is added. `isAllIn` requires exhausting the acting player's stack, even when covering the opponent.

## Preflop input

The [preflop adapter](../desktop/src/solver/preflop.ts) replaces ranges with each saved node's source-precision incoming ranges and multiplies the actor's by the chosen action's captured frequencies, normalized for display rounding; [sync-preflop-fixtures.py](../scripts/sync-preflop-fixtures.py) derives test inputs the same way. Missing branches or continuation hands have no fallback. It maps IP to hero, OOP to villain, and one source bb to one scenario chip. Catalog `rakePercent` and `rakeCap` pass unchanged to the scenario; the cap is in bb and must come from the source solution rather than its stake label. Folded contributions remain in the pot; folded-player card-removal correlations are excluded.
