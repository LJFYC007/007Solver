# Solver contracts

This document records semantics shared across modules. Implementation details live with the code; setup and verification commands are in the [README](../README.md).

## Boundaries and ownership

The calculation layers are `analysis -> engine -> game -> core`. The `io` adapter parses scenarios; `service` composes the solve and query lifecycle. JSON and presentation concerns stay outside core, game and engine.

Each service process owns one solve. It checks exploitability on the current average strategy without consuming training state, stopping at the target or update limit. It then exports one strategy snapshot, releases training state, and retains the result for queries. The service owns progress and final metrics; `SolveResult` owns only the problem and snapshot. The final metrics belong to that same strategy; checkpoint normalization must match snapshot export. A result and its snapshot must refer to the same compiled game. Sparse solver snapshots use uniform play for omitted exact-hand entries. Node views and borrowed strategy data must not outlive their owners.

Each CPU training iteration updates one player, alternating across successive `Run` calls. Reach propagation, value backup and strategy accumulation use the policy at entry to each decision node. Training may batch disjoint chance subtrees within an update: ancestor regrets remain unchanged while workers replay their entry paths and solve those subtrees. Workers own separate scratch buffers and result rows; after they finish, ancestor updates consume the results in the original tree/action order. Scheduling must not change reach propagation, floating-point backup order, or the alternating-player iteration boundary.

The desktop scopes requests and caches to a solution generation. [App](../desktop/src/App.tsx) owns solve replacement and cancellation; [useStudyWorkspace](../desktop/src/hooks/useStudyWorkspace.ts) routes study edits through that invalidation before applying local changes. Navigation is suspended before invalidation. Replacing or cancelling a solve invalidates pending requests and cached reports; responses from an old generation must not update the current view.

Service stdout is one protocol JSON message per line; diagnostics go to stderr. Protocol changes must agree across the [C++ serializer](service/JsonAdapter.cpp), [Rust envelope](../desktop/src-tauri/src/solver_protocol.rs) and [TypeScript types](../desktop/src/solver/types.ts). Rust passes node and equity payloads through.

The service accuracy target is exploitability as a percentage of the initial pot (`0.01` means `0.01%`). Ready reports actual completed updates and whether accuracy or the iteration limit stopped training. Elapsed time includes preparation, checks and export. Remaining seconds require a measured decreasing convergence trend and include future checks; an insufficient or unstable trend has no estimate. The iteration limit alone is not a convergence forecast. The UI interpolates time between samples and reserves completion for Ready.

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

Every scenario supplies a complete `bettingTree` that configures `flop`, `turn` and `river` independently. Each street requires `bet` and `raise` arrays of positive percentages (up to two decimal places); an empty array disables that aggression. `maxRaises` and `allInSpr` are also required. Both players use the same settings. The desktop initializes bets and raises to `[50]`, `maxRaises` to `2` and `allInSpr` to `0.15`. See [the correctness fixture](../tests/fixtures/weighted-flop.json).

Bet percentages use the current pot. Raise percentages specify the additional raise above a call, as a fraction of the pot after calling. Amounts round half up to tenths of a chip, then clamp to the legal minimum and effective-stack maximum.

`maxRaises` accepts 0–2 and counts raises per street, excluding the opening bet. At the cap only call/fold remain; reaching the cap never forces a shove. `allInSpr` replaces a configured bet/raise with the effective-stack maximum when the remaining effective stack divided by the pot **after the opponent calls** is at most the threshold. Zero disables this replacement. Equal resulting sizes are deduplicated. No separate all-in size is added. The application, correctness fixtures and benchmark use the same scenario settings and build rules.

All-in remains a property of a bet, raise or call, not a separate action kind. Covering the opponent's stack does not necessarily set `isAllIn` for the acting player.

## Preflop input

The [catalog](../resources/gtowizard-preflop/) is the source of application and test ranges. Missing branches and missing continuation hands have no inferred fallback.

The [preflop adapter](../desktop/src/solver/preflop.ts) forms ranges by multiplying each player's own captured action frequencies, normalized for display rounding. Folded contributions remain in the pot. It maps IP to hero, OOP to villain, and one source bb to one scenario chip; it does not carry folded-player card-removal correlations into the heads-up solve.

Fixture subsets and reduced pot/stacks are recorded in each input's `rangeSource`. Follow [the reference update workflow](../tests/README.md#updating-inputs-and-references) when changing them.
