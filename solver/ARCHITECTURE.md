# Solver Architecture

007 Solver is a Windows/macOS desktop application for solving a heads-up postflop game and browsing its strategy. The C++ service performs one solve at startup and then serves node queries.

```text
Scenario file -> game and ranges -> concrete decision tree
              -> CPU DCFR training -> strategy snapshot -> exploitability evaluation
              -> read-only result -> node analysis <-> NDJSON <-> Rust/Tauri <-> React
```

## Responsibilities

- **core** defines cards, chip amounts, heads-up player indices, streets, exact-combo ranges and the SKPokerEval adapter.
- **game** defines public betting state, legal actions, configured betting sizes, tree construction and terminal settlement. It does not own ranges.
- **engine** owns training state, exports a strategy snapshot and evaluates best responses.
- **analysis** queries a result, propagates range weights along the tree and computes fixed-strategy node EVs.
- **io** parses the scenario and expands hand-class notation into exact combos.
- **service** composes these steps and implements NDJSON. `main.cpp` handles the command-line argument and exit code.

Dependencies point from service into io and analysis, from analysis into engine, from engine into game, and from game into core. The io adapter also uses the game and range types. The core calculation library has no JSON, Tauri or React dependency.

## Game and card semantics

`GameSpec` contains the initial board, street, pot, remaining stacks, out-of-position player and betting sizes. The compiler accepts a three-card flop at the start of a betting round. The game is heads-up hold'em without rake, with a minimum bet of 0.1 chip. A zero remaining stack runs out the board without further betting.

`PlayerId` is a solver index, restricted to 0 and 1. The scenario and wire format map player 0 to hero and player 1 to villain. These are not persistent player identities or seats in an imported hand.

`Chips` stores integer tenths of a chip. The scenario has no big-blind or currency conversion. Pot and stack arithmetic use `Chips`; strategy probabilities, regrets and stored EVs use `float`. DCFR uses `double` for range normalization, reach and terminal masses. EVs are expressed in external chip units. Fractional sizing rounds to the nearest tenth, with exact halves rounded up.

Cards have one labeling throughout the program. `Board::CardAt()` preserves the input flop order followed by the dealt turn and river, and `Append()` adds the next card. `HoleCards` is an unordered two-card value; `Cards()` returns a deterministic order for evaluation and display. Packed representations are private to the card types.

`Range` stores exact combos with finite weights in [0, 1]. Missing combos have weight zero. `RangeNotationParser` expands `AA`, `AKs` and `AKo`, assigning the configured weight to every combo in the class. Blockers are applied when solving and analyzing a board.

## Building the tree

`GetLegalActions(state)` describes legal passive actions and the legal aggressive interval. `BettingAbstraction::SelectActions()` selects configured sizes inside that interval. `ApplyAction()` updates the public state and reports whether play continues, the round ends, or a player folds.

An action is fold, check, call, bet or raise, with an amount-to for that street. `ChipsCommitted()` derives the additional chips paid from the parent state. `IsAllIn()` derives whether the acting player empties their stack. The largest legal size can leave chips behind when the opponent has the shorter stack.

The default sizing offers check, a half-pot bet if legal, and the maximum bet. Facing a bet, it offers fold, call and the maximum raise. Explicit bet fractions use the current pot; raise fractions use the pot after calling and specify the raise-by. The maximum-size flags append that size in addition to explicit sizes, without duplicates.

`CompileGame()` creates an immutable `CompiledGame` containing the spec and a vector of `GameNode` values. Every action and every concrete dealt card creates its own child. Different histories remain different nodes even when they produce the same pot, stacks or final board. Every non-root node stores its parent and incoming edge index.

`CompiledGame` also owns precomputed showdown ranks, independent of player ranges and shared by training and evaluation. Reversed turn/river runouts share a rank row but retain distinct histories and node IDs. The table remains alive with the game.

`NodeId` is a vector index in this one tree. It also identifies the complete observable history, so analysis and the UI use the same ID. `InfoSetKey` is a node ID plus an exact private hand. IDs have meaning only with their tree and are not persistent solution or hand-record identities.

Chance edges list cards absent from the public board. The solver rejects outcomes blocked by either private hand. For a legal hand pair, each remaining card has probability `1 / (52 - boardCardCount - 4)`.

## Training, result and evaluation

`SolveProblem` holds a shared immutable game and the two ranges. `CpuDcfrSession` owns regrets, strategy sums and traversal buffers.

`CpuDcfrSession` uses DCFR with fixed `(alpha, beta, gamma) = (1.5, 0, 2)`. Each iteration updates one player across the full tree; successive `Run()` calls preserve player alternation and discount counters. OpenMP processes nodes in parallel within each depth.

DCFR own reach starts at one and includes only that player's action probabilities; opponent reach includes normalized range weights, opponent actions and chance. Terminal values are divided by each hand's fixed compatible root-opponent mass. Reach is not renormalized at each node.

The service explicitly trains and exports one `StrategySnapshot`. The training session goes out of scope immediately after export. The service then calls `EvaluateExploitability()` and constructs a `SolveResult`. The result keeps the problem, snapshot and single solve report alive for analysis.

The report records algorithm `dcfr`, execution backend `cpu`, completed full-player updates, training-loop time and best-response metrics. Training time excludes session initialization, snapshot export and evaluation.

Snapshots store node offsets, sorted exact hands and a contiguous probability buffer. DCFR exports all board-compatible hands in each decision's acting range in node/hand order, using a uniform distribution when strategy sums are zero. The public entry constructor also accepts unsorted fixed policies. Both construction paths check decision-node identity, blockers, action counts, duplicate entries and finite non-negative probabilities summing to one within `1e-5`. `FindStrategy()` returns a borrowed probability pointer or null for a missing entry; its length is the node's action count, and destruction, move or assignment of the snapshot invalidates it. `StrategyOrUniform()` returns an owning vector and explicitly supplies a uniform strategy for entries absent from a supplied fixed policy. A result and its snapshot must refer to the same compiled game.

`EvaluateExploitability()` computes each player's best response against the fixed snapshot. Exploitability is half the sum of their best-response values. Evaluation is a separate operation from training.

The iteration budget is fixed. The service logs exploitability to stderr and reports ready after evaluation; it does not enforce a convergence threshold.

## Payoff and analysis

`CompiledGame::CalculateTerminalSettlement()` determines the winner or tie and the contributions made since a specified start node, using precomputed ranks for showdowns. `NetPayoffFromStart()` awards the pot share and subtracts those later contributions. DCFR uses the same ranks and payoff rules for its terminal hand vectors.

Two consumers use different baselines:

- Solver utility starts at the tree root, subtracts half the initial pot from player 0's net payoff, and negates for player 1. These utilities sum to zero.
- Node strategy EV starts at the queried node. The pot already there is dead money; the two players' EVs sum to that pot.

`AnalysisSession` assembles a `NodeReport`. `ReachCalculator` caches the reach data for queried nodes, deriving it from parent edges. `CalculateNodeHandEv()` evaluates the fixed strategy from the queried node, conditioned on the selected hand and its compatible opponent reach.

Each reported hand carries:

| Field | Meaning |
|---|---|
| `inputRangeWeight` | Original exact-combo weight |
| `ownReachWeight` | Original weight times this player's action probabilities; excludes opponent actions and chance |
| `marginalReachMass` | Marginal of the joint legal hand-pair mass, including both strategies, chance and blockers |
| `nodeStrategyEv` | Expected net payoff from this node under the snapshot |

The initial joint profile is normalized over legal hand pairs. Products and normalization totals use `double` before conversion to `float`, so small positive input weights do not all underflow before normalization. The profile is not renormalized at each queried node. Opponent masses for a selected hand remain unnormalized until used to calculate its conditional EV.

A hand with zero joint reach has an empty strategy and null node EV in the report. It may still have positive own reach. The UI distinguishes this from a combo excluded by the board or absent from the input range. Matrix fill uses own reach weights; action aggregation uses own reach weights only for hands with an available strategy. With no available strategy, the UI displays an empty state.

Queries compute the requested node's hand EVs synchronously. The service serves one request at a time; it does not support cancellation, re-solving or switching results.

## Desktop boundary

The service writes one JSON message per stdout line; diagnostics go to stderr. Startup events are `building_tree`, `solving` and `ready`, or `failed`. Ready provides `iterations`, `nodeCount` and `rootNodeId`.

A `query_node` request carries `requestId` and `nodeId`. Success returns a node with `nodeId`, state, and decision actions or chance outcomes containing `nextNodeId`. The JSON adapter formats cards and chip amounts. Invalid requests receive an error without ending the query loop; errors with no usable request ID use 0.

Rust manages the child process and pending requests and passes node JSON through to React. TypeScript describes the node shape; Rust checks the response envelope rather than duplicating the complete node DTO. Protocol changes must be coordinated across the C++ serializer, Rust envelope and TypeScript consumer.

## Current limits

The full concrete tree, tabular training state and exact evaluation limit the size of practical scenarios. More betting branches and deeper stacks grow the tree; wider ranges increase training and analysis costs.

DCFR allocates dense root-hand rows for all nodes, trading higher memory use for regular full traversals and parallel execution.

The desktop uses one bundled flop scenario. There is no scenario editor, hand-history importer, solution storage, GPU implementation or neural model.
