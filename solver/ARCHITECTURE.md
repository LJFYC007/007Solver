# Solver Architecture

007 Solver is a Windows/macOS desktop application for solving a heads-up postflop game and browsing its strategy. The desktop starts with captured preflop strategies; each C++ service process performs one requested postflop solve and then serves node queries.

```text
Scenario file or first stdin line -> game and ranges -> compact betting topology + memory estimate
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

The default sizing offers check and a half-pot bet; facing a bet it offers fold, call and a half-pot raise. Bet fractions use the current pot; raise fractions use the pot after calling and specify the raise-by, added to the call amount-to. Each street allows one non-final raise, then only the effective-stack maximum for further aggression. `PublicState::raiseCount` resets on each street. Explicit sizes clamp to the legal minimum/maximum, retaining the short effective-stack all-in exception, and duplicate amounts are removed. The optional maximum-size flags append an independent maximum bet/raise; both default to false. This is an action abstraction: exploitability measures convergence within the chosen tree, not errors from omitted sizes.

`CompileGame()` creates an immutable `CompiledGame` containing the spec and a compact vector of betting-history templates. A chance template has one next-street continuation; all concrete deals share that betting topology. Templates record checked logical subtree spans and active traversal counts. `GameNode` is a small value view with the concrete board and parent edge. `GetNode()` decodes a logical preorder ID; `Child()` derives a child view directly. Views borrow immutable topology and must not outlive the game. References taken through temporary views or temporary edges must not escape the full expression. Different action/card histories remain different strategy nodes even when they produce the same pot, stacks or final board.

`CompiledGame` also owns precomputed showdown ranks, independent of player ranges and shared by training and evaluation. Reversed turn/river runouts share a rank row but retain distinct histories and node IDs. The table remains alive with the game.

`NodeId` is a logical preorder index in this one tree, including unmaterialized forced-runout histories. It is not an index into the stored topology. Span arithmetic uses 64-bit intermediates and rejects trees beyond the existing signed 32-bit node limit. It also identifies the complete observable history, so analysis and the UI use the same ID. `InfoSetKey` is a node ID plus an exact private hand. IDs have meaning only with their tree and are not persistent solution or hand-record identities.

Chance edges list cards absent from the public board. The solver rejects outcomes blocked by either private hand. For a legal hand pair, each remaining card has probability `1 / (52 - boardCardCount - 4)`.

## Training, result and evaluation

`SolveProblem` holds a shared immutable game and the two ranges. `CpuDcfrSession` owns regrets, strategy sums and traversal buffers. `HandTraversal` supplies the concrete CPU layout and hand-vector kernels used by both training and snapshot evaluation; it owns no regrets or averaging state. Each operation builds its layout for the relevant subtree, retaining original `NodeId` values alongside compact local buffer indices. Its prepared hand/rank tables and action-major strategy offsets are reused throughout that operation. Active nodes are stored in logical preorder; forced-runout chance nodes have no materialized traversal descendants. Snapshots and training offsets remain independent for every concrete decision history.

`CpuDcfrSession` uses DCFR with fixed `(alpha, beta, gamma) = (1.5, 0, 2)`. Each iteration updates one player across the full tree; successive `Run()` calls preserve player alternation and discount counters. Traversal is depth-first. Each operation reuses reach, child-value and double-precision accumulation rows indexed by depth. A decision retains its old strategy and all child action values until its own regret/average update. At the first ordinary chance boundary on each path, OpenMP dynamically distributes concrete-card subtrees over a bounded worker pool; descendants run serially within each worker. There is no nested parallelism. Parent values are reduced in original action/card order after workers finish. The configured worker limit is capped at 49 and can be reduced by the OpenMP runtime; each allocated worker owns one reusable workspace.

Shared value backups and DCFR regret/average updates traverse actions outside and contiguous hands inside. Per-call stack scratch holds up to 1,326 exact hands, avoiding allocation in the hot loops and sharing no mutable scratch between workers. Each hand retains its action accumulation order; shared value sums stay in `double` until the final `float` store, positive-regret sums stay in `float`, and best-response backups retain maximum selection.

DCFR own reach starts at one and includes only that player's action probabilities; opponent reach includes normalized range weights, opponent actions and chance. Terminal values are divided by each hand's fixed compatible root-opponent mass. Reach is not renormalized at each node.

Reach is propagated into the next depth row before descending. Evaluation propagates only opponent reach; training also propagates own reach for averaging. At a called all-in, exact runout evaluation streams the remaining public cards with blockers and the 45/44 conditional chance denominators, reusing shared showdown ranks and bounded scratch. It preserves the operation root's utility baseline and fixed compatible masses; it does not start an independent turn/river solve. The logical runout chance/terminal nodes remain queryable. Shared terminal evaluation skips the compatible-mass baseline for showdowns with exactly zero tie utility. Folds and subtrees with nonzero tie utility retain the baseline calculation and small-mass precision fallback.

The service explicitly trains and exports one `StrategySnapshot`. The training session goes out of scope immediately after export. The service then calls `EvaluateExploitability()` and constructs a `SolveResult`. The result keeps the problem, snapshot and single solve report alive for analysis.

The report records algorithm `dcfr`, execution backend `cpu`, completed full-player updates, training-loop time and best-response metrics. Training time excludes session initialization, snapshot export and evaluation.

Snapshots store sparse decision-node blocks, sorted exact hands and a contiguous probability buffer. Both node and hand lookup use sorted indices; there is no array sized by the full logical node count. DCFR exports all board-compatible hands in each decision's acting range in node/hand order, using a uniform distribution when strategy sums are zero. The public entry constructor also accepts unsorted fixed policies. Both construction paths check decision-node identity, blockers, action counts, duplicate entries and finite non-negative probabilities summing to one within `1e-5`. `FindStrategy()` returns a borrowed probability pointer or null for a missing entry; its length is the node's action count, and destruction, move or assignment of the snapshot invalidates it. `StrategyOrUniform()` returns an owning vector and explicitly supplies a uniform strategy for entries absent from a supplied fixed policy. A result and its snapshot must refer to the same compiled game.

`EvaluateExploitability()` packs the fixed snapshot once and computes each player's best response with the shared hand-vector kernels. Opponent reach is propagated once per public node, terminal values use sorted showdown ranks and blocker masses, and each responding hand chooses its maximum-valued action after aggregating opponent hands. A fixed compatible root-opponent mass normalizes each hand throughout the pass; root aggregation restores the joint range weighting. Exploitability is half the sum of the best-response values. Snapshot evaluation runs serially, packs the selected subtree's policy once and reuses the same bounded depth-first workspace as training. Layout, policy and workspace are released when the operation finishes. It does not retain training state.

The iteration budget is fixed. The service logs exploitability to stderr and reports ready after evaluation; it does not enforce a convergence threshold.

## Payoff and analysis

`CompiledGame::CalculateTerminalSettlement()` determines the winner or tie and the contributions made since a specified start node, using precomputed ranks for showdowns. `NetPayoffFromStart()` awards the pot share and subtracts those later contributions. DCFR uses the same ranks and payoff rules for its terminal hand vectors.

Two consumers use different baselines:

- Solver utility starts at the tree root, subtracts half the initial pot from player 0's net payoff, and negates for player 1. These utilities sum to zero.
- Node strategy EV starts at the queried node. The pot already there is dead money; the two players' EVs sum to that pot.

`AnalysisSession` assembles a `NodeReport`. `ReachCalculator` caches the reach data for queried nodes, deriving it from parent edges. `EvaluateNodeStrategyEvs()` evaluates all acting-player hands in one fixed-policy subtree pass, using the same kernels as training and best response, with strategy-weighted backups. Its starting opponent weights are proportional to input weights times that opponent's ancestor action probabilities, computed in `double` and filtered against the queried board. The selected hand's own reach and the earlier chance-probability factor cancel under conditioning. Each hand's compatible starting mass remains the denominator for the whole subtree. Terminal contributions are measured from the queried node; adding half that node's pot to the traversal's zero-sum values restores node net EV for either player. Existing joint reach still determines report eligibility, including null EV for hands with zero joint reach.

Each reported hand carries:

| Field | Meaning |
|---|---|
| `inputRangeWeight` | Original exact-combo weight |
| `ownReachWeight` | Original weight times this player's action probabilities; excludes opponent actions and chance |
| `marginalReachMass` | Marginal of the joint legal hand-pair mass, including both strategies, chance and blockers |
| `nodeStrategyEv` | Expected net payoff from this node under the snapshot |

The initial joint profile is normalized over legal hand pairs. Products and normalization totals use `double` before conversion to `float`, so small positive input weights do not all underflow before normalization. The profile is not renormalized at each queried node. Opponent masses for a selected hand remain unnormalized until used to calculate its conditional EV.

A hand with zero joint reach has an empty strategy and null node EV in the report. It may still have positive own reach. The UI distinguishes this from a combo excluded by the board or absent from the input range. Matrix fill uses own reach weights; action aggregation uses own reach weights only for hands with an available strategy. With no available strategy, the UI displays an empty state.

Queries compute the requested node's hand EVs synchronously. The service serves one request at a time and retains one result. Desktop cancellation or re-solving terminates that process; a new solve uses a new process.

## Preflop input boundary

`resources/gtowizard-preflop/` stores observed GTO Wizard action matrices, solution settings and source histories. `catalog.json` holds shared metadata; each `6max/<actor>.json` or `8max/<actor>.json` holds that acting position's nodes. TypeScript's `catalog.ts` indexes them by solution and full history, rejecting duplicate histories and position/solution mismatches. `preflop.ts` replays a selected line. It normalizes each displayed frequency row, multiplies only the acting player's own range by that action, and drops hands absent from that source matrix. It does not infer missing strategies. JSON catalog details stay outside C++ core/game/engine.

Replay tracks folded seats, total commitments and pending actors. Raises are amount-to, calls match the highest commitment, and folded blinds/contributions remain in the pot. A completed two-player line plus three distinct cards becomes a postflop scenario. Postflop seat order determines OOP/villain (1) and IP/hero (0), independent of who opened. The adapter maps one source bb to one scenario chip; core chip arithmetic is unchanged. Exact combos share their class weight, and the engine applies public-board blockers. The imported heads-up model does not retain folded-player card-removal correlations.

The catalog is frozen at 208 captured decision spots (87 for 6-max and 121 for 8-max); unavailable branches cannot start an invented continuation. The UI retains its preflop history and board while showing a solve. The CLI example and all range-containing fixtures derive from this catalog; test reductions are explicitly recorded in each scenario's `rangeSource` metadata.

## Desktop boundary

`solver_service <path>` reads a scenario file. `solver_service --stdin` reads the first line as scenario JSON, then uses the remaining stdin lines for node queries. `io::ReadScenario` supplies the same parser and validation to both paths.

The service writes one JSON message per stdout line; diagnostics go to stderr. Startup events are `building_tree`, `solving` and `ready`, or `failed`. Ready provides `iterations`, logical `nodeCount` and `rootNodeId`. Solving and ready events also carry an `estimate` containing `logicalNodes`, `topologyNodes`, `traversalNodes`, `strategyEntries`, `peakBytes` and the configured `workers` limit. Rust checks and forwards these fields; the solve panel displays estimated memory beside progress.

A `query_node` request carries `requestId` and `nodeId`. Success returns a node with `nodeId`, state, and decision actions or chance outcomes containing `nextNodeId`. The JSON adapter formats cards and chip amounts. Invalid requests receive an error without ending the query loop; errors with no usable request ID use 0.

A `query_equity` request uses the same request/node IDs and returns `equity` containing both players' overall showdown equity and exact-hand own weights/equities. `AnalysisSession::QueryEquity` uses the current `ReachFor` own ranges, enumerates public runouts, and calls the existing core evaluator. Sorted ranks and per-card blocker mass give each hand's win/tie/compatible mass without traversing future betting actions. Identical hands receive the double-blocker correction; small residual masses use direct compatible-hand summation. Per-hand equity divides by compatible opponent mass; overall equity weights that mass by the player's own weight. Empty joint support yields null equity. Node state also includes both players' board-filtered own-reach combo counts, so the UI need not reconstruct unavailable zero-joint strategies.

Rust starts idle and manages the child process and pending requests under one mutex. Starting or cancelling increments a generation, kills the old child and rejects its pending requests. Child events and queries must match the active generation; React resets its node cache for each generation. Rust passes node and equity JSON through to React. TypeScript describes the node shape; Rust checks the response envelope rather than duplicating the complete node DTO. Protocol changes must be coordinated across the C++ serializer, Rust envelope and TypeScript consumer.

## Current limits

Tabular strategy storage and exact traversal still limit practical scenarios. More betting branches and deeper stacks grow the logical tree; wider ranges increase training and analysis costs. Shared topology reduces structural storage but does not merge infosets. Regrets, strategy sums and current strategies still use dense root-hand strides at decision nodes, while reach/value workspace scales with depth, action count and worker limit rather than the full node count. Quantized strategy storage, suit-isomorphism reduction and CFR-D / safe subgame re-solving are not implemented.

After compiling the small topology, `EstimateCpuMemory()` counts root hands and uses topology multiplicities to estimate the peak before allocating the active layout or strategy tables. It includes layout/rank data, all three training tables, per-worker workspaces, snapshot export coexistence, evaluation, 12.5% headroom and a 64 MiB runtime allowance. Board-blocked decision slots count toward this conservative estimate. Later user-driven navigation caches are excluded; it is an estimate rather than a reservation or an OS memory guarantee.

The service rejects estimates exceeding its budget. An optional positive integer `memoryBudgetMiB` in scenario JSON sets an explicit budget. Otherwise Windows uses 75% of the smaller available physical/commit memory; macOS uses 75% of free plus inactive physical pages. If OS probing fails, the fallback budget is 3 GiB. Diagnostics report logical and stored sizes, estimated peak and budget. The imported ranges are never trimmed to fit memory.

The desktop offers a unified preflop-to-river Study workspace with shared inspector and board picker. History selection preserves the current solution. A changed preflop line clears the board and solution; a completed playable heads-up line opens the flop picker automatically. A changed flop invalidates the solution. Board history cards provide the edit entry point. A chance node with either remaining stack at zero is a forced runout: the UI ends the betting line without deleting chance outcomes or asking for more cards. Facing an all-in remains a decision until call/fold resolves it. The inspector shares a fixed-size detail area for Table and Equity chart; selecting a different hand only changes combo tiles. Its SVG uses the observed plot dimensions for rendering and pointer coordinates. At decision nodes, joint-marginal-weighted actor EV and current pot determine both players' EV; at forced showdowns, EV is equity times pot. EQR is explicitly defined as current-node EV / (equity × current pot). Ordinary chance EV is unavailable. React guards stale navigation/equity responses by solution generation and selected node. Multiway postflop solving, arbitrary range editing, hand-history import, solution storage, GPU implementation and neural models are not implemented.
