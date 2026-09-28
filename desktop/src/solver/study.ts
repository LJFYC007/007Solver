import { nodeFor, solutionFor, type PreflopChoice, type PreflopNode, type TableFormat } from "./catalog";
import {
    aggregatePreflopActions,
    comboCount,
    postflopOrder,
    preflopActionKind,
    preflopOutcome,
    probabilities,
    replayPreflop,
    type PostflopScenario,
    type PreflopState,
} from "./preflop";
import { type HandRow, equityRealization, evShare } from "./display";
import {
    aggregateActions,
    cardsKey,
    describeActions,
    formatNumber,
    handClassCombos,
    isForcedRunout,
    nodeEv,
} from "./strategy";
import type { DecisionAction, DecisionNode, Player, SolverNode, SolverStatus } from "./types";

export interface PreflopEntry {
    node: PreflopNode;
    history: PreflopChoice[];
    stack: number;
}
export interface DetailAction {
    id: string;
    kind: DecisionAction["kind"];
    label: string;
    size?: string;
    color: string;
    probability: number;
    /** Postflop only: the EV of taking the action, then following the strategy. */
    ev?: number;
}
/** A combination of the selected hand class, with the measures the Hands and Summary tabs show. */
export interface DetailCombo {
    cards: string[];
    /** Its hand class. */
    label: string;
    weight: number;
    actions: DetailAction[];
    /** Postflop only: the node strategy EV, or null until it is available. */
    ev?: number | null;
    /** Postflop only: undefined until the equity report arrives. */
    equity?: number | null;
    eqr?: number;
    evPot?: number;
    /** Postflop only: EVs in node action order, for Compare EV. */
    actionEvs?: number[];
    /** False when the filters leave the combination out. */
    matches: boolean;
    /** Shown instead of the strategy when there is none, and as the tile tooltip. */
    description: string;
}
export interface SpotSeat {
    position: string;
    stack: number;
    folded: boolean;
    acting?: boolean;
    committed?: number;
    combos?: number;
    ev?: number | null;
}

export const stopReasonLabel = (reason: "accuracy" | "iterationLimit") =>
    reason === "accuracy" ? "Target reached" : "Update limit reached";

const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

function buildPreflopTimeline(format: TableFormat, history: PreflopChoice[]): PreflopEntry[] {
    const solution = solutionFor(format);
    const entries: PreflopEntry[] = [];
    function add(node: PreflopNode, prefix: PreflopChoice[]) {
        const seat = replayPreflop(format, prefix).seats.find((s) => s.position === node.actor)!;
        entries.push({ node, history: prefix, stack: solution.stack - seat.committed });
    }
    for (let i = 0; i < history.length; i++) {
        const prefix = history.slice(0, i),
            node = nodeFor(format, prefix);
        if (node) add(node, prefix);
    }
    let prefix = history;
    for (let i = 0; i < solution.positions.length; i++) {
        const node = nodeFor(format, prefix);
        if (!node) break;
        add(node, prefix);
        if (!node.actions.some((a) => a.label === "Fold")) break;
        prefix = [...prefix, { actor: node.actor, action: "Fold" }];
    }
    return entries;
}

function preflopHandDetails(selected: string, range: Record<string, number>, preNode?: PreflopNode): DetailCombo[] {
    const selectedWeights = preNode ? probabilities(preNode, selected) : undefined;
    const selectedActions =
        preNode && selectedWeights
            ? preNode.actions.map((action, index) => ({
                  id: action.code,
                  kind: preflopActionKind(action.label),
                  label: action.label,
                  color: action.color,
                  probability: selectedWeights[index],
              }))
            : [];
    return handClassCombos(selected).map((cards) => ({
        cards,
        label: selected,
        weight: range[selected] ?? 0,
        actions: selectedActions,
        matches: true,
        description:
            preNode && !selectedWeights ? "No captured strategy" : `Range weight ${percent(range[selected] ?? 0)}`,
    }));
}

/**
 * The class's combinations at a postflop node, from the acting player's rows at its last decision, if
 * any; with filters, `shown` holds the keys of the rows they keep.
 */
export function postflopHandDetails(
    selected: string,
    board: readonly string[],
    rows: ReadonlyMap<string, HandRow>,
    shown?: ReadonlySet<string>,
    decision?: DecisionNode,
): DetailCombo[] {
    const pot = decision?.state.pot ?? 0;
    return handClassCombos(selected).map((cards) => {
        const row = rows.get(cardsKey(cards));
        const hand = row?.hand;
        const blocked = board.some((c) => cards.includes(c));
        return {
            cards,
            label: selected,
            weight: hand?.ownReachWeight ?? 0,
            actions:
                decision && hand && hand.ownReachWeight > 0 && hand.strategy.length === decision.actions.length
                    ? describeActions(decision).map((a) => ({
                          id: String(a.index),
                          kind: a.kind,
                          label: a.label,
                          size: a.size,
                          color: a.color,
                          probability: hand.strategy[a.index],
                          ev: hand.actionEvs[a.index],
                      }))
                    : [],
            ev: hand?.nodeStrategyEv,
            equity: row?.equity,
            eqr: equityRealization(row?.ev, row?.equity, pot),
            evPot: evShare(row?.ev, pot),
            actionEvs: hand?.actionEvs,
            // Hands outside the range keep their own description.
            matches: !row || !shown || shown.has(row.key),
            description: blocked
                ? "Blocked by board"
                : !hand
                  ? "Not in range"
                  : hand.ownReachWeight === 0
                    ? "Zero own reach"
                    : hand.marginalReachMass === 0
                      ? "No compatible opponent"
                      : `Reach ${percent(hand.ownReachWeight)}`,
        };
    });
}

function buildSpotSeats({
    state,
    stack,
    showPostflop,
    preActor,
    current,
    players,
    streetRoot,
}: {
    state: PreflopState;
    stack: number;
    showPostflop: boolean;
    preActor?: string;
    current?: SolverNode;
    players?: Record<Player, string>;
    streetRoot?: SolverNode;
}): SpotSeat[] {
    const seats: SpotSeat[] = state.seats.map((s) => ({
        position: s.position,
        stack: stack - s.committed,
        folded: s.folded,
        acting: !showPostflop && preActor === s.position,
        committed: showPostflop ? undefined : s.committed,
        combos: comboCount(s.range),
    }));
    if (current && players) {
        for (const player of ["hero", "villain"] as const) {
            const seat = seats.find((s) => s.position === players[player]);
            if (!seat) continue;
            seat.stack = current.state.stacks[player];
            seat.committed = (streetRoot?.state.stacks[player] ?? seat.stack) - seat.stack;
            seat.acting = current.kind === "decision" && current.actor === player;
            seat.combos = current.state.rangeCombos[player];
            if (current.kind === "terminal" && current.result.reason === "fold") {
                seat.folded = current.result.foldedBy === player;
                seat.ev = seat.folded ? 0 : current.state.pot;
            }
            const ev = current.kind === "decision" ? nodeEv(current, player) : undefined;
            if (ev !== undefined) seat.ev = ev;
        }
    }
    return seats;
}

type StudyMatrix =
    | { kind: "preflop"; node?: PreflopNode; range: Record<string, number> }
    | { kind: "postflop"; node: DecisionNode }
    | { kind: "empty"; title: string; description: string };

export function buildStudyView({
    format,
    history,
    preIndex,
    board,
    rangeSeat,
    path,
    activeIndex,
    scenario,
    solveState,
    solveError,
}: {
    format: TableFormat;
    history: PreflopChoice[];
    preIndex: number | null;
    board: string[];
    rangeSeat?: string;
    path: SolverNode[];
    activeIndex: number;
    scenario?: PostflopScenario;
    solveState: SolverStatus["state"];
    solveError: string;
}) {
    const solution = solutionFor(format);
    const fullState = replayPreflop(format, history);
    const shownHistory = history.slice(0, preIndex ?? history.length);
    const state = replayPreflop(format, shownHistory);
    const preNode = nodeFor(format, shownHistory);
    const outcome = preflopOutcome(fullState, solution.stack);
    const canPlayPostflop = outcome === "headsUp";
    const showPostflop = preIndex === null;
    const busy = solveState !== "idle" && solveState !== "ready" && solveState !== "failed";
    const players: Record<Player, string> = {
        hero: scenario?.heroPosition ?? "IP",
        villain: scenario?.villainPosition ?? "OOP",
    };
    const current = showPostflop ? path[activeIndex] : undefined;
    const forcedRunout = isForcedRunout(current);
    const decision =
        current?.kind === "decision"
            ? current
            : current?.kind === "chance" && !forcedRunout
              ? path
                    .slice(0, activeIndex)
                    .reverse()
                    .find((node): node is DecisionNode => node.kind === "decision")
              : undefined;
    const alive = state.seats.filter((seat) => !seat.folded);
    const order = postflopOrder(solution.positions);
    const [oop, ip] = fullState.seats
        .filter((seat) => !seat.folded)
        .sort((a, b) => order.indexOf(a.position) - order.indexOf(b.position));
    const raises = history.filter((choice) => preflopActionKind(choice.action) === "raise").length;
    const preActor = state.seats.find((seat) => seat.position === preNode?.actor);
    const range = (preActor ?? alive.find((seat) => seat.position === rangeSeat) ?? alive[0]).range;
    const available = preNode
        ? Object.fromEntries(Object.entries(range).filter(([hand]) => !!preNode.hands[hand]))
        : range;
    const postActions = decision ? aggregateActions(decision) : [];
    const actions: (DetailAction & { combos: number; index?: number; nextNodeId?: number })[] = showPostflop
        ? postActions.map((action) => ({
              ...action,
              id: String(action.index),
              nextNodeId: current?.kind === "decision" ? current.actions[action.index].nextNodeId : undefined,
          }))
        : preNode
          ? aggregatePreflopActions(preNode, range)
          : [];
    const streetRoot = current ? path.find((node) => node.state.street === current.state.street) : undefined;
    let matrix: StudyMatrix;
    if (showPostflop) {
        if (decision) matrix = { kind: "postflop", node: decision };
        else if (forcedRunout)
            matrix = {
                kind: "empty",
                title: "All-in · Betting complete",
                description: "No further betting decisions. Equity includes every legal runout.",
            };
        else if (current?.kind === "terminal")
            matrix = {
                kind: "empty",
                title: "Line complete",
                description:
                    current.result.reason === "fold"
                        ? `${players[current.result.foldedBy]} folded.`
                        : "Showdown reached.",
            };
        else
            matrix = {
                kind: "empty",
                title: solveState === "failed" ? "Unable to solve" : "Solving strategy",
                description:
                    solveState === "failed" ? solveError : "Preparing your strategy. Progress is shown on the right.",
            };
    } else if (!preNode && !state.complete)
        matrix = {
            kind: "empty",
            title: "This branch is unavailable",
            description: "The saved GTO Wizard catalog has no strategy for this action history.",
        };
    else matrix = { kind: "preflop", node: preNode, range };

    return {
        showPostflop,
        busy,
        players,
        preflop: {
            node: preNode,
            shownHistory,
            timeline: buildPreflopTimeline(format, history),
            complete: fullState.complete,
            pot: fullState.pot,
            canPlayPostflop,
            matchup: canPlayPostflop
                ? {
                      title: `${ip.position} vs ${oop.position}`,
                      oop: oop.position,
                      ip: ip.position,
                      detail: `${raises === 0 ? "Limped pot" : raises === 1 ? "Single-raised pot" : `${raises + 1}-bet pot`} · ${formatNumber(fullState.pot)} pot · ${formatNumber(solution.stack - ip.committed)} behind`,
                  }
                : undefined,
            resultLabel:
                outcome === "fold"
                    ? "Hand complete"
                    : outcome === "allIn"
                      ? "All-in · Betting complete"
                      : "Multiway pot",
        },
        strategy: {
            matrix,
            title: showPostflop
                ? decision
                    ? `${players[decision.actor]} strategy${current?.kind === "chance" ? " · last decision" : ""}`
                    : "Postflop"
                : preNode
                  ? `${preNode.actor} strategy`
                  : "Preflop ranges",
            combos: showPostflop
                ? decision?.hands.reduce((sum, hand) => sum + hand.ownReachWeight, 0)
                : comboCount(available),
            actor: showPostflop && decision ? players[decision.actor] : preNode?.actor,
            actions,
            // Postflop combinations need the equity report and filters (see postflopHandDetails).
            handDetails: showPostflop ? undefined : (selected: string) => preflopHandDetails(selected, range, preNode),
        },
        spot: {
            seats: buildSpotSeats({
                state: showPostflop ? fullState : state,
                stack: solution.stack,
                showPostflop,
                preActor: preNode?.actor,
                current,
                players: scenario ? players : undefined,
                streetRoot,
            }),
            pot: current?.state.pot ?? state.pot,
            board: current?.state.board ?? (state.complete ? board : []),
            toCall:
                current?.kind === "decision"
                    ? current.actions.find((action) => action.kind === "call")?.chipsCommitted
                    : preActor && preNode?.actions.some((action) => action.label === "Call")
                      ? Math.max(...state.seats.map((seat) => seat.committed)) - preActor.committed
                      : undefined,
            basePot: streetRoot?.state.pot,
            players: current ? players : undefined,
            showdown: forcedRunout || (current?.kind === "terminal" && current.result.reason === "showdown"),
            nodeId: current?.nodeId,
        },
        panel:
            state.complete && canPlayPostflop && (!showPostflop || busy || solveState === "failed")
                ? "solve"
                : forcedRunout || (outcome === "allIn" && state.complete) || current?.kind === "terminal"
                  ? "complete"
                  : "actions",
        boardTarget:
            current?.kind === "chance" && !forcedRunout
                ? "runout"
                : !showPostflop && canPlayPostflop
                  ? "flop"
                  : undefined,
    };
}
