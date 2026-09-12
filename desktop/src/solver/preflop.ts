import { actionColor, HAND_CLASSES, handClassCombos } from "./strategy";
import {
    nodeFor,
    solutionFor,
    type PreflopAction,
    type PreflopChoice,
    type PreflopNode,
    type TableFormat,
} from "./catalog";

export const probabilities = (node: PreflopNode, hand: string) => {
    const row = node.hands[hand];
    const total = row?.reduce((sum, value) => sum + value, 0) ?? 0;
    return total > 0 ? row.map((value) => value / total) : undefined;
};
export const comboCount = (range: Record<string, number>) =>
    Object.entries(range).reduce((sum, [hand, weight]) => sum + weight * handClassCombos(hand).length, 0);

export function aggregatePreflopActions(node: PreflopNode, range: Record<string, number>) {
    const totals = node.actions.map(() => 0);
    let total = 0;
    for (const [hand, weight] of Object.entries(range)) {
        const row = probabilities(node, hand);
        if (!row) continue;
        const combos = weight * handClassCombos(hand).length;
        total += combos;
        row.forEach((probability, index) => {
            totals[index] += combos * probability;
        });
    }
    return node.actions.map((action, index) => ({
        id: action.code,
        label: action.label,
        color: preflopActionColor(action),
        probability: total > 0 ? totals[index] / total : 0,
        combos: totals[index],
    }));
}

export interface PreflopSeat {
    position: string;
    committed: number;
    folded: boolean;
    range: Record<string, number>;
}
export interface PreflopState {
    seats: PreflopSeat[];
    pending: string[];
    pot: number;
    complete: boolean;
}

export function replayPreflop(format: TableFormat, history: PreflopChoice[]): PreflopState {
    const solution = solutionFor(format);
    const seats: PreflopSeat[] = solution.positions.map((position) => ({
        position,
        committed:
            solution.ante + (position === "SB" ? solution.smallBlind : position === "BB" ? solution.bigBlind : 0),
        folded: false,
        range: Object.fromEntries(HAND_CLASSES.map((hand) => [hand, 1])),
    }));
    let pending = [...solution.positions];
    let highest = solution.bigBlind + solution.ante;
    for (let index = 0; index < history.length; index++) {
        const choice = history[index];
        const node = nodeFor(format, history.slice(0, index));
        if (!node || node.actor !== choice.actor || pending[0] !== choice.actor)
            throw new Error("This action history has no captured strategy.");
        const actionIndex = node.actions.findIndex((action) => action.label === choice.action);
        if (actionIndex < 0) throw new Error("This action is absent from the captured solution.");
        const seat = seats.find((item) => item.position === choice.actor)!;
        // Empty source cells carry no usable strategy; never invent a continuation for them.
        seat.range = Object.fromEntries(
            Object.entries(seat.range).flatMap(([hand, weight]) => {
                const probability = probabilities(node, hand)?.[actionIndex];
                return probability && weight > 0 ? [[hand, weight * probability]] : [];
            }),
        );
        pending = pending.slice(1);
        if (choice.action === "Fold") seat.folded = true;
        else if (choice.action === "Call") seat.committed = Math.min(solution.stack, highest);
        else if (choice.action === "Check") {
            if (seat.committed !== highest) throw new Error("Cannot check facing a bet.");
        } else {
            const amountTo = Number(choice.action.split(" ")[1]);
            if (!Number.isFinite(amountTo) || amountTo <= highest || amountTo > solution.stack)
                throw new Error("Invalid source raise size.");
            seat.committed = amountTo;
            highest = amountTo;
            const seatIndex = seats.indexOf(seat);
            pending = [...seats.slice(seatIndex + 1), ...seats.slice(0, seatIndex)]
                .filter((item) => !item.folded && item.committed < solution.stack)
                .map((item) => item.position);
        }
        if (seats.filter((item) => !item.folded).length === 1) pending = [];
        // With no opponent able to act, an all-in needs only the outstanding call/fold decision.
        if (
            pending.length === 1 &&
            seats.filter((item) => !item.folded && item.committed < solution.stack).length === 1 &&
            seats.find((item) => item.position === pending[0])!.committed === highest
        )
            pending = [];
    }
    return {
        seats,
        pending,
        pot: seats.reduce((sum, seat) => sum + seat.committed, 0),
        complete: pending.length === 0,
    };
}

export interface PostflopScenario {
    board: string;
    heroPosition: string;
    villainPosition: string;
    heroActsFirst: boolean;
    initialPot: number;
    heroStack: number;
    villainStack: number;
    algorithm: "dcfr";
    iterations: number;
    ranges: Record<string, Record<string, number>>;
    rangeSource: { solution: TableFormat; history: PreflopChoice[] };
}
export function postflopScenario(
    format: TableFormat,
    history: PreflopChoice[],
    board: string[],
    iterations: number,
): PostflopScenario {
    const state = replayPreflop(format, history);
    const solution = solutionFor(format);
    const alive = state.seats.filter((seat) => !seat.folded);
    if (!state.complete || alive.length !== 2) throw new Error("Select a completed heads-up preflop line.");
    if (alive.some((seat) => seat.committed >= solution.stack))
        throw new Error("Betting is complete after the all-in call.");
    if (board.length !== 3 || new Set(board).size !== 3 || board.some((card) => !/^[AKQJT2-9][shdc]$/.test(card)))
        throw new Error("Select three different flop cards.");
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 2147483647)
        throw new Error("Iterations must be a positive integer.");
    const order = ["SB", "BB", ...solution.positions.filter((position) => position !== "SB" && position !== "BB")];
    alive.sort((a, b) => order.indexOf(a.position) - order.indexOf(b.position));
    const [villain, hero] = alive;
    for (const seat of alive) {
        if (
            !Object.entries(seat.range).some(
                ([hand, weight]) =>
                    weight > 0 && handClassCombos(hand).some((cards) => cards.every((card) => !board.includes(card))),
            )
        )
            throw new Error(`${seat.position} has no hands available on this flop.`);
    }
    return {
        board: board.join(" "),
        heroPosition: hero.position,
        villainPosition: villain.position,
        heroActsFirst: false,
        initialPot: state.pot,
        heroStack: solution.stack - hero.committed,
        villainStack: solution.stack - villain.committed,
        algorithm: "dcfr",
        iterations,
        ranges: { [hero.position]: hero.range, [villain.position]: villain.range },
        rangeSource: { solution: format, history },
    };
}

export function preflopActionColor(action: PreflopAction): string {
    const kind = action.code === "F" ? "fold" : action.code === "C" ? "call" : action.code === "X" ? "check" : "raise";
    return actionColor({ kind, isAllIn: action.code === "RAI" });
}
