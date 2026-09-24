import type { DecisionAction, DecisionNode, HandStrategy, SolverNode } from "./types";

export const isForcedRunout = (node?: SolverNode) =>
    node?.kind === "chance" && (node.state.stacks.hero === 0 || node.state.stacks.villain === 0);

export interface ActionInfo {
    color: string;
    index: number;
    isAllIn: boolean;
    kind: DecisionAction["kind"];
    label: string;
    /** Rounded % pot (after calling, for raises) of sized bets and raises that are not all-in. */
    size?: string;
}

export interface AggregatedAction extends ActionInfo {
    combos: number;
    probability: number;
}

export interface HandGroup {
    actions: AggregatedAction[];
    hands: HandStrategy[];
    ownReachWeight: number;
}

export const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"] as const;
export const SUITS = ["s", "h", "d", "c"] as const;
export const HAND_CLASSES = RANKS.flatMap((_, row) => RANKS.map((__, column) => matrixLabel(row, column)));
export const SUIT_SYMBOLS: Record<string, string> = {
    c: "♣",
    d: "♦",
    h: "♥",
    s: "♠",
};

function actionPotPercent(action: DecisionAction, node: DecisionNode): number {
    const toCall = action.kind === "raise" ? (node.actions.find((a) => a.kind === "call")?.chipsCommitted ?? 0) : 0;
    return ((action.chipsCommitted - toCall) / (node.state.pot + toCall)) * 100;
}

function actionLabel(action: DecisionAction): string {
    if (action.kind === "fold") return "Fold";
    if (action.kind === "check") return "Check";
    if (action.kind === "call") return "Call";
    // Matches the catalog's preflop labels, such as "Allin 100".
    return `${action.isAllIn ? "Allin" : action.kind === "bet" ? "Bet" : "Raise"} ${formatNumber(action.amountTo)}`;
}

// GTO Wizard's Tadara theme colors anchored at pot percentages; sizes between anchors blend.
const SIZE_STOPS: readonly (readonly [number, string])[] = [
    [25, "var(--action-smallest)"],
    [33, "var(--action-small)"],
    [50, "var(--action-medium)"],
    [100, "var(--action-large)"],
    [150, "var(--action-overbet)"],
];
const LAST_STOP = SIZE_STOPS.length - 1;
// Minimum ramp distance between sizes of one node, so each size keeps a distinct shade.
const SIZE_GAP = 0.6;
const PASSIVE_COLORS: Partial<Record<DecisionAction["kind"], string>> = {
    fold: "var(--action-fold)",
    check: "var(--action-check)",
    call: "var(--action-call)",
};

/** Position of a pot percentage on the size ramp, from 0 to the last stop. */
export function sizePosition(percent: number): number {
    for (let i = 1; i <= LAST_STOP; i++) {
        const [lower] = SIZE_STOPS[i - 1];
        const [upper] = SIZE_STOPS[i];
        if (percent <= upper) return Math.max(0, i - 1 + (percent - lower) / (upper - lower));
    }
    return LAST_STOP;
}

function rampColor(position: number): string {
    const index = Math.min(Math.floor(position), LAST_STOP - 1);
    const share = Math.round((position - index) * 100);
    return share === 0
        ? SIZE_STOPS[index][1]
        : `color-mix(in srgb, ${SIZE_STOPS[index + 1][1]} ${share}%, ${SIZE_STOPS[index][1]})`;
}

/** Colors for ascending size positions, pushed apart where sizes are too close to tell apart. */
export function spreadSizeColors(positions: number[]): string[] {
    const spread = positions.slice();
    for (let i = 1; i < spread.length; i++) spread[i] = Math.max(spread[i], spread[i - 1] + SIZE_GAP);
    if (spread.length) spread[spread.length - 1] = Math.min(spread[spread.length - 1], LAST_STOP);
    for (let i = spread.length - 2; i >= 0; i--) spread[i] = Math.min(spread[i], spread[i + 1] - SIZE_GAP);
    return spread.map((position) => rampColor(Math.max(0, position)));
}

function actionColors(node: DecisionNode): string[] {
    const colors = node.actions.map((action) => PASSIVE_COLORS[action.kind] ?? "");
    for (const kind of ["bet", "raise"] as const) {
        const sized = node.actions
            .map((action, index) => ({ action, index }))
            .filter(({ action }) => action.kind === kind)
            .sort((first, second) => first.action.amountTo - second.action.amountTo);
        const shades = spreadSizeColors(
            sized.map(({ action }) => (action.isAllIn ? LAST_STOP : sizePosition(actionPotPercent(action, node)))),
        );
        sized.forEach(({ index }, i) => (colors[index] = shades[i]));
    }
    return colors;
}

const passivity = (kind: DecisionAction["kind"]) =>
    kind === "bet" || kind === "raise" ? 0 : kind === "call" || kind === "check" ? 1 : 2;
const nodeActions = new WeakMap<DecisionNode, readonly ActionInfo[]>();

/** Node actions from most to least aggressive: larger bets and raises first, then call or check, then fold. */
export function describeActions(node: DecisionNode): readonly ActionInfo[] {
    let actions = nodeActions.get(node);
    if (!actions) {
        const colors = actionColors(node);
        actions = node.actions
            .map((action, index) => ({
                color: colors[index],
                index,
                isAllIn: action.isAllIn,
                kind: action.kind,
                label: actionLabel(action),
                size:
                    (action.kind === "bet" || action.kind === "raise") && !action.isAllIn
                        ? `${Math.round(actionPotPercent(action, node))}%`
                        : undefined,
            }))
            .sort(
                (first, second) =>
                    passivity(first.kind) - passivity(second.kind) ||
                    node.actions[second.index].amountTo - node.actions[first.index].amountTo,
            );
        nodeActions.set(node, actions);
    }
    return actions;
}

export function aggregateActions(node: DecisionNode, hands: HandStrategy[] = node.hands): AggregatedAction[] {
    const handsWithStrategy = hands.filter((hand) => hand.strategy.length === node.actions.length);
    const totalOwnReachWeight = handsWithStrategy.reduce((sum, hand) => sum + Math.max(0, hand.ownReachWeight), 0);
    if (totalOwnReachWeight <= 0) return [];

    return describeActions(node).map((action) => {
        const weightedProbability = handsWithStrategy.reduce(
            (sum, hand) => sum + Math.max(0, hand.ownReachWeight) * hand.strategy[action.index],
            0,
        );

        return {
            ...action,
            combos: weightedProbability,
            probability: weightedProbability / totalOwnReachWeight,
        };
    });
}

export function toHandClass(cards: readonly string[]): string {
    if (cards.length !== 2) return cards.join(" ");

    const [first, second] = cards;
    const firstRank = first[0];
    const secondRank = second[0];
    const firstIndex = RANKS.indexOf(firstRank as (typeof RANKS)[number]);
    const secondIndex = RANKS.indexOf(secondRank as (typeof RANKS)[number]);

    if (firstIndex < 0 || secondIndex < 0) return cards.join(" ");
    if (firstRank === secondRank) return `${firstRank}${secondRank}`;

    const highRank = firstIndex < secondIndex ? firstRank : secondRank;
    const lowRank = firstIndex < secondIndex ? secondRank : firstRank;
    const suited = first[1] === second[1];
    return `${highRank}${lowRank}${suited ? "s" : "o"}`;
}

export function groupHands(node: DecisionNode): Map<string, HandGroup> {
    const rawGroups = new Map<string, HandStrategy[]>();
    const blockedCards = new Set(node.state.board);

    for (const hand of node.hands) {
        const label = toHandClass(hand.cards);
        const group = rawGroups.get(label);
        if (group) group.push(hand);
        else rawGroups.set(label, [hand]);
    }

    return new Map(
        [...rawGroups.entries()].map(([label, hands]) => {
            const availableComboCount = handClassCombos(label).filter((cards) =>
                cards.every((card) => !blockedCards.has(card)),
            ).length;

            return [
                label,
                {
                    actions: aggregateActions(node, hands),
                    hands,
                    ownReachWeight:
                        hands.reduce((sum, hand) => sum + hand.ownReachWeight, 0) / Math.max(1, availableComboCount),
                },
            ];
        }),
    );
}

export function handClassCombos(label: string): string[][] {
    const firstRank = label[0];
    const secondRank = label[1];

    if (firstRank === secondRank) {
        const pairs: string[][] = [];
        for (let secondSuit = 1; secondSuit < SUITS.length; secondSuit += 1) {
            for (let firstSuit = 0; firstSuit < secondSuit; firstSuit += 1) {
                pairs.push([`${firstRank}${SUITS[firstSuit]}`, `${secondRank}${SUITS[secondSuit]}`]);
            }
        }
        return pairs;
    }

    if (label.endsWith("s")) return SUITS.map((suit) => [`${firstRank}${suit}`, `${secondRank}${suit}`]);

    return SUITS.flatMap((firstSuit) =>
        SUITS.filter((secondSuit) => secondSuit !== firstSuit).map((secondSuit) => [
            `${firstRank}${firstSuit}`,
            `${secondRank}${secondSuit}`,
        ]),
    );
}

export function matrixLabel(row: number, column: number): string {
    if (row === column) return `${RANKS[row]}${RANKS[column]}`;
    if (row < column) return `${RANKS[row]}${RANKS[column]}s`;
    return `${RANKS[column]}${RANKS[row]}o`;
}

export function strategyGradient(actions: readonly { color: string; probability: number }[]): string | undefined {
    const visibleActions = actions.filter((action) => action.probability > 0);
    const total = visibleActions.reduce((sum, action) => sum + action.probability, 0);
    if (total <= 0) return undefined;

    let position = 0;
    const stops: string[] = [];
    for (const action of visibleActions) {
        const start = position;
        position += (action.probability / total) * 100;
        stops.push(`${action.color} ${start.toFixed(2)}%`, `${action.color} ${position.toFixed(2)}%`);
    }

    return `linear-gradient(90deg, ${stops.join(", ")})`;
}

export function formatNumber(value: number): string {
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
}
