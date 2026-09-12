import type { DecisionAction, DecisionNode, HandStrategy, SolverNode } from "./types";

export const isForcedRunout = (node?: SolverNode) =>
    node?.kind === "chance" && (node.state.stacks.hero === 0 || node.state.stacks.villain === 0);

export interface AggregatedAction {
    combos: number;
    color: string;
    index: number;
    isAllIn: boolean;
    kind: DecisionAction["kind"];
    label: string;
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

export function actionLabel(action: Pick<DecisionAction, "amountTo" | "chipsCommitted" | "isAllIn" | "kind">): string {
    if (action.kind === "fold") return "Fold";
    if (action.kind === "check") return "Check";
    if (action.kind === "call") return "Call";
    if (action.isAllIn) return `Allin ${formatNumber(action.amountTo)}`;
    return `${action.kind === "bet" ? "Bet" : "Raise"} ${formatNumber(action.amountTo)}`;
}

export function actionColor(action: Pick<DecisionAction, "isAllIn" | "kind">): string {
    if (action.kind === "fold") return "var(--action-fold)";
    if (action.kind === "call") return "var(--action-call)";
    if (action.kind === "check") return "var(--action-check)";
    if (action.isAllIn) return "var(--action-allin)";
    if (action.kind === "bet" || action.kind === "raise") return "var(--action-raise)";
    return "var(--action-other)";
}

function actionDisplayOrder(action: Pick<DecisionAction, "isAllIn" | "kind">): number {
    if (action.isAllIn) return 0;
    if (action.kind === "bet" || action.kind === "raise") return 1;
    if (action.kind === "call" || action.kind === "check") return 2;
    return 3;
}

export function orderedActions(actions: readonly AggregatedAction[]): AggregatedAction[] {
    return [...actions].sort(
        (first, second) => actionDisplayOrder(first) - actionDisplayOrder(second) || first.index - second.index,
    );
}

export function aggregateActions(node: DecisionNode, hands: HandStrategy[] = node.hands): AggregatedAction[] {
    const handsWithStrategy = hands.filter((hand) => hand.strategy.length === node.actions.length);
    const totalOwnReachWeight = handsWithStrategy.reduce((sum, hand) => sum + Math.max(0, hand.ownReachWeight), 0);
    if (totalOwnReachWeight <= 0) return [];

    return node.actions.map((action, index) => {
        const weightedProbability = handsWithStrategy.reduce(
            (sum, hand) => sum + Math.max(0, hand.ownReachWeight) * hand.strategy[index],
            0,
        );

        return {
            combos: weightedProbability,
            color: actionColor(action),
            index,
            isAllIn: action.isAllIn,
            kind: action.kind,
            label: actionLabel(action),
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
        rawGroups.set(label, [...(rawGroups.get(label) ?? []), hand]);
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
