import type { DecisionAction, DecisionNode, HandStrategy, Player } from "./types";

export interface AggregatedAction {
    amountTo: number;
    chipsCommitted: number;
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
    label: string;
    ownReachWeight: number;
}

export const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"] as const;
export const SUITS = ["s", "h", "d", "c"] as const;
export const SUIT_SYMBOLS: Record<string, string> = {
    c: "♣",
    d: "♦",
    h: "♥",
    s: "♠",
};

export function playerLabel(player: Player): string {
    return player === "hero" ? "Hero" : "Villain";
}

export function actionLabel(action: Pick<DecisionAction, "amountTo" | "chipsCommitted" | "isAllIn" | "kind">): string {
    const allInSuffix = action.isAllIn ? " (all-in)" : "";
    if (action.kind === "fold") return "Fold";
    if (action.kind === "check") return "Check";
    if (action.kind === "call") return `Call ${formatNumber(action.chipsCommitted)}${allInSuffix}`;
    if (action.kind === "bet") return `Bet ${formatNumber(action.amountTo)}${allInSuffix}`;
    return `Raise to ${formatNumber(action.amountTo)}${allInSuffix}`;
}

export function actionColor(action: Pick<DecisionAction, "isAllIn" | "kind">): string {
    if (action.isAllIn) return "#74102f";
    if (action.kind === "fold") return "#147a9f";
    if (action.kind === "call") return "#65c261";
    if (action.kind === "check") return "#54ad63";
    if (action.kind === "bet" || action.kind === "raise") return "#c91f59";
    return "#777c79";
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
            amountTo: action.amountTo,
            chipsCommitted: action.chipsCommitted,
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
                    label,
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

export function strategyGradient(actions: AggregatedAction[]): string | undefined {
    const visibleActions = orderedActions(actions).filter((action) => action.probability > 0.000001);
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

export function formatPercent(value: number): string {
    const percentage = Math.max(0, value) * 100;
    if (percentage > 0 && percentage < 0.1) return "<0.1%";
    return `${percentage.toFixed(1)}%`;
}

export function formatEv(value: number | null): string {
    if (value === null) return "—";
    return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

export function formatNumber(value: number): string {
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
}
