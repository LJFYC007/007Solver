export const STREETS = ["flop", "turn", "river"] as const;
export type Street = (typeof STREETS)[number];
export const POSITIONS = ["oop", "ip"] as const;
export type Position = (typeof POSITIONS)[number];

type StreetSizes<T> = Record<Street, { bet: T; raise: T }>;

type Tree<T> = Record<Position, StreetSizes<T>> & {
    maxRaises: number;
    allInSpr: number;
};

export type BettingTree = Tree<number[]>;
/** Sizes as typed, one entry per size field. */
export type BettingTreeDraft = Tree<string[]>;

const defaultSizes = (): StreetSizes<string[]> => ({
    flop: { bet: ["50"], raise: ["50"] },
    turn: { bet: ["50"], raise: ["50"] },
    river: { bet: ["50"], raise: ["50"] },
});

export const defaultBettingTree = (): BettingTreeDraft => ({
    oop: defaultSizes(),
    ip: defaultSizes(),
    maxRaises: 2,
    allInSpr: 0.15,
});

/** Parses a typed pot percentage, or returns undefined unless it is positive with at most two decimals. */
export function parsePercentage(text: string): number | undefined {
    const token = text.trim();
    const value = Number(token);
    return /^\d+(?:\.\d{1,2})?$/.test(token) && value > 0 && value * 100 <= 2147483647 ? value : undefined;
}

/** Parses the typed sizes of one field, skipping empty ones. */
function parsePercentages(texts: string[], field: string): number[] {
    return texts
        .filter((text) => text.trim())
        .map((text) => {
            const value = parsePercentage(text);
            if (value === undefined)
                throw new Error(`${field}: sizes must be positive percentages with up to two decimals.`);
            return value;
        });
}

export function parseBettingTree(draft: BettingTreeDraft): BettingTree {
    if (!Number.isInteger(draft.maxRaises) || draft.maxRaises < 0 || draft.maxRaises > 2)
        throw new Error("Maximum raises must be 0, 1 or 2, excluding the opening bet.");
    if (!Number.isFinite(draft.allInSpr) || draft.allInSpr < 0)
        throw new Error("All-in SPR must be a non-negative number.");
    const street = (position: Position, name: Street) => {
        const label = `${position.toUpperCase()} ${name}`;
        return {
            bet: parsePercentages(draft[position][name].bet, `${label} bet`),
            raise: parsePercentages(draft[position][name].raise, `${label} raise`),
        };
    };
    const sizes = (position: Position): StreetSizes<number[]> => ({
        flop: street(position, "flop"),
        turn: street(position, "turn"),
        river: street(position, "river"),
    });
    return {
        oop: sizes("oop"),
        ip: sizes("ip"),
        maxRaises: draft.maxRaises,
        allInSpr: draft.allInSpr,
    };
}
