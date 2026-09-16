export const STREETS = ["flop", "turn", "river"] as const;
export type Street = (typeof STREETS)[number];

export type BettingTree = Record<Street, { bet: number[]; raise: number[] }> & {
    maxRaises: number;
    allInSpr: number;
};

export type BettingTreeDraft = Record<Street, { bet: string; raise: string }> & {
    maxRaises: number;
    allInSpr: number;
};

export const defaultBettingTree = (): BettingTreeDraft => ({
    flop: { bet: "50", raise: "50" },
    turn: { bet: "50", raise: "50" },
    river: { bet: "50", raise: "50" },
    maxRaises: 2,
    allInSpr: 0.15,
});

export function parseBettingTree(draft: BettingTreeDraft): BettingTree {
    const percentages = (text: string, field: string) => {
        if (!text.trim()) return [];
        return text.split(",").map((token) => {
            const value = Number(token.trim());
            if (
                !/^\d+(?:\.\d{1,2})?$/.test(token.trim()) ||
                !Number.isFinite(value) ||
                value <= 0 ||
                value * 100 > 2147483647
            )
                throw new Error(`${field}: enter positive percentages separated by commas (up to two decimals).`);
            return value;
        });
    };
    if (!Number.isInteger(draft.maxRaises) || draft.maxRaises < 0 || draft.maxRaises > 2)
        throw new Error("Maximum raises must be 0, 1 or 2, excluding the opening bet.");
    if (!Number.isFinite(draft.allInSpr) || draft.allInSpr < 0)
        throw new Error("All-in SPR must be a non-negative number.");
    const street = (name: Street) => ({
        bet: percentages(draft[name].bet, `${name} bet`),
        raise: percentages(draft[name].raise, `${name} raise`),
    });
    return {
        flop: street("flop"),
        turn: street("turn"),
        river: street("river"),
        maxRaises: draft.maxRaises,
        allInSpr: draft.allInSpr,
    };
}
