// Hand categories of GTO Wizard's Filters tab. Cards are strings such as "As" or "Td".

import { SUITS } from "./strategy";

export type FilterSectionId = "hands" | "eqSimple" | "draws" | "eqAdvanced";

export interface FilterCategory {
    id: string;
    label: string;
}

export interface FilterSection {
    id: FilterSectionId;
    title: string;
    categories: readonly FilterCategory[];
}

const section = (id: FilterSectionId, title: string, categories: readonly (readonly [string, string])[]) => ({
    id,
    title,
    categories: categories.map(([categoryId, label]) => ({ id: categoryId, label })),
});

export const FILTER_SECTIONS: readonly FilterSection[] = [
    section("hands", "Hands", [
        ["straightFlush", "Straight flush"],
        ["quads", "Quads"],
        ["fullHouse", "Full house"],
        ["flush", "Flush"],
        ["straight", "Straight"],
        ["set", "Set"],
        ["trips", "Trips"],
        ["twoPair", "Two pair"],
        ["overpair", "Overpair"],
        ["topPair", "Top pair"],
        ["underpair", "Underpair"],
        ["secondPair", "Second pair"],
        ["thirdPair", "Third pair"],
        ["fourthPair", "Fourth pair"],
        ["fifthPair", "Fifth pair"],
        ["aceHigh", "Ace high"],
        ["kingHigh", "King high"],
        ["noMadeHand", "No made hand"],
    ]),
    section("eqSimple", "EQ buckets - Simple", [
        ["best", "Best hands"],
        ["good", "Good hands"],
        ["weak", "Weak hands"],
        ["trash", "Trash hands"],
    ]),
    section("draws", "Draws", [
        ["comboDraw", "Combo draw"],
        ["flushDrawNuts", "Flush draw nuts"],
        ["flushDraw", "Flush draw"],
        ["oesd", "OESD"],
        ["gutshot", "Gutshot"],
        ["bdfd2", "BDFD 2 cards"],
        ["bdfd1", "BDFD 1 card"],
        ["bdsd", "BDSD"],
        ["noDraw", "No draw"],
    ]),
    section("eqAdvanced", "EQ buckets - Advanced", [
        ["eq90", "Equity 90-100"],
        ["eq80", "Equity 80-90"],
        ["eq70", "Equity 70-80"],
        ["eq60", "Equity 60-70"],
        ["eq50", "Equity 50-60"],
        ["eq25", "Equity 25-50"],
        ["eq0", "Equity 0-25"],
    ]),
];

const RANK_CHARS = "23456789TJQKA";
const PAIR_IDS = ["topPair", "secondPair", "thirdPair", "fourthPair", "fifthPair"];

/** Rank value from 2 to 14 (Ace). */
const rankOf = (card: string) => RANK_CHARS.indexOf(card[0]) + 2;
const ofSuit = (cards: readonly string[], suit: string) => cards.filter((card) => card[1] === suit);
// An Ace also sets bit 1, so it plays low in straights.
const rankBits = (rank: number) => (rank === 14 ? (1 << 14) | 2 : 1 << rank);
const rankMask = (cards: readonly string[]) => cards.reduce((mask, card) => mask | rankBits(rankOf(card)), 0);
const runs = (length: number) => Array.from({ length: 15 - length }, (_, i) => ((1 << length) - 1) << (i + 1));
// A-2-3-4-5 through T-J-Q-K-A.
const STRAIGHTS = runs(5);
const BACKDOOR_RUNS = runs(3);
const hasStraight = (mask: number) => STRAIGHTS.some((straight) => (mask & straight) === straight);

function strongMadeHand(cards: readonly string[]): string | undefined {
    const suited = SUITS.map((suit) => ofSuit(cards, suit));
    if (suited.some((group) => group.length >= 5 && hasStraight(rankMask(group)))) return "straightFlush";
    const counts = [...RANK_CHARS].map((rank) => cards.filter((card) => card[0] === rank).length).sort((a, b) => b - a);
    if (counts[0] === 4) return "quads";
    if (counts[0] === 3 && counts[1] >= 2) return "fullHouse";
    if (suited.some((group) => group.length >= 5)) return "flush";
    if (hasStraight(rankMask(cards))) return "straight";
    return undefined;
}

export function classifyMadeHand(hole: readonly string[], board: readonly string[]): string {
    const strong = strongMadeHand([...hole, ...board]);
    if (strong) return strong;
    const boardRanks = board.map(rankOf);
    const distinct = [...new Set(boardRanks)].sort((a, b) => b - a);
    const [high, low] = hole.map(rankOf).sort((a, b) => b - a);
    if (high === low) {
        if (distinct.includes(high)) return "set";
        const above = distinct.filter((rank) => rank > high).length;
        if (above === 0) return "overpair";
        return above === distinct.length ? "underpair" : PAIR_IDS[above];
    }
    if ([high, low].some((rank) => boardRanks.filter((boardRank) => boardRank === rank).length >= 2)) return "trips";
    const matched = [high, low].filter((rank) => distinct.includes(rank));
    if (matched.length === 2) return "twoPair";
    if (matched.length === 1) return PAIR_IDS[distinct.indexOf(matched[0])];
    return high === 14 ? "aceHigh" : high === 13 ? "kingHigh" : "noMadeHand";
}

function flushDraw(hole: readonly string[], board: readonly string[]): string | undefined {
    for (const suit of SUITS) {
        const own = ofSuit(hole, suit).map(rankOf);
        if (own.length === 0 || own.length + ofSuit(board, suit).length !== 4) continue;
        let nut = 14;
        while (board.includes(RANK_CHARS[nut - 2] + suit)) nut--;
        return Math.max(...own) === nut ? "flushDrawNuts" : "flushDraw";
    }
    return undefined;
}

/** Ranks that complete a straight the hole cards take part in; 0 when the hand already has a straight. */
function straightOuts(hole: readonly string[], board: readonly string[]): number {
    const known = [...hole, ...board];
    const mask = rankMask(known);
    if (hasStraight(mask)) return 0;
    const boardMask = rankMask(board);
    return [...RANK_CHARS].filter((char, i) => {
        const bits = rankBits(i + 2);
        return (
            known.filter((card) => card[0] === char).length < 4 &&
            STRAIGHTS.some(
                (straight) => ((mask | bits) & straight) === straight && ((boardMask | bits) & straight) !== straight,
            )
        );
    }).length;
}

/** Hole cards of the suit with exactly three known cards, 0 when there is none. */
function backdoorFlushCards(hole: readonly string[], board: readonly string[]): number {
    return Math.max(
        ...SUITS.map((suit) => {
            const own = ofSuit(hole, suit).length;
            return own + ofSuit(board, suit).length === 3 ? own : 0;
        }),
    );
}

function hasBackdoorStraight(hole: readonly string[], board: readonly string[]): boolean {
    const [first, second] = hole.map((card) => rankBits(rankOf(card)));
    const boardMask = rankMask(board);
    if (first === second || (first | second) & boardMask) return false;
    const mask = first | second | boardMask;
    return BACKDOOR_RUNS.some((run) => (mask & run) === run && (first & run) !== 0 && (second & run) !== 0);
}

export function classifyDraw(hole: readonly string[], board: readonly string[]): string | undefined {
    if (board.length === 5) return undefined;
    const flush = flushDraw(hole, board);
    const outs = straightOuts(hole, board);
    if (flush) return outs > 0 ? "comboDraw" : flush;
    if (outs > 0) return outs >= 2 ? "oesd" : "gutshot";
    if (board.length === 3) {
        const backdoor = backdoorFlushCards(hole, board);
        if (backdoor > 0) return backdoor === 2 ? "bdfd2" : "bdfd1";
        if (hasBackdoorStraight(hole, board)) return "bdsd";
    }
    return "noDraw";
}

const SIMPLE_BUCKETS = [
    [0.75, "best"],
    [0.5, "good"],
    [0.33, "weak"],
] as const;
const ADVANCED_BUCKETS = [
    [0.9, "eq90"],
    [0.8, "eq80"],
    [0.7, "eq70"],
    [0.6, "eq60"],
    [0.5, "eq50"],
    [0.25, "eq25"],
] as const;

/** The simple buckets' equity ranges, as the Filters tab explains them. */
export const SIMPLE_BUCKETS_HELP =
    "Hands are separated by their equity: best hands 100% - 75%, good hands 75% - 50%, weak hands 50% - 33%, trash hands 33% - 0%.";

export function equitySimpleBucket(equity: number): string {
    return SIMPLE_BUCKETS.find(([min]) => equity >= min)?.[1] ?? "trash";
}

export function equityAdvancedBucket(equity: number): string {
    return ADVANCED_BUCKETS.find(([min]) => equity >= min)?.[1] ?? "eq0";
}
