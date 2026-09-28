import {
    FILTER_SECTIONS,
    type FilterSectionId,
    classifyDraw,
    classifyMadeHand,
    equityAdvancedBucket,
    equitySimpleBucket,
} from "./handCategories";
import {
    HAND_CLASSES,
    aggregateActions,
    cardsKey,
    handClassCombos,
    strategyGradient,
    toHandClass,
    weightedMean,
} from "./strategy";
import type { DecisionNode, EquityReport, HandStrategy, OpponentEvReport, Player } from "./types";

export type DisplayMode =
    "strategy" | "strategyEv" | "strategyEq" | "ev" | "compareEv" | "equity" | "eqr" | "range" | "all";

/** GTO Wizard's strategy display options; the preflop catalog has no EV or equity. */
export const DISPLAY_MODES: readonly { id: DisplayMode; label: string; preflop: boolean }[] = [
    { id: "strategy", label: "Strategy", preflop: true },
    { id: "strategyEv", label: "Strategy + EV", preflop: false },
    { id: "strategyEq", label: "Strategy + EQ", preflop: false },
    { id: "ev", label: "EV", preflop: false },
    { id: "compareEv", label: "Compare EV", preflop: false },
    { id: "equity", label: "Equity", preflop: false },
    { id: "eqr", label: "EQR", preflop: false },
    { id: "range", label: "Range", preflop: true },
    { id: "all", label: "All", preflop: true },
];

export const modeLabel = (mode: DisplayMode) => DISPLAY_MODES.find((option) => option.id === mode)!.label;

export const usesEquity = (mode: DisplayMode) =>
    mode === "strategyEq" || mode === "equity" || mode === "eqr" || mode === "all";

/** One exact hand of a player at a decision node. */
export interface HandRow {
    cards: string[];
    key: string;
    label: string;
    weight: number;
    /** Weights EV and equity averages: joint reach, or own reach while the waiting player's EVs are missing. */
    mass: number;
    /** Node strategy EV; the waiting player's come from a separate query. */
    ev: number | null;
    /** Undefined until the equity report arrives. */
    equity?: number | null;
    /** The acting player's strategy entry. */
    hand?: HandStrategy;
    made?: string;
    draw?: string;
}

/** Out of position first, as the matrices and charts show the players. */
export const PLAYERS: readonly Player[] = ["villain", "hero"];
export const PLAYER_COLORS: Record<Player, string> = { hero: "#8cce83", villain: "#d1cd69" };

export function frequency(probability: number) {
    return probability > 0 && probability < 0.001 ? "<0.1" : (probability * 100).toFixed(1);
}

export const percent = (value: number | null | undefined) => (value == null ? "—" : `${(value * 100).toFixed(1)}%`);

/** The actor's hands at a decision node, without equity so that its arrival does not classify them again. */
export function actorRows(node: DecisionNode): HandRow[] {
    return node.hands.map((hand) => ({
        cards: hand.cards,
        key: cardsKey(hand.cards),
        label: toHandClass(hand.cards),
        weight: hand.ownReachWeight,
        mass: hand.marginalReachMass,
        ev: hand.nodeStrategyEv,
        hand,
        made: classifyMadeHand(hand.cards, node.state.board),
        draw: classifyDraw(hand.cards, node.state.board),
    }));
}

/** A player's rows with their equities, once the report arrives. */
export function withEquity(rows: HandRow[], player: Player, report?: EquityReport): HandRow[] {
    if (!report) return rows;
    const equity = new Map(report.players[player].hands.map((hand) => [cardsKey(hand.cards), hand.equity]));
    return rows.map((row) => ({ ...row, equity: equity.get(row.key) ?? null }));
}

/** The waiting player's hands, which the equity report lists with positive own reach. */
export function opponentRows(report: EquityReport, player: Player, opponentEvs?: OpponentEvReport): HandRow[] {
    const evs = new Map(opponentEvs?.hands.map((hand) => [cardsKey(hand.cards), hand]));
    return report.players[player].hands.map((hand) => {
        const key = cardsKey(hand.cards);
        const ev = evs.get(key);
        return {
            cards: hand.cards,
            key,
            label: toHandClass(hand.cards),
            weight: hand.ownReachWeight,
            mass: ev ? ev.marginalReachMass : hand.ownReachWeight,
            ev: ev?.nodeStrategyEv ?? null,
            equity: hand.equity,
        };
    });
}

export const totalWeight = (rows: readonly HandRow[]) => rows.reduce((sum, row) => sum + row.weight, 0);

/** The acting player's action mix over the rows' hands. */
export const rowActions = (node: DecisionNode, rows: readonly HandRow[]) =>
    aggregateActions(
        node,
        rows.flatMap((row) => (row.hand ? [row.hand] : [])),
    );

const average = (rows: readonly HandRow[], value: (row: HandRow) => number | null | undefined) =>
    weightedMean(rows, value, (row) => row.mass);

export const evShare = (ev: number | null | undefined, pot: number) => (ev == null || pot <= 0 ? undefined : ev / pot);

export const equityRealization = (ev: number | null | undefined, equity: number | null | undefined, pot: number) =>
    ev != null && equity != null && equity > 0 && pot > 0 ? ev / (equity * pot) : undefined;

/** Rows by key, leaving out rows without one. */
function groupRows(rows: readonly HandRow[], key: (row: HandRow) => string | undefined) {
    const groups = new Map<string, HandRow[]>();
    for (const row of rows) {
        const id = key(row);
        if (id === undefined) continue;
        const group = groups.get(id);
        if (group) group.push(row);
        else groups.set(id, [row]);
    }
    return groups;
}

type ColorStops = readonly (readonly [number, readonly [number, number, number]])[];
// GTO Wizard's gradient: red, orange, yellow, green.
const HEAT_STOPS: ColorStops = [
    [0, [214, 67, 63]],
    [0.35, [233, 137, 47]],
    [0.65, [216, 196, 60]],
    [1, [88, 179, 92]],
];
// Compare EV: khaki for equal EVs (GTO Wizard's pale yellow, darkened for white labels), red or green
// once the difference reaches a tenth of the pot.
const DIFFERENCE_STOPS: ColorStops = [
    [0, [214, 67, 63]],
    [0.5, [176, 164, 98]],
    [1, [88, 179, 92]],
];
const FULL_DIFFERENCE = 0.1;

function gradientColor(stops: ColorStops, position: number) {
    const t = Math.min(1, Math.max(0, position));
    const upper = stops.findIndex(([stop]) => stop >= t);
    if (upper <= 0) return `rgb(${stops[0][1].join(" ")})`;
    const [start, from] = stops[upper - 1];
    const [end, to] = stops[upper];
    const share = (t - start) / (end - start);
    return `rgb(${from.map((value, i) => Math.round(value + (to[i] - value) * share)).join(" ")})`;
}

const heatColor = (position: number) => gradientColor(HEAT_STOPS, position);

const differenceColor = (difference: number, pot: number) =>
    gradientColor(DIFFERENCE_STOPS, pot > 0 ? 0.5 + difference / (2 * FULL_DIFFERENCE * pot) : 0.5);

/** Compare EV's choice: one action, by node action index, against another or each hand's best action. */
export interface EvComparison {
    action: number;
    versus: number | "best";
}

/** Compare EV's choice as positions in the node's displayed action order, so it carries over to other sizes. */
export type ComparisonChoice = EvComparison;

/**
 * The choice at a node with `count` actions: positions clamp to the last action, and an action that clamping
 * makes face itself faces the best action instead.
 */
export function resolveChoice(choice: ComparisonChoice, count: number): ComparisonChoice {
    const clamp = (position: number) => Math.min(position, count - 1);
    const action = clamp(choice.action);
    const versus = choice.versus === "best" ? "best" : clamp(choice.versus);
    return { action, versus: versus === action ? "best" : versus };
}

/** The chosen action's EV minus the compared EV, when the hand has action EVs. */
function evDifference(actionEvs: readonly number[], comparison: EvComparison) {
    if (!actionEvs.length) return undefined;
    const versus = comparison.versus === "best" ? Math.max(...actionEvs) : actionEvs[comparison.versus];
    return actionEvs[comparison.action] - versus;
}

export type Metric = "ev" | "equity" | "eqr";

const metricOf = (mode: DisplayMode): Metric | undefined =>
    mode === "ev" || mode === "equity" || mode === "eqr" ? mode : undefined;

/** Maps metric values onto the gradient: equity as is, EQR from 50% to 150%, EV as a share of the pot. */
function metricScale(metric: Metric, pot: number): (value: number) => number {
    if (metric === "equity") return (value) => value;
    if (metric === "eqr") return (value) => value - 0.5;
    return (value) => (pot > 0 ? value / pot : 0.5);
}

type HandColor = (
    ev: number | null | undefined,
    equity: number | null | undefined,
    actionEvs: readonly number[] | undefined,
) => string | undefined;

/** How the metric and Compare EV modes color a hand, undefined in other modes; hands without the value have none. */
export function handColor(mode: DisplayMode, pot: number, comparison?: EvComparison): HandColor | undefined {
    if (mode === "compareEv")
        return (
            comparison &&
            ((_ev, _equity, actionEvs) => {
                const difference = evDifference(actionEvs ?? [], comparison);
                return difference === undefined ? undefined : differenceColor(difference, pot);
            })
        );
    const metric = metricOf(mode);
    if (!metric) return undefined;
    const scale = metricScale(metric, pot);
    return (ev, equity) => {
        const value = metric === "ev" ? ev : metric === "equity" ? equity : equityRealization(ev, equity, pot);
        return value == null ? undefined : heatColor(scale(value));
    };
}

/** Stripes of the class's hands, sized by own reach; hands without a color have no value to show. */
function rowGradient(rows: readonly HandRow[], color: (row: HandRow) => string | undefined) {
    return strategyGradient(
        rows.map((row) => ({ color: color(row) ?? "var(--surface-hover)", probability: row.weight })),
    );
}

export function formatMetric(metric: Metric, value: number | undefined) {
    if (value === undefined) return undefined;
    return metric === "ev" ? value.toFixed(2) : (value * 100).toFixed(1);
}

export interface MatrixCell {
    label: string;
    description: string;
    included: boolean;
    weight: number;
    background?: string;
    value?: string;
}
export const RANGE_BACKGROUND = "linear-gradient(var(--range-color), var(--range-color))";

/** A matrix cell's number in the modes that show one. */
export function cellValue(mode: DisplayMode, weight: number, ev?: number, equity?: number, eqr?: number) {
    if (mode === "strategyEv" || mode === "ev") return formatMetric("ev", ev);
    if (mode === "strategyEq" || mode === "equity") return formatMetric("equity", equity);
    if (mode === "eqr") return formatMetric("eqr", eqr);
    if (mode === "range") return weight.toFixed(2);
}

/**
 * Hand classes of one player at a decision node: the actor's rows carry its strategy and action EVs, the
 * waiting player's only reach, equity and EV. Filtered rows leave out classes without a matching hand.
 */
export function postflopCells({
    node,
    rows,
    mode,
    comparison,
    filtered = false,
}: {
    node: DecisionNode;
    rows: HandRow[];
    mode: DisplayMode;
    /** Required in Compare EV mode. */
    comparison?: EvComparison;
    filtered?: boolean;
}): MatrixCell[] {
    const { board, pot } = node.state;
    const groups = groupRows(rows, (row) => row.label);
    const color = handColor(mode, pot, comparison);
    return HAND_CLASSES.map((label) => {
        const group = groups.get(label);
        const available = handClassCombos(label).filter((cards) => cards.every((card) => !board.includes(card))).length;
        if (!group) {
            return {
                label,
                included: false,
                weight: 0,
                description: `${label}, ${available === 0 ? "blocked by board" : filtered ? "no hand matches the filters" : "not in range"}`,
            };
        }
        // Share of the class's board-compatible combos, as in the strategy matrix.
        const weight = totalWeight(group) / Math.max(1, available);
        const actions = rowActions(node, group);
        const ev = average(group, (row) => row.ev);
        const equity = average(group, (row) => row.equity);
        const eqr = equityRealization(ev, equity, pot);
        const compared = mode === "compareEv" ? comparison : undefined;
        const difference = (row: HandRow) => compared && evDifference(row.hand?.actionEvs ?? [], compared);
        // The chosen action's EV as a share of the pot, as GTO Wizard's EV/POT.
        const actionShare =
            compared &&
            evShare(
                average(group, (row) => row.hand?.actionEvs[compared.action]),
                pot,
            );
        const meanDifference = compared && evShare(average(group, difference), pot);
        const actionName = (index: number | "best") =>
            index === "best" ? "the best action" : actions.find((action) => action.index === index)?.label;
        const facts = [
            compared && actionShare !== undefined
                ? `${actionName(compared.action)} EV/POT ${percent(actionShare)}`
                : "",
            compared && meanDifference !== undefined
                ? `${percent(meanDifference)} of the pot against ${actionName(compared.versus)}`
                : "",
            ...actions.map((action) => `${action.label} ${percent(action.probability)}`),
            ev === undefined ? "" : `EV ${ev.toFixed(2)}`,
            equity === undefined ? "" : `equity ${percent(equity)}`,
            eqr === undefined ? "" : `EQR ${(eqr * 100).toFixed(0)}%`,
            `range ${percent(weight)}`,
        ].filter(Boolean);
        return {
            label,
            included: true,
            weight,
            description:
                weight === 0
                    ? `${label}, ${group.length} combinations, zero own reach`
                    : `${label}, ${facts.join(", ")}`,
            background:
                mode === "range"
                    ? RANGE_BACKGROUND
                    : color
                      ? rowGradient(group, (row) => color(row.ev, row.equity, row.hand?.actionEvs))
                      : strategyGradient(actions),
            value: compared ? formatMetric("equity", actionShare) : cellValue(mode, weight, ev, equity, eqr),
        };
    });
}

export interface HandFilters {
    categories: readonly string[];
    suitMode: "include" | "exclude";
    offsuit: readonly string[];
    suited: readonly string[];
}

export const NO_FILTERS: HandFilters = { categories: [], suitMode: "include", offsuit: [], suited: [] };

export const hasFilters = (filters: HandFilters) =>
    filters.categories.length + filters.offsuit.length + filters.suited.length > 0;

export const isEquitySection = (section?: FilterSectionId) => section === "eqSimple" || section === "eqAdvanced";

const sectionOf = new Map(FILTER_SECTIONS.flatMap((section) => section.categories.map((c) => [c.id, section.id])));
export const categoryLabel = new Map(
    FILTER_SECTIONS.flatMap((section) => section.categories.map((c) => [c.id, c.label])),
);

function rowCategory(row: HandRow, section: FilterSectionId) {
    if (section === "hands") return row.made;
    if (section === "draws") return row.draw;
    if (row.equity == null) return undefined;
    return section === "eqSimple" ? equitySimpleBucket(row.equity) : equityAdvancedBucket(row.equity);
}

/** One section's categories with their hands and shares of the hands that have a category there. */
export function categoryGroups(rows: readonly HandRow[], section: FilterSectionId) {
    const groups = groupRows(rows, (row) => rowCategory(row, section));
    const total = totalWeight([...groups.values()].flat());
    const share = (category: string) => (total > 0 ? totalWeight(groups.get(category) ?? []) / total : undefined);
    return { groups, share };
}

export const filtersUseEquity = (filters: HandFilters) =>
    filters.categories.some((id) => isEquitySection(sectionOf.get(id)));

// Offsuit suits keep unsuited hands (pairs included) holding one of them, suited suits keep suited hands of
// them; exclusion keeps the hands of each kind without them.
function matchesSuits(cards: readonly string[], filters: HandFilters) {
    if (!filters.offsuit.length && !filters.suited.length) return true;
    const include = filters.suitMode === "include";
    if (cards[0][1] === cards[1][1])
        return filters.suited.length > 0 && filters.suited.includes(cards[0][1]) === include;
    return filters.offsuit.length > 0 && cards.some((card) => filters.offsuit.includes(card[1])) === include;
}

/** Categories of one section combine as alternatives, sections and suits as requirements. */
export function filterRows(rows: HandRow[], filters: HandFilters): HandRow[] {
    const sections = new Map<FilterSectionId, string[]>();
    for (const id of filters.categories) {
        const section = sectionOf.get(id)!;
        sections.set(section, [...(sections.get(section) ?? []), id]);
    }
    return rows.filter(
        (row) =>
            matchesSuits(row.cards, filters) &&
            [...sections].every(([section, ids]) => ids.includes(rowCategory(row, section) ?? "")),
    );
}
