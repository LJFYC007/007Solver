import { memo, useLayoutEffect, useMemo, useRef, useState } from "react";
import { strategyGradient } from "../solver";
import { formatMetric, frequency, percent } from "../solver/display";
import type { DetailCombo } from "../solver/study";
import { Card } from "./PlayingCards";

type SortKey = "hand" | "strategy" | "range" | "evPot" | "equity" | "eqr";
// Preflop has only the first three.
const COLUMNS: [SortKey, string][] = [
    ["hand", "Hand"],
    ["strategy", "Strategy"],
    ["range", "Range"],
    ["evPot", "EV/POT"],
    ["equity", "EQ %"],
    ["eqr", "EQR %"],
];
// Rows keep this height (see .summary-table), so only those near the visible part render.
const ROW_HEIGHT = 24;
const OVERSCAN = 10;
// The EQ % and EQR % columns name the unit in the heading.
const share = (value: number | null | undefined) => formatMetric("equity", value ?? undefined) ?? "—";

// Combinations arrive in matrix order, which the hand column keeps; other columns sort by value, hands
// without one last.
function sortValue(combo: DetailCombo, key: SortKey, index: number): number | undefined {
    if (key === "hand") return -index;
    // Preflop and postflop list actions in different orders, so strategy sorts by aggression.
    if (key === "strategy")
        return combo.actions.length
            ? combo.actions.reduce((sum, a) => sum + (a.kind === "bet" || a.kind === "raise" ? a.probability : 0), 0)
            : undefined;
    if (key === "range") return combo.weight;
    if (key === "evPot") return combo.evPot;
    if (key === "equity") return combo.equity ?? undefined;
    return combo.eqr;
}

// Memoized so hovering the matrix re-renders only the rows whose selection changed.
const SummaryRow = memo(function SummaryRow({
    combo,
    index,
    postflop,
    selected,
    onPick,
}: {
    combo: DetailCombo;
    index: number;
    postflop: boolean;
    selected: boolean;
    onPick: (label: string) => void;
}) {
    return (
        <tr className={selected ? "selected" : undefined} aria-rowindex={index + 2} onClick={() => onPick(combo.label)}>
            <td>
                <button type="button" className="summary-hand" aria-label={`Show ${combo.label}`}>
                    <span className="combo-cards summary-cards">
                        {combo.cards.map((card) => (
                            <Card key={card} card={card} />
                        ))}
                    </span>
                </button>
            </td>
            <td>
                <span
                    className="summary-strategy"
                    style={{ background: strategyGradient(combo.actions) }}
                    title={combo.actions.map((a) => `${a.label} ${frequency(a.probability)}%`).join(", ")}
                />
            </td>
            <td>{combo.weight.toFixed(2)}</td>
            {postflop && (
                <>
                    <td>{percent(combo.evPot)}</td>
                    <td>{share(combo.equity)}</td>
                    <td>{share(combo.eqr)}</td>
                </>
            )}
        </tr>
    );
});

export default function SummaryTable({
    combos,
    postflop,
    filtered,
    selected,
    onPick,
}: {
    combos: DetailCombo[];
    postflop: boolean;
    filtered: boolean;
    selected: string;
    onPick: (label: string) => void;
}) {
    const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({ key: "hand", descending: true });
    const rows = useMemo(() => {
        const keyed = combos.map((combo, index) => ({ combo, value: sortValue(combo, sort.key, index) }));
        const direction = sort.descending ? -1 : 1;
        return keyed
            .sort((a, b) =>
                a.value === undefined
                    ? b.value === undefined
                        ? 0
                        : 1
                    : b.value === undefined
                      ? -1
                      : (a.value - b.value) * direction,
            )
            .map(({ combo }) => combo);
    }, [combos, sort]);
    const total = rows.length;
    const scroller = useRef<HTMLDivElement>(null);
    const body = useRef<HTMLTableSectionElement>(null);
    const [windowed, setWindowed] = useState({ first: 0, count: 0 });
    useLayoutEffect(() => {
        const element = scroller.current;
        if (!element) return;
        const update = () => {
            // Bounding boxes include the root zoom, so the row height comes from the body, whose rows and
            // stand-ins are whole rows high.
            const view = element.getBoundingClientRect();
            const rowsBox = body.current!.getBoundingClientRect();
            const row = rowsBox.height / total;
            if (!row) return;
            const first = Math.max(0, Math.floor((view.top - rowsBox.top) / row) - OVERSCAN);
            const count = Math.ceil(view.height / row) + 2 * OVERSCAN;
            setWindowed((current) => (current.first === first && current.count === count ? current : { first, count }));
        };
        update();
        element.addEventListener("scroll", update, { passive: true });
        const observer = new ResizeObserver(update);
        observer.observe(element);
        return () => {
            element.removeEventListener("scroll", update);
            observer.disconnect();
        };
    }, [total]);
    const columns = postflop ? COLUMNS : COLUMNS.slice(0, 3);
    if (!total)
        return <p className="muted-copy">{filtered ? "No hands match the current filters." : "No hands in range."}</p>;
    const first = Math.min(windowed.first, total);
    const end = Math.min(total, first + windowed.count);
    const gap = (count: number) => (
        <tr className="summary-gap" aria-hidden="true" style={{ height: count * ROW_HEIGHT }}>
            <td colSpan={columns.length} />
        </tr>
    );
    return (
        <div className="summary-scroll" ref={scroller}>
            <table className="summary-table" aria-rowcount={total + 1}>
                <thead>
                    <tr>
                        {columns.map(([key, name]) => (
                            <th
                                key={key}
                                scope="col"
                                aria-sort={
                                    sort.key === key ? (sort.descending ? "descending" : "ascending") : undefined
                                }
                            >
                                <button
                                    type="button"
                                    onClick={() =>
                                        setSort((current) => ({
                                            key,
                                            descending: current.key === key ? !current.descending : true,
                                        }))
                                    }
                                >
                                    {name}
                                    <span aria-hidden="true">
                                        {sort.key === key ? (sort.descending ? "▾" : "▴") : ""}
                                    </span>
                                </button>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody ref={body}>
                    {first > 0 && gap(first)}
                    {rows.slice(first, end).map((combo, index) => (
                        <SummaryRow
                            key={combo.cards.join("")}
                            combo={combo}
                            index={first + index}
                            postflop={postflop}
                            selected={combo.label === selected}
                            onPick={onPick}
                        />
                    ))}
                    {end < total && gap(total - end)}
                </tbody>
            </table>
        </div>
    );
}
