import { type CSSProperties, memo, useMemo } from "react";
import { type DecisionNode, HAND_CLASSES, strategyGradient } from "../solver";
import type { PreflopNode } from "../solver/catalog";
import {
    type DisplayMode,
    type EvComparison,
    type HandRow,
    type MatrixCell,
    RANGE_BACKGROUND,
    cellValue,
    postflopCells,
} from "../solver/display";
import { probabilities } from "../solver/preflop";

export interface Selection {
    selected: string;
    locked: boolean;
    onHover: (label: string) => void;
    onPick: (label: string) => void;
}

// Memoized so hovering a cell re-renders only the cells whose selection changed.
const RangeCell = memo(function RangeCell({
    cell,
    selected,
    locked,
    onHover,
    onPick,
}: {
    cell: MatrixCell;
    selected: boolean;
    locked: boolean;
    onHover: (label: string) => void;
    onPick: (label: string) => void;
}) {
    return (
        <button
            type="button"
            role="gridcell"
            aria-label={cell.description}
            aria-selected={selected}
            className={`range-cell${cell.included ? " in-range" : ""}${cell.included && cell.weight === 0 ? " zero-own-reach" : ""}${locked ? " locked" : ""}`}
            onMouseEnter={() => onHover(cell.label)}
            onFocus={() => onHover(cell.label)}
            onClick={() => onPick(cell.label)}
            style={
                {
                    "--strategy-background": cell.background,
                    "--strategy-height": `${Math.min(1, Math.max(0, cell.weight)) * 100}%`,
                } as CSSProperties
            }
            title={cell.description}
        >
            <strong>{cell.label}</strong>
            {cell.value !== undefined && cell.weight > 0 && <small className="range-cell-value">{cell.value}</small>}
        </button>
    );
});

export function RangeMatrix({
    cells,
    selected,
    locked,
    onHover,
    onPick,
    compact = false,
}: Selection & { cells: MatrixCell[]; compact?: boolean }) {
    return (
        <div className={`range-grid${compact ? " compact" : ""}`} aria-label="13 by 13 strategy matrix" role="grid">
            {cells.map((cell) => (
                <RangeCell
                    key={cell.label}
                    cell={cell}
                    selected={selected === cell.label}
                    locked={locked && selected === cell.label}
                    onHover={onHover}
                    onPick={onPick}
                />
            ))}
        </div>
    );
}

export function PreflopRangeMatrix({
    node,
    range,
    mode,
    ...selection
}: Selection & { node?: PreflopNode; range: Record<string, number>; mode: DisplayMode }) {
    const cells = useMemo(
        () =>
            HAND_CLASSES.map((label) => {
                const weights = node ? probabilities(node, label) : undefined;
                const weight = range[label] ?? 0;
                return {
                    label,
                    weight,
                    included: weight > 0 && (!node || !!weights),
                    description: `${label}, ${node ? node.actions.map((action, index) => `${action.label} ${((weights?.[index] ?? 0) * 100).toFixed(2)}%`).join(", ") : `${(weight * 100).toFixed(2)}%`}`,
                    background:
                        mode === "range"
                            ? RANGE_BACKGROUND
                            : node
                              ? strategyGradient(
                                    node.actions.map((action, index) => ({
                                        color: action.color,
                                        probability: weights?.[index] ?? 0,
                                    })),
                                )
                              : "linear-gradient(#69b86b,#69b86b)",
                    value: cellValue(mode, weight),
                };
            }),
        [node, range, mode],
    );
    return <RangeMatrix cells={cells} {...selection} />;
}

export function PostflopRangeMatrix({
    node,
    rows,
    mode,
    comparison,
    filtered,
    ...selection
}: Selection & {
    node: DecisionNode;
    rows: HandRow[];
    mode: DisplayMode;
    comparison?: EvComparison;
    filtered: boolean;
}) {
    const cells = useMemo(
        () => postflopCells({ node, rows, mode, comparison, filtered }),
        [node, rows, mode, comparison, filtered],
    );
    return <RangeMatrix cells={cells} {...selection} />;
}
