import { type CSSProperties, memo, useMemo, useRef } from "react";
import { type DecisionNode, HAND_CLASSES, matrixLabel, strategyGradient } from "../solver";
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
import PinGlyph from "./PinGlyph";

export interface Selection {
    selected: string;
    locked: boolean;
    onHover: (label: string) => void;
    onPick: (label: string) => void;
}

/** Lock editing: the classes the selection consists of, and how clicks and strokes change it. */
interface ClassSelection {
    /** The selected classes; undefined while the whole range is selected. */
    classes?: ReadonlySet<string>;
    onSelect: (labels: string[], how: "replace" | "add" | "toggle") => void;
}

/** How a cell asks to be selected: pressed (toggling with Shift), entered with the button held, or clicked with a key. */
type CellSelect = (label: string, input: "press" | "enter" | "key", toggle?: boolean) => void;

/** The classes on the straight line from one matrix cell to another, without the first. */
function cellsBetween(from: string, to: string) {
    const [a, b] = [HAND_CLASSES.indexOf(from), HAND_CLASSES.indexOf(to)];
    const [row, column] = [Math.floor(a / 13), a % 13];
    const [rows, columns] = [Math.floor(b / 13) - row, (b % 13) - column];
    const steps = Math.max(Math.abs(rows), Math.abs(columns));
    return Array.from({ length: steps }, (_, i) => {
        const t = (i + 1) / steps;
        return matrixLabel(row + Math.round(rows * t), column + Math.round(columns * t));
    });
}

// Memoized so hovering a cell re-renders only the cells whose selection changed.
const RangeCell = memo(function RangeCell({
    cell,
    selected,
    locked,
    dimmed,
    pinned,
    onHover,
    onPick,
    onSelect,
}: {
    cell: MatrixCell;
    selected: boolean;
    locked: boolean;
    /** Lock editing: the class is outside the selection. */
    dimmed: boolean;
    /** The class has hands a node lock pins. */
    pinned: boolean;
    onHover: (label: string) => void;
    onPick: (label: string) => void;
    onSelect?: CellSelect;
}) {
    return (
        <button
            type="button"
            role="gridcell"
            aria-label={pinned ? `${cell.description}, hands pinned` : cell.description}
            aria-selected={selected}
            className={`range-cell${cell.included ? " in-range" : ""}${cell.included && cell.weight === 0 ? " zero-own-reach" : ""}${locked ? " locked" : ""}${dimmed ? " dimmed" : ""}`}
            onMouseEnter={() => onHover(cell.label)}
            onFocus={() => onHover(cell.label)}
            onPointerDown={onSelect && ((event) => event.button === 0 && onSelect(cell.label, "press", event.shiftKey))}
            onPointerEnter={onSelect && ((event) => event.buttons === 1 && onSelect(cell.label, "enter"))}
            // Pointer strokes select on press; keys click with no detail.
            onClick={(event) => (onSelect ? event.detail === 0 && onSelect(cell.label, "key") : onPick(cell.label))}
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
            {pinned && <PinGlyph />}
        </button>
    );
});

export function RangeMatrix({
    cells,
    selected,
    locked,
    onHover,
    onPick,
    selection,
    pinnedClasses,
    compact = false,
}: Selection & {
    cells: MatrixCell[];
    compact?: boolean;
    /** Lock editing: pressing, dragging and Shift-clicking select classes instead of picking one. */
    selection?: ClassSelection;
    /** The classes, by label, with hands a node lock pins; they show the pin. */
    pinnedClasses?: ReadonlySet<string>;
}) {
    const onSelect = selection?.onSelect;
    // A stroke starts with a press on a cell here and lasts until the button comes up anywhere; it holds the cell
    // it started on, then the last cell it selected.
    const stroke = useRef<{ start: string; last: string }>(undefined);
    const selectCell = useMemo<CellSelect | undefined>(
        () =>
            onSelect &&
            ((label, input, toggle) => {
                if (input === "press") {
                    // A value being typed elsewhere applies before the stroke, as it would on losing focus.
                    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
                    stroke.current = { start: label, last: label };
                    const end = () => {
                        stroke.current = undefined;
                        window.removeEventListener("pointerup", end);
                        window.removeEventListener("pointercancel", end);
                    };
                    window.addEventListener("pointerup", end);
                    window.addEventListener("pointercancel", end);
                    onSelect([label], toggle ? "toggle" : "replace");
                } else if (input === "enter") {
                    if (!stroke.current) return;
                    // A fast stroke skips cells between pointer samples, so it selects every cell on its way; the
                    // cell it started on is selected again in case the press deselected it.
                    const { start, last } = stroke.current;
                    const passed = [...(last === start ? [start] : []), ...cellsBetween(last, label)];
                    stroke.current = { start, last: label };
                    onSelect(passed, "add");
                } else onSelect([label], "toggle");
            }),
        [onSelect],
    );
    return (
        <div
            className={`range-grid${compact ? " compact" : ""}${selection ? " selecting" : ""}`}
            aria-label="13 by 13 strategy matrix"
            role="grid"
            // A native drag would cancel the pointer events that carry a stroke.
            onDragStart={selection && ((event) => event.preventDefault())}
        >
            {cells.map((cell) => (
                <RangeCell
                    key={cell.label}
                    cell={cell}
                    selected={selection ? !!selection.classes?.has(cell.label) : selected === cell.label}
                    locked={!selection && locked && selected === cell.label}
                    dimmed={!!selection?.classes && !selection.classes.has(cell.label)}
                    pinned={!!pinnedClasses?.has(cell.label)}
                    onHover={onHover}
                    onPick={onPick}
                    onSelect={selectCell}
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
    ...matrix
}: Selection & {
    node: DecisionNode;
    rows: HandRow[];
    mode: DisplayMode;
    comparison?: EvComparison;
    filtered: boolean;
    pinnedClasses?: ReadonlySet<string>;
    selection?: ClassSelection;
}) {
    const cells = useMemo(
        () => postflopCells({ node, rows, mode, comparison, filtered }),
        [node, rows, mode, comparison, filtered],
    );
    return <RangeMatrix cells={cells} {...matrix} />;
}
