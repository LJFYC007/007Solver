import { type CSSProperties, memo, useMemo } from "react";
import { type DecisionNode, HAND_CLASSES, groupHands, handClassCombos, strategyGradient } from "../solver";
import type { PreflopNode } from "../solver/catalog";
import { probabilities } from "../solver/preflop";

interface MatrixCell {
    label: string;
    description: string;
    included: boolean;
    weight: number;
    background?: string;
}
interface Selection {
    selected: string;
    onSelect: (label: string) => void;
}

// Memoized so hovering a cell re-renders only the cells whose selection changed.
const RangeCell = memo(function RangeCell({
    cell,
    selected,
    onSelect,
}: {
    cell: MatrixCell;
    selected: boolean;
    onSelect: (label: string) => void;
}) {
    return (
        <button
            type="button"
            role="gridcell"
            aria-label={cell.description}
            aria-selected={selected}
            className={`range-cell${cell.included ? " in-range" : ""}${cell.included && cell.weight === 0 ? " zero-own-reach" : ""}`}
            onMouseEnter={() => onSelect(cell.label)}
            onFocus={() => onSelect(cell.label)}
            onClick={() => onSelect(cell.label)}
            style={
                {
                    "--strategy-background": cell.background,
                    "--strategy-height": `${Math.min(1, Math.max(0, cell.weight)) * 100}%`,
                } as CSSProperties
            }
            title={cell.description}
        >
            <strong>{cell.label}</strong>
        </button>
    );
});

function RangeMatrix({ cells, selected, onSelect }: Selection & { cells: MatrixCell[] }) {
    return (
        <div className="range-grid" aria-label="13 by 13 strategy matrix" role="grid">
            {cells.map((cell) => (
                <RangeCell key={cell.label} cell={cell} selected={selected === cell.label} onSelect={onSelect} />
            ))}
        </div>
    );
}

export function PreflopRangeMatrix({
    node,
    range,
    ...selection
}: Selection & { node?: PreflopNode; range: Record<string, number> }) {
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
                    background: node
                        ? strategyGradient(
                              node.actions.map((action, index) => ({
                                  color: action.color,
                                  probability: weights?.[index] ?? 0,
                              })),
                          )
                        : "linear-gradient(#69b86b,#69b86b)",
                };
            }),
        [node, range],
    );
    return <RangeMatrix cells={cells} {...selection} />;
}

export function PostflopRangeMatrix({ node, ...selection }: Selection & { node: DecisionNode }) {
    const cells = useMemo(() => {
        const groups = groupHands(node);
        return HAND_CLASSES.map((label) => {
            const group = groups.get(label);
            if (!group) {
                const blocked = !handClassCombos(label).some((cards) =>
                    cards.every((card) => !node.state.board.includes(card)),
                );
                return {
                    label,
                    included: false,
                    weight: 0,
                    description: `${label}, ${blocked ? "blocked by board" : "not in input range"}`,
                };
            }
            return {
                label,
                included: true,
                weight: group.ownReachWeight,
                description:
                    group.ownReachWeight === 0
                        ? `${label}, ${group.hands.length} combinations, zero own reach`
                        : `${label}, ${group.actions.map((action) => `${action.label} ${(action.probability * 100).toFixed(1)}%`).join(", ")}`,
                background: strategyGradient(group.actions),
            };
        });
    }, [node]);
    return <RangeMatrix cells={cells} {...selection} />;
}
