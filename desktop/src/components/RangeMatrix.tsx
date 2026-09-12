import { type CSSProperties, useMemo } from "react";
import {
    type DecisionNode,
    HAND_CLASSES,
    groupHands,
    handClassCombos,
    orderedActions,
    strategyGradient,
} from "../solver";
import type { PreflopNode } from "../solver/catalog";
import { preflopActionColor, probabilities } from "../solver/preflop";

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

function RangeMatrix({ cells, selected, onSelect }: Selection & { cells: MatrixCell[] }) {
    return (
        <div className="range-grid" aria-label="13 by 13 strategy matrix" role="grid">
            {cells.map((cell) => (
                <button
                    type="button"
                    role="gridcell"
                    key={cell.label}
                    aria-label={cell.description}
                    aria-selected={selected === cell.label}
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
                                  color: preflopActionColor(action),
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
            const blocked = !handClassCombos(label).some((cards) =>
                cards.every((card) => !node.state.board.includes(card)),
            );
            return {
                label,
                included: !!group,
                weight: group?.ownReachWeight ?? 0,
                description: group
                    ? `${label}, ${group.hands.length} combinations${group.ownReachWeight === 0 ? ", zero own reach" : ""}`
                    : `${label}, ${blocked ? "blocked by board" : "not in input range"}`,
                background: group ? strategyGradient(orderedActions(group.actions)) : undefined,
            };
        });
    }, [node]);
    return <RangeMatrix cells={cells} {...selection} />;
}
