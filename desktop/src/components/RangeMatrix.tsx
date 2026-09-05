import type { CSSProperties } from "react";
import { type HandGroup, RANKS, handClassCombos, matrixLabel, strategyGradient } from "../solver";

export default function RangeMatrix({
    groups,
    board,
    selected,
    onHover,
}: {
    groups: Map<string, HandGroup>;
    board: string[];
    selected?: string;
    onHover: (label: string) => void;
}) {
    return (
        <div className="range-grid" aria-label="13 by 13 strategy matrix" role="grid">
            {RANKS.flatMap((_, row) =>
                RANKS.map((__, column) => {
                    const label = matrixLabel(row, column);
                    const group = groups.get(label);
                    const blockedByBoard = !handClassCombos(label).some((cards) =>
                        cards.every((card) => !board.includes(card)),
                    );
                    const description = group
                        ? `${label}, ${group.hands.length} combinations${group.ownReachWeight === 0 ? ", zero own reach" : ""}`
                        : `${label}, ${blockedByBoard ? "blocked by board" : "not in input range"}`;
                    const style = group
                        ? ({
                              "--strategy-background": strategyGradient(group.actions) ?? "#1b1b1b",
                              "--strategy-height": `${Math.min(1, Math.max(0, group.ownReachWeight)) * 100}%`,
                          } as CSSProperties)
                        : undefined;

                    return (
                        <div
                            aria-label={description}
                            aria-selected={selected === label}
                            className={`range-cell${group ? " in-range" : ""}${group?.ownReachWeight === 0 ? " zero-own-reach" : ""}`}
                            key={label}
                            onMouseEnter={() => group && onHover(label)}
                            role="gridcell"
                            style={style}
                            title={description}
                        >
                            <strong>{label}</strong>
                        </div>
                    );
                }),
            )}
        </div>
    );
}
