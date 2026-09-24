import { Fragment } from "react";
import { type Player, type SolverNode, describeActions, formatNumber, isForcedRunout } from "../solver";
import { BoardCard } from "./PlayingCards";
import TimelineChoice from "./TimelineChoice";

export function PostflopTimeline({
    path,
    activeIndex,
    visible,
    disabled,
    players,
    onPath,
    onAction,
    onBoard,
    onFlop,
}: {
    path: SolverNode[];
    activeIndex: number;
    visible: boolean;
    disabled: boolean;
    players: Record<Player, string>;
    onPath: (index: number) => void;
    onAction: (index: number, nodeId: number) => void;
    onBoard: (index: number) => void;
    onFlop: () => void;
}) {
    return (
        <>
            {path.map((node, index) => {
                const previous = path[index - 1];
                const board = node.state.board.slice(previous?.state.board.length ?? 0);
                const nextCard = path[index + 1]?.state.board[node.state.board.length];
                const active = visible && index === activeIndex;
                return (
                    <Fragment key={`${node.nodeId}:${index}`}>
                        {board.length > 0 && previous?.kind !== "chance" && (
                            <button
                                type="button"
                                className="board-stage"
                                disabled={disabled}
                                onClick={onFlop}
                                aria-label={`Change flop ${board.join(" ")}`}
                            >
                                <span className="timeline-node-title">
                                    <strong>{node.state.street.toUpperCase()}</strong>
                                    <span>{formatNumber(node.state.pot)}</span>
                                </span>
                                <span className="stage-cards">
                                    {board.map((c) => (
                                        <BoardCard key={c} card={c} />
                                    ))}
                                </span>
                            </button>
                        )}
                        <section
                            className={`timeline-node${active ? " active" : ""}${index > activeIndex && visible ? " future" : ""}`}
                        >
                            <button type="button" className="timeline-node-title" onClick={() => onPath(index)}>
                                <strong>
                                    {isForcedRunout(node)
                                        ? "All-in"
                                        : node.kind === "decision"
                                          ? players[node.actor]
                                          : node.kind === "chance"
                                            ? node.state.street === "flop"
                                                ? "TURN"
                                                : "RIVER"
                                            : "Result"}
                                </strong>
                                {node.kind === "decision" && <span>{formatNumber(node.state.stacks[node.actor])}</span>}
                            </button>
                            {node.kind === "decision" ? (
                                describeActions(node)
                                    .slice()
                                    .reverse()
                                    .map((a) => {
                                        const nextNodeId = node.actions[a.index].nextNodeId;
                                        return (
                                            <TimelineChoice
                                                key={nextNodeId}
                                                label={a.label}
                                                size={a.size}
                                                color={a.color}
                                                chosen={path[index + 1]?.nodeId === nextNodeId}
                                                disabled={disabled}
                                                onClick={() => onAction(index, nextNodeId)}
                                            />
                                        );
                                    })
                            ) : isForcedRunout(node) ? (
                                <span className="history-node-choice">Betting complete</span>
                            ) : node.kind === "chance" ? (
                                <button
                                    className="next-card-button"
                                    disabled={disabled}
                                    type="button"
                                    onClick={() => onBoard(index)}
                                    aria-label={`Choose ${node.state.street === "flop" ? "turn" : "river"}`}
                                >
                                    <BoardCard card={nextCard} />
                                    <span>{nextCard ? "Change card" : "Select card"}</span>
                                </button>
                            ) : (
                                <span className="history-node-choice">
                                    {node.result.reason === "fold"
                                        ? `${players[node.result.foldedBy]} folds`
                                        : "Showdown"}
                                </span>
                            )}
                        </section>
                    </Fragment>
                );
            })}
        </>
    );
}
