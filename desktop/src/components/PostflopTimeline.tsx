import { type Player, type SolverNode, describeActions, formatNumber, isForcedRunout } from "../solver";
import LockGlyph from "./LockGlyph";
import { BoardCard } from "./PlayingCards";
import TimelineChoice from "./TimelineChoice";

/** A decision's node lock as the timeline marks it. */
interface TimelineLock {
    /** Whether the current solution was solved with the lock. */
    applied: boolean;
}

export function PostflopTimeline({
    path,
    activeIndex,
    visible,
    disabled,
    players,
    locks,
    onPath,
    onAction,
    onBoard,
    onLock,
}: {
    path: SolverNode[];
    activeIndex: number;
    visible: boolean;
    disabled: boolean;
    players: Record<Player, string>;
    /** Node locks by NodeId. */
    locks: ReadonlyMap<number, TimelineLock>;
    onPath: (index: number) => void;
    onAction: (index: number, nodeId: number) => void;
    onBoard: (index: number) => void;
    /** Opens the lock editor at the active decision, whose column then shows a lock button. */
    onLock?: () => void;
}) {
    return (
        <>
            {path.map((node, index) => {
                const nextCard = path[index + 1]?.state.board[node.state.board.length];
                const active = visible && index === activeIndex;
                const lock = locks.get(node.nodeId);
                const lockClass = `timeline-lock${lock ? (lock.applied ? " applied" : " draft") : ""}`;
                const lockMark =
                    active && onLock ? (
                        <button
                            type="button"
                            className={lockClass}
                            aria-label={
                                lock ? `Edit lock, Lock ${lock.applied ? "applied" : "not applied"}` : "Lock node"
                            }
                            onClick={onLock}
                        >
                            <LockGlyph />
                        </button>
                    ) : (
                        lock && (
                            <span className={lockClass}>
                                <LockGlyph />
                            </span>
                        )
                    );
                return (
                    <section
                        key={`${node.nodeId}:${index}`}
                        className={`timeline-node${active ? " active" : ""}${index > activeIndex && visible ? " future" : ""}`}
                    >
                        <div className="timeline-node-heading">
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
                            {lockMark}
                        </div>
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
                                {node.result.reason === "fold" ? `${players[node.result.foldedBy]} folds` : "Showdown"}
                            </span>
                        )}
                    </section>
                );
            })}
        </>
    );
}
