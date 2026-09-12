import type { SolverStatus } from "../solver";
import { BoardCard } from "./PlayingCards";

export default function SolvePanel({
    board,
    status,
    busy,
    changing,
    ready,
    iterations,
    context,
    onBoard,
    onSolve,
    onCancel,
    onIterations,
}: {
    board: string[];
    status: SolverStatus;
    busy: boolean;
    changing: boolean;
    ready: boolean;
    iterations: number;
    context: string;
    onBoard: () => void;
    onSolve: () => void;
    onCancel: () => void;
    onIterations: (value: number) => void;
}) {
    return (
        <section className="solve-setup">
            <div className="panel-strip-title">
                <strong>
                    {busy
                        ? "Solving postflop"
                        : ready
                          ? "Solution ready"
                          : status.state === "failed"
                            ? "Solve failed"
                            : board.length === 3
                              ? "Ready to solve"
                              : "Select a flop"}
                </strong>
                <button type="button" disabled={busy || changing} onClick={onBoard}>
                    Select cards
                </button>
            </div>
            <div className="solve-setup-row">
                <button
                    type="button"
                    className="stage-cards"
                    aria-label="Choose flop cards"
                    disabled={busy || changing}
                    onClick={onBoard}
                >
                    {Array.from({ length: 3 }, (_, i) => (
                        <BoardCard key={i} card={board[i]} />
                    ))}
                </button>
                {busy ? (
                    <div className="solve-phase" role="status">
                        <strong>
                            {status.state === "solving"
                                ? status.completedIterations === status.totalIterations
                                    ? "Finalizing solution"
                                    : "Training strategy"
                                : status.state === "buildingTree"
                                  ? "Preparing solution"
                                  : "Starting solver"}
                        </strong>
                        <small>
                            {status.state === "solving"
                                ? `${Math.round((status.completedIterations / status.totalIterations) * 100)}% complete`
                                : "Please wait…"}
                        </small>
                    </div>
                ) : (
                    <button className="solve-button" disabled={board.length !== 3 || changing} onClick={onSolve}>
                        {ready ? "View solution" : status.state === "failed" ? "Retry solve" : "Solve postflop"}
                    </button>
                )}
            </div>
            {busy && (
                <>
                    <progress
                        aria-label="Postflop solve progress"
                        max={status.state === "solving" ? status.totalIterations : 1}
                        value={status.state === "solving" ? status.completedIterations : undefined}
                    />
                    <button className="quiet-button cancel-solve" disabled={changing} onClick={onCancel}>
                        Cancel solve
                    </button>
                </>
            )}
            {!busy && !ready && (
                <details className="solve-settings">
                    <summary>Solver settings</summary>
                    <label>
                        Iterations
                        <input
                            aria-label="Iterations"
                            type="number"
                            min="1"
                            max="2147483647"
                            step="1"
                            value={iterations}
                            onChange={(e) => onIterations(Number(e.target.value))}
                        />
                    </label>
                </details>
            )}
            <small className="solve-context">{context}</small>
        </section>
    );
}
