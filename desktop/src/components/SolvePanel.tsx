import { useEffect, useState } from "react";
import type { SolverStatus } from "../solver";
import { BoardCard } from "./PlayingCards";
import BettingTreeSettings from "./BettingTreeSettings";
import type { BettingTreeDraft } from "../solver/bettingTree";

export default function SolvePanel({
    board,
    status,
    busy,
    changing,
    ready,
    iterations,
    accuracyPercent,
    context,
    onBoard,
    onSolve,
    onResolve,
    onCancel,
    onIterations,
    onAccuracyPercent,
    bettingTree,
    onBettingTree,
}: {
    board: string[];
    status: SolverStatus;
    busy: boolean;
    changing: boolean;
    ready: boolean;
    iterations: number;
    accuracyPercent: number;
    context: string;
    onBoard: () => void;
    onSolve: () => void;
    onResolve: () => void;
    onCancel: () => void;
    onIterations: (value: number) => void;
    onAccuracyPercent: (value: number) => void;
    bettingTree: BettingTreeDraft;
    onBettingTree: (value: BettingTreeDraft) => void;
}) {
    const progress = status.state === "solving" ? status : undefined;
    const sample = progress?.elapsedSeconds;
    const [tick, setTick] = useState<{ sample?: number; age: number }>({ age: 0 });
    useEffect(() => {
        if (sample === undefined) return;
        const receivedAt = performance.now();
        const timer = window.setInterval(() => setTick({ sample, age: (performance.now() - receivedAt) / 1000 }), 250);
        return () => window.clearInterval(timer);
    }, [sample]);
    const age = tick.sample === sample ? tick.age : 0;
    const elapsed = (sample ?? 0) + age;
    const remaining =
        progress?.estimatedRemainingSeconds == null ? undefined : progress.estimatedRemainingSeconds - age;
    const timeProgress =
        progress && remaining !== undefined ? Math.min(0.99, elapsed / (elapsed + Math.max(1, remaining))) : undefined;
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
                                ? status.phase === "finalizing"
                                    ? "Finalizing solution"
                                    : status.phase === "checking"
                                      ? "Checking accuracy"
                                      : "Training strategy"
                                : status.state === "buildingTree"
                                  ? "Preparing solution"
                                  : "Starting solver"}
                        </strong>
                        <small>
                            {status.state === "solving"
                                ? `${remaining === undefined ? "Measuring convergence…" : remaining <= 0 ? "Updating time estimate…" : `About ${Math.ceil(remaining)} seconds remaining`} · ${Math.floor(elapsed)}s elapsed`
                                : "Please wait…"}
                        </small>
                        {progress && (
                            <small>
                                {progress.accuracyPercent === null
                                    ? `Target ${progress.targetAccuracyPercent}% pot`
                                    : `${progress.accuracyPercent.toPrecision(3)}% pot · Target ${progress.targetAccuracyPercent}%`}
                            </small>
                        )}
                        {progress && (
                            <small>
                                {progress.completedIterations.toLocaleString()} /{" "}
                                {progress.totalIterations.toLocaleString()} updates
                                {progress.estimate
                                    ? ` · ${(progress.estimate.peakBytes / 1024 ** 3).toFixed(2)} GiB estimated`
                                    : ""}
                            </small>
                        )}
                    </div>
                ) : (
                    <button className="solve-button" disabled={board.length !== 3 || changing} onClick={onSolve}>
                        {ready ? "View solution" : status.state === "failed" ? "Retry solve" : "Solve postflop"}
                    </button>
                )}
            </div>
            {busy && (
                <>
                    <progress aria-label="Estimated solve time progress" max={1} value={timeProgress} />
                    <button className="quiet-button cancel-solve" disabled={changing} onClick={onCancel}>
                        Cancel solve
                    </button>
                </>
            )}
            {!busy && (
                <details className="solve-settings">
                    <summary>Solver settings</summary>
                    <label>
                        Accuracy (% pot)
                        <input
                            aria-label="Accuracy (% pot)"
                            type="number"
                            min="0.000001"
                            step="any"
                            disabled={changing}
                            value={accuracyPercent}
                            onChange={(e) => onAccuracyPercent(Number(e.target.value))}
                        />
                    </label>
                    <label>
                        Iteration limit
                        <input
                            aria-label="Iteration limit"
                            type="number"
                            min="1"
                            max="2147483647"
                            step="1"
                            disabled={changing}
                            value={iterations}
                            onChange={(e) => onIterations(Number(e.target.value))}
                        />
                    </label>
                    {ready && (
                        <button type="button" className="solve-button" disabled={changing} onClick={onResolve}>
                            Solve again
                        </button>
                    )}
                </details>
            )}
            <BettingTreeSettings value={bettingTree} disabled={busy || changing} onChange={onBettingTree} />
            <small className="solve-context">{context}</small>
        </section>
    );
}
