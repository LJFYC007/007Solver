import { useEffect, useState } from "react";
import type { SolverStatus } from "../solver";
import BettingTreeSettings from "./BettingTreeSettings";
import { BoardCard } from "./PlayingCards";
import type { PostflopScenario } from "../solver/preflop";
import { parseBettingTree, type BettingTreeDraft } from "../solver/bettingTree";
import { stopReasonLabel } from "../solver/study";

export default function SolvePanel({
    board,
    status,
    busy,
    changing,
    iterations,
    accuracyPercent,
    settingsChanged,
    onSolve,
    onCancel,
    onReset,
    onIterations,
    onAccuracyPercent,
    bettingTree,
    onBettingTree,
    scenario,
    matchup,
    error,
}: {
    scenario?: PostflopScenario;
    matchup?: { title: string; detail: string; oop: string; ip: string };
    error?: string;
    board: string[];
    status: SolverStatus;
    busy: boolean;
    changing: boolean;
    iterations: number;
    accuracyPercent: number;
    settingsChanged: boolean;
    onSolve: () => void;
    onCancel: () => void;
    onReset: () => void;
    onIterations: (value: number) => void;
    onAccuracyPercent: (value: number) => void;
    bettingTree: BettingTreeDraft;
    onBettingTree: (value: BettingTreeDraft) => void;
}) {
    const progress = status.state === "solving" ? status : undefined;
    const result = status.state === "ready" ? status : undefined;
    const sample = progress?.elapsedSeconds;
    const [tick, setTick] = useState<{ sample?: number; age: number }>({ age: 0 });
    useEffect(() => {
        if (sample === undefined) return;
        const receivedAt = performance.now();
        const timer = window.setInterval(() => setTick({ sample, age: (performance.now() - receivedAt) / 1000 }), 250);
        return () => window.clearInterval(timer);
    }, [sample]);
    const age = tick.sample === sample ? tick.age : 0;
    const elapsed = result?.elapsedSeconds ?? (sample === undefined ? undefined : sample + age);
    const remaining =
        progress?.estimatedRemainingSeconds == null ? undefined : progress.estimatedRemainingSeconds - age;
    const timeProgress =
        elapsed !== undefined && remaining !== undefined
            ? Math.min(0.99, elapsed / (elapsed + Math.max(1, remaining)))
            : undefined;
    const accuracy = result?.accuracyPercent ?? progress?.accuracyPercent;
    const updates = result?.iterations ?? progress?.completedIterations;
    const limit = result ? scenario?.iterations : progress?.totalIterations;
    const target = result?.targetAccuracyPercent ?? progress?.targetAccuracyPercent;
    let treeError: string | undefined;
    try {
        parseBettingTree(bettingTree);
    } catch (failure) {
        treeError = failure instanceof Error ? failure.message : String(failure);
    }
    const phase =
        status.state === "solving"
            ? status.phase === "finalizing"
                ? "Finalizing solution"
                : status.phase === "checking"
                  ? "Checking accuracy"
                  : "Training strategy"
            : status.state === "buildingTree"
              ? "Preparing solution"
              : status.state === "queued"
                ? "Waiting for GPU"
                : "Starting solver";
    return (
        <section className="solve-setup" aria-label="Postflop solver">
            <header className="solve-heading">
                <h2>Postflop solver</h2>
                <span className={`solve-state${result?.stopReason === "accuracy" ? " reached" : ""}`}>
                    {result
                        ? stopReasonLabel(result.stopReason)
                        : busy
                          ? "Solving"
                          : status.state === "failed"
                            ? "Failed"
                            : "Setup"}
                </span>
            </header>
            {matchup && (
                <div className="solve-spot">
                    <span className="solve-spot-board">
                        {Array.from({ length: 3 }, (_, i) => (
                            <BoardCard key={i} card={board[i]} />
                        ))}
                    </span>
                    <span>
                        <strong>{matchup.title}</strong>
                        <small>{matchup.detail}</small>
                    </span>
                </div>
            )}
            {/* Messages come before the settings, which can scroll far. */}
            {error && (
                <p className="solve-error" role="alert">
                    {error}
                </p>
            )}
            {status.state === "failed" && (
                <p className="solve-error" role="alert">
                    {status.message}
                </p>
            )}
            {board.length !== 3 && <p className="solve-hint">Choose a flop in the action history to continue.</p>}
            {(busy || result) && (
                <div className="solve-report">
                    <dl className="solve-metrics">
                        <div>
                            <dt>Elapsed</dt>
                            <dd>{elapsed === undefined ? "—" : `${elapsed.toFixed(1)}s`}</dd>
                        </div>
                        <div>
                            <dt>Updates</dt>
                            <dd>{updates?.toLocaleString() ?? "—"}</dd>
                            <small>{limit ? `of ${limit.toLocaleString()}` : "Preparing"}</small>
                        </div>
                        <div>
                            <dt>Exploitability</dt>
                            <dd>{accuracy == null ? "—" : `${accuracy.toPrecision(3)}%`}</dd>
                            <small>{target === undefined ? "Initial pot" : `Target ${target}% · initial pot`}</small>
                        </div>
                    </dl>
                    {busy && (
                        <div className="solve-phase" role="status">
                            <div>
                                <strong>{phase}</strong>
                                <span>
                                    {remaining === undefined
                                        ? "Estimating time…"
                                        : remaining <= 0
                                          ? "Updating estimate…"
                                          : `~${Math.ceil(remaining)}s remaining`}
                                </span>
                            </div>
                            <progress aria-label="Estimated solve time progress" max={1} value={timeProgress} />
                        </div>
                    )}
                </div>
            )}
            {!busy && (
                <section className="solve-settings" aria-label="Solve settings">
                    <header>
                        <h3>Settings</h3>
                        <button
                            type="button"
                            className="quiet-button"
                            disabled={!settingsChanged || changing}
                            title="Restore the default settings"
                            onClick={onReset}
                        >
                            Reset
                        </button>
                    </header>
                    <div className="solve-targets">
                        <label>
                            Target exploitability
                            <div className="solve-input-unit">
                                <input
                                    aria-label="Target exploitability (% initial pot)"
                                    type="number"
                                    min="0.000001"
                                    step="any"
                                    disabled={changing}
                                    value={Number.isNaN(accuracyPercent) ? "" : accuracyPercent}
                                    onChange={(e) => onAccuracyPercent(e.target.valueAsNumber)}
                                />
                                <span>%</span>
                            </div>
                            <small>of initial pot</small>
                        </label>
                        <label>
                            Update limit
                            <input
                                aria-label="Update limit"
                                type="number"
                                min="1"
                                max="2147483647"
                                step="1"
                                disabled={changing}
                                value={Number.isNaN(iterations) ? "" : iterations}
                                onChange={(e) => onIterations(e.target.valueAsNumber)}
                            />
                        </label>
                    </div>
                    <BettingTreeSettings
                        value={bettingTree}
                        players={matchup}
                        disabled={changing}
                        error={treeError}
                        onChange={onBettingTree}
                    />
                </section>
            )}
            <footer className="solve-footer">
                {busy ? (
                    <button type="button" className="quiet-button" disabled={changing} onClick={onCancel}>
                        Cancel solve
                    </button>
                ) : (
                    <button
                        type="button"
                        className="solve-button"
                        disabled={board.length !== 3 || changing || !!treeError}
                        onClick={onSolve}
                    >
                        {result ? "Solve again" : status.state === "failed" ? "Retry solve" : "Solve"}
                    </button>
                )}
            </footer>
        </section>
    );
}
