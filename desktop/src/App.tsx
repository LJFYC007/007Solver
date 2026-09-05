import { useEffect, useState } from "react";
import SolverWorkspace from "./components/SolverWorkspace";
import { type SolverNode, type SolverStatus, getSolverStatus, querySolverNode } from "./solver";

function SolverLoading({ status }: { status: SolverStatus }) {
    const failed = status.state === "failed";
    const solving = status.state === "solving";
    const progress = solving
        ? Math.min(100, Math.max(0, (status.completedIterations / status.totalIterations) * 100))
        : undefined;
    const title = failed
        ? "Solver failed to start"
        : status.state === "starting"
          ? "Starting solver"
          : status.state === "buildingTree"
            ? "Building decision tree"
            : solving
              ? "Solving strategy"
              : "Loading strategy";
    const message = failed
        ? status.message
        : status.state === "starting"
          ? "Starting the C++ solver service…"
          : status.state === "buildingTree"
            ? `Preparing the game tree for ${status.totalIterations.toLocaleString()} solver iterations…`
            : status.state === "solving"
              ? `${status.completedIterations.toLocaleString()} of ${status.totalIterations.toLocaleString()} iterations completed.`
              : "Loading the solved strategy…";

    return (
        <main className={`solver-loading${failed ? " failed" : ""}`}>
            <div className="loading-brand">
                <div className="brand-mark">007</div>
                <div>
                    <strong>007 Solver</strong>
                    <span>Strategy explorer</span>
                </div>
            </div>
            {!failed && !solving && <span className="solver-spinner" />}
            <h1>{title}</h1>
            <p>{message}</p>
            {progress !== undefined && (
                <div className="solver-progress">
                    <div className="solver-progress-label">
                        <span>Solver progress</span>
                        <strong>{progress.toFixed(1)}%</strong>
                    </div>
                    <div
                        aria-label="Solver progress"
                        aria-valuemax={100}
                        aria-valuemin={0}
                        aria-valuenow={progress}
                        className="solver-progress-track"
                        role="progressbar"
                    >
                        <span style={{ width: `${progress}%` }} />
                    </div>
                </div>
            )}
            {!failed && <small>The strategy explorer will open automatically when solving completes.</small>}
        </main>
    );
}

export default function App() {
    const [root, setRoot] = useState<SolverNode>();
    const [status, setStatus] = useState<SolverStatus>({ state: "starting" });

    useEffect(() => {
        let cancelled = false;
        let pollTimer: number | undefined;

        async function refreshStatus() {
            try {
                const nextStatus = await getSolverStatus();
                if (cancelled) return;
                setStatus(nextStatus);

                if (nextStatus.state === "ready") {
                    const rootNode = await querySolverNode(nextStatus.rootNodeId);
                    if (!cancelled) setRoot(rootNode);
                    return;
                }

                if (nextStatus.state !== "failed") pollTimer = window.setTimeout(() => void refreshStatus(), 500);
            } catch (error) {
                if (!cancelled)
                    setStatus({
                        message: error instanceof Error ? error.message : String(error),
                        state: "failed",
                    });
            }
        }

        void refreshStatus();
        return () => {
            cancelled = true;
            if (pollTimer !== undefined) window.clearTimeout(pollTimer);
        };
    }, []);

    if (status.state !== "ready" || !root) return <SolverLoading status={status} />;
    return <SolverWorkspace iterations={status.iterations} nodeCount={status.nodeCount} root={root} />;
}
