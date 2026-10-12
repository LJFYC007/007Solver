import { useEffect, useRef, useState } from "react";
import StudyWorkspace from "./components/StudyWorkspace";
import { type SolutionCaches, requestReport, solutionCaches } from "./hooks/useNodeReport";
import {
    type SolverNode,
    type SolverStatus,
    cancelSolver,
    getSolverStatus,
    restoreReference,
    solveScenario,
} from "./solver";
import { type NodeLock, lockedScenario } from "./solver/locks";
import type { PostflopScenario } from "./solver/preflop";

/** A locked solve: the locks it was started with and the reports of the GTO solution it keeps as the reference. */
interface LockedSolve {
    locks: readonly NodeLock[];
    reference: SolutionCaches;
}

export default function App() {
    const [root, setRoot] = useState<SolverNode>();
    const [status, setStatus] = useState<SolverStatus>({ state: "idle" });
    // The current solution's reports, which also name its generation.
    const [caches, setCaches] = useState<SolutionCaches>();
    const [scenario, setScenario] = useState<PostflopScenario>();
    // Set while the current solve is a locked one.
    const [lockedSolve, setLockedSolve] = useState<LockedSolve>();
    const reference = lockedSolve?.reference;
    const [applying, setApplying] = useState(false);
    const [changing, setChanging] = useState(false);
    const request = useRef(0);
    const invalidation = useRef<Promise<void> | undefined>(undefined);
    const restoring = useRef(false);
    // The unlocked scenario, which locked solves extend.
    const baseline = scenario && { ...scenario, locks: undefined };
    /**
     * Solves the scenario as a new request, keeping the `locking` solve's reference; when the solve cannot start,
     * the kept solution stays the one to retry from or return to.
     */
    async function solve(next: PostflopScenario, locking?: LockedSolve) {
        const epoch = ++request.current;
        setScenario(next);
        // Set before the bridge holds it: control requests keep their order, so a restore follows this start.
        setLockedSolve(locking);
        setRoot(undefined);
        setCaches(undefined);
        setStatus({ state: "starting" });
        try {
            const generation = await solveScenario(next, locking?.reference.node.generation);
            if (epoch === request.current) setCaches(solutionCaches(generation));
        } catch (error) {
            if (epoch === request.current) setStatus({ state: "failed", message: String(error) });
        }
    }
    async function start(next: PostflopScenario) {
        if (invalidation.current || restoring.current) return;
        await solve(next);
    }
    /**
     * Solves the baseline with the locks; the GTO solution stays queryable as the reference. Rejects, leaving the
     * current solution as it is, when the earlier decisions' strategies cannot be read.
     */
    async function applyLocks(next: NodeLock[]) {
        // Earlier decisions lock to the GTO solution's strategies, which keeps the reports loaded so far.
        const source = reference ?? caches;
        if (restoring.current || !baseline || !source) return;
        // The current solution stays, and keeps polling, until the locked scenario is ready.
        const epoch = request.current;
        // A newer request outlives these locks, and so does Back to GTO from when it starts.
        const outlived = () => epoch !== request.current || restoring.current;
        setApplying(true);
        let locked: PostflopScenario;
        try {
            locked = await lockedScenario(baseline, next, (nodeId) => requestReport(source.node, nodeId));
        } catch (error) {
            // A failure that the solution or study has outlived is no news.
            if (outlived()) return;
            throw error;
        } finally {
            setApplying(false);
        }
        if (outlived()) return;
        await solve(locked, { locks: next, reference: source });
    }
    /**
     * Makes the GTO solution current again, resolving whether this call did. Rejects, leaving the current solution as
     * it is, when the bridge cannot.
     */
    async function restore() {
        // Synchronous double clicks run both handlers before React re-renders.
        if (restoring.current || !reference) return false;
        restoring.current = true;
        // The locked solution stays current, and keeps polling, until the bridge has swapped the solutions.
        const epoch = request.current;
        try {
            // The bridge restores it under its own generation, so its reports stay valid.
            const restored = await restoreReference();
            // Its root's report is cached, so the view returns without waiting for the next status poll.
            const restoredRoot =
                restored.state === "ready" ? await requestReport(reference.node, restored.rootNodeId) : undefined;
            if (epoch !== request.current) return false;
            request.current++;
            setScenario(baseline);
            setLockedSolve(undefined);
            setRoot(restoredRoot);
            // Ready at once: a starting state would offer a Cancel solve that ends the study.
            setStatus(restored);
            setCaches(reference);
            return true;
        } finally {
            restoring.current = false;
        }
    }
    function invalidate(): Promise<void> {
        if (invalidation.current) return invalidation.current;
        if (status.state === "idle") return Promise.resolve();
        const epoch = ++request.current;
        setChanging(true);
        invalidation.current = cancelSolver()
            .then(() => {
                if (epoch !== request.current) return;
                setRoot(undefined);
                setCaches(undefined);
                setScenario(undefined);
                setLockedSolve(undefined);
                setStatus({ state: "idle" });
            })
            .catch((error) => {
                // A pending start belongs to the old request and cannot restore polling.
                if (epoch === request.current && caches === undefined)
                    setStatus({ state: "failed", message: String(error) });
                throw error;
            })
            .finally(() => {
                invalidation.current = undefined;
                setChanging(false);
            });
        return invalidation.current;
    }
    useEffect(() => {
        if (!caches || changing) return;
        const active = caches;
        const epoch = request.current;
        let cancelled = false;
        let timer: number | undefined;
        async function poll() {
            try {
                const next = await getSolverStatus();
                if (cancelled || epoch !== request.current) return;
                setStatus(next);
                if (next.state === "ready") {
                    const value = await requestReport(active.node, next.rootNodeId);
                    if (!cancelled && epoch === request.current) setRoot(value);
                } else if (next.state !== "failed") timer = window.setTimeout(() => void poll(), 500);
            } catch (error) {
                if (!cancelled && epoch === request.current) setStatus({ state: "failed", message: String(error) });
            }
        }
        void poll();
        return () => {
            cancelled = true;
            if (timer !== undefined) window.clearTimeout(timer);
        };
    }, [caches, changing]);
    return (
        <StudyWorkspace
            root={root}
            status={status}
            caches={caches}
            scenario={scenario}
            changing={changing}
            reference={reference}
            // A failed solve leaves no solution its locks applied to.
            appliedLocks={status.state === "failed" ? undefined : lockedSolve?.locks}
            applying={applying}
            onSolve={start}
            onInvalidate={invalidate}
            onApplyLocks={applyLocks}
            onRestore={restore}
        />
    );
}
