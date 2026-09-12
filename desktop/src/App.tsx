import { useEffect, useRef, useState } from "react";
import StudyWorkspace from "./components/StudyWorkspace";
import {
    type SolverNode,
    type SolverStatus,
    cancelSolver,
    getSolverStatus,
    querySolverNode,
    solveScenario,
} from "./solver";
import type { PostflopScenario } from "./solver/preflop";

export default function App() {
    const [root, setRoot] = useState<SolverNode>();
    const [status, setStatus] = useState<SolverStatus>({ state: "idle" });
    const [generation, setGeneration] = useState<number>();
    const [scenario, setScenario] = useState<PostflopScenario>();
    const [changing, setChanging] = useState(false);
    const request = useRef(0);
    const invalidation = useRef<Promise<void> | undefined>(undefined);
    async function start(next: PostflopScenario) {
        if (invalidation.current) return;
        const epoch = ++request.current;
        setScenario(next);
        setRoot(undefined);
        setGeneration(undefined);
        setStatus({ state: "starting" });
        try {
            const value = await solveScenario(next);
            if (epoch === request.current) setGeneration(value);
        } catch (error) {
            if (epoch === request.current) setStatus({ state: "failed", message: String(error) });
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
                setGeneration(undefined);
                setScenario(undefined);
                setStatus({ state: "idle" });
            })
            .catch((error) => {
                // A pending start belongs to the old request and cannot restore polling.
                if (epoch === request.current && generation === undefined)
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
        if (generation === undefined || changing) return;
        const activeGeneration = generation;
        const epoch = request.current;
        let cancelled = false;
        let timer: number | undefined;
        async function poll() {
            try {
                const next = await getSolverStatus();
                if (cancelled || epoch !== request.current) return;
                setStatus(next);
                if (next.state === "ready") {
                    const value = await querySolverNode(next.rootNodeId, activeGeneration);
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
    }, [generation, changing]);
    return (
        <StudyWorkspace
            root={root}
            status={status}
            generation={generation}
            scenario={scenario}
            changing={changing}
            onSolve={start}
            onInvalidate={invalidate}
        />
    );
}
