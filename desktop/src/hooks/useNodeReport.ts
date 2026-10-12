import { useEffect, useState } from "react";
import { type QueryKind, type SolverReports, querySolver } from "../solver";

/**
 * One kind of per-node report within one solution generation, shared so each node is requested once. Hooks tell
 * solutions apart by these objects, not by generations, which a new web-server session numbers from the start again.
 */
export interface ReportCache<K extends QueryKind> {
    generation: number;
    kind: K;
    reports: Map<number, Promise<SolverReports[K]>>;
}

/**
 * One solution's reports, which it keeps while it is current and while it is a locked solution's reference, so
 * returning to it requests nothing again.
 */
export interface SolutionCaches {
    node: ReportCache<"node">;
    equity: ReportCache<"equity">;
    opponentEv: ReportCache<"opponentEv">;
}

export const solutionCaches = (generation: number): SolutionCaches => ({
    node: { generation, kind: "node", reports: new Map() },
    equity: { generation, kind: "equity", reports: new Map() },
    opponentEv: { generation, kind: "opponentEv", reports: new Map() },
});

/** The node's report from the cache, requested once; a failed request leaves the cache, so a later one retries. */
export function requestReport<K extends QueryKind>({ generation, kind, reports }: ReportCache<K>, nodeId: number) {
    let pending = reports.get(nodeId);
    if (!pending) {
        pending = querySolver(kind, nodeId, generation);
        pending.catch(() => reports.delete(nodeId));
        reports.set(nodeId, pending);
    }
    return pending;
}

/** The node's report while enabled; there is none without a cache. */
export function useNodeReport<K extends QueryKind>(
    cache: ReportCache<K> | undefined,
    nodeId: number | undefined,
    enabled: boolean,
) {
    const [result, setResult] = useState<{
        cache: ReportCache<K>;
        nodeId: number;
        data?: SolverReports[K];
        error?: string;
    }>();
    if (result && result.cache !== cache) setResult(undefined);
    useEffect(() => {
        if (!enabled || !cache || nodeId === undefined) return;
        let cancelled = false;
        void requestReport(cache, nodeId).then(
            (data) => {
                if (!cancelled) setResult({ cache, nodeId, data });
            },
            (error) => {
                if (!cancelled) setResult({ cache, nodeId, error: String(error) });
            },
        );
        return () => {
            cancelled = true;
        };
    }, [enabled, cache, nodeId]);
    return result?.cache === cache && result?.nodeId === nodeId ? result : undefined;
}
