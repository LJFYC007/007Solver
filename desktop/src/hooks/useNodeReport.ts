import { useEffect, useState } from "react";
import { type QueryKind, type SolverReports, querySolver } from "../solver";

/** One kind of per-node report within one solution generation, shared so each node is requested once. */
export interface ReportCache<K extends QueryKind> {
    generation?: number;
    kind: K;
    reports: Map<number, Promise<SolverReports[K]>>;
}

export function useNodeReport<K extends QueryKind>(
    { generation, kind, reports }: ReportCache<K>,
    nodeId: number | undefined,
    enabled: boolean,
) {
    const [result, setResult] = useState<{
        generation: number;
        nodeId: number;
        data?: SolverReports[K];
        error?: string;
    }>();
    if (result && result.generation !== generation) setResult(undefined);
    useEffect(() => {
        if (!enabled || generation === undefined || nodeId === undefined) return;
        let cancelled = false;
        let pending = reports.get(nodeId);
        if (!pending) {
            pending = querySolver(kind, nodeId, generation);
            pending.catch(() => reports.delete(nodeId));
            reports.set(nodeId, pending);
        }
        void pending.then(
            (data) => {
                if (!cancelled) setResult({ generation, nodeId, data });
            },
            (error) => {
                if (!cancelled) setResult({ generation, nodeId, error: String(error) });
            },
        );
        return () => {
            cancelled = true;
        };
    }, [enabled, generation, kind, reports, nodeId]);
    return result?.generation === generation && result?.nodeId === nodeId ? result : undefined;
}
