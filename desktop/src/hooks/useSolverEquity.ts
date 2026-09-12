import { useEffect, useMemo, useState } from "react";
import { type EquityReport, querySolverEquity } from "../solver";

export function useSolverEquity(generation: number | undefined, nodeId: number | undefined, enabled: boolean) {
    const cache = useMemo(() => ({ generation, reports: new Map<number, Promise<EquityReport>>() }), [generation]);
    const [result, setResult] = useState<{ generation: number; nodeId: number; data?: EquityReport; error?: string }>();
    if (result && result.generation !== generation) setResult(undefined);
    useEffect(() => {
        if (!enabled || generation === undefined || nodeId === undefined) return;
        let cancelled = false;
        let pending = cache.reports.get(nodeId);
        if (!pending) {
            pending = querySolverEquity(nodeId, generation);
            cache.reports.set(nodeId, pending);
        }
        void pending.then(
            (data) => {
                if (!cancelled) setResult({ generation, nodeId, data });
            },
            (error) => {
                cache.reports.delete(nodeId);
                if (!cancelled) setResult({ generation, nodeId, error: String(error) });
            },
        );
        return () => {
            cancelled = true;
        };
    }, [enabled, generation, nodeId, cache]);
    return result?.generation === generation && result?.nodeId === nodeId ? result : undefined;
}
