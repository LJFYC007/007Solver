import { useEffect, useRef, useState } from "react";
import type { SolverNode } from "../solver";
import { type ReportCache, requestReport } from "./useNodeReport";

interface Navigation {
    root?: SolverNode;
    cache?: ReportCache<"node">;
    path: SolverNode[];
    activeIndex: number;
}
export function useSolverNavigation(root?: SolverNode, cache?: ReportCache<"node">) {
    const [stored, setStored] = useState<Navigation>({ path: [], activeIndex: 0 });
    const current: Navigation =
        stored.root === root && stored.cache === cache
            ? stored
            : { root, cache, path: root ? [root] : [], activeIndex: 0 };
    if (current !== stored) setStored(current);
    const [pending, setPending] = useState<{ cache?: ReportCache<"node">; error?: string; busy: boolean }>();
    if (pending && pending.cache !== cache) setPending(undefined);
    const navigationRequest = useRef(0);
    useEffect(
        () => () => {
            navigationRequest.current++;
        },
        [cache],
    );
    const node = current.path[current.activeIndex];
    const navigationPending = !!pending?.busy;
    /** Loads nodes, then changes the path with them unless a later request or another solution came first. */
    async function navigate<T>(load: () => Promise<T>, change: (previous: Navigation, loaded: T) => Navigation) {
        const request = ++navigationRequest.current;
        setPending({ cache, busy: true });
        try {
            const loaded = await load();
            if (request !== navigationRequest.current) return;
            setStored((previous) =>
                previous.root === root && previous.cache === cache ? change(previous, loaded) : previous,
            );
            setPending({ cache, busy: false });
            return loaded;
        } catch (error) {
            if (request === navigationRequest.current) setPending({ cache, busy: false, error: String(error) });
        }
    }
    async function selectChild(nodeId: number, parentIndex = current.activeIndex) {
        if (!cache || navigationPending) return;
        return navigate(
            () => requestReport(cache, nodeId),
            (previous, child) => ({
                ...previous,
                path:
                    previous.path[parentIndex + 1]?.nodeId === child.nodeId
                        ? previous.path
                        : [...previous.path.slice(0, parentIndex + 1), child],
                activeIndex: parentIndex + 1,
            }),
        );
    }
    /** Shows the path through `nodeIds`, root first, such as a node lock's in another solve of the same tree. */
    async function openPath(nodeIds: readonly number[], activeIndex = nodeIds.length - 1) {
        if (!cache) return;
        // The root's report is in the cache, where the app loaded it.
        const path = await navigate(
            () => Promise.all(nodeIds.map((nodeId) => requestReport(cache, nodeId))),
            (previous, path) => ({ ...previous, path, activeIndex }),
        );
        return path?.[activeIndex];
    }
    return {
        ...current,
        node,
        navigationPending,
        navigationError: pending?.error,
        selectChild,
        openPath,
        selectPath: (index: number) => {
            navigationRequest.current++;
            setPending(undefined);
            // Build on the latest state so a path update queued in the same tick is kept.
            setStored((previous) =>
                previous.root === root && previous.cache === cache
                    ? { ...previous, activeIndex: index }
                    : { ...current, activeIndex: index },
            );
        },
        suspend: () => {
            navigationRequest.current++;
            setPending(undefined);
        },
    };
}
