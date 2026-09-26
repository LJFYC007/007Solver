import { useEffect, useMemo, useRef, useState } from "react";
import { type SolverNode, querySolverNode } from "../solver";

interface Navigation {
    root?: SolverNode;
    generation?: number;
    path: SolverNode[];
    activeIndex: number;
}
export function useSolverNavigation(root?: SolverNode, generation?: number) {
    const [stored, setStored] = useState<Navigation>({ path: [], activeIndex: 0 });
    const current: Navigation =
        stored.root === root && stored.generation === generation
            ? stored
            : { root, generation, path: root ? [root] : [], activeIndex: 0 };
    if (current !== stored) setStored(current);
    const [pending, setPending] = useState<{ generation?: number; error?: string; busy: boolean }>();
    const navigationRequest = useRef(0);
    const cache = useMemo(() => ({ generation, nodes: new Map<number, SolverNode>() }), [generation]);
    useEffect(
        () => () => {
            navigationRequest.current++;
        },
        [generation],
    );
    const node = current.path[current.activeIndex];
    const navigationPending = pending?.generation === generation && pending?.busy;
    async function selectChild(nodeId: number, parentIndex = current.activeIndex) {
        if (generation === undefined || navigationPending) return;
        const request = ++navigationRequest.current;
        setPending({ generation, busy: true });
        try {
            let child = cache.nodes.get(nodeId);
            if (!child) {
                child = await querySolverNode(nodeId, generation);
                cache.nodes.set(nodeId, child);
            }
            if (request !== navigationRequest.current) return;
            const selected = child;
            setStored((previous) => {
                if (previous.root !== root || previous.generation !== generation) return previous;
                const path =
                    previous.path[parentIndex + 1]?.nodeId === selected.nodeId
                        ? previous.path
                        : [...previous.path.slice(0, parentIndex + 1), selected];
                return { ...previous, path, activeIndex: parentIndex + 1 };
            });
            setPending({ generation, busy: false });
            return child;
        } catch (error) {
            if (request === navigationRequest.current) setPending({ generation, busy: false, error: String(error) });
        }
    }
    return {
        ...current,
        node,
        navigationPending: !!navigationPending,
        navigationError: pending?.generation === generation ? pending?.error : undefined,
        selectChild,
        selectPath: (index: number) => {
            navigationRequest.current++;
            setPending(undefined);
            // Build on the latest state so a path update queued in the same tick is kept.
            setStored((previous) =>
                previous.root === root && previous.generation === generation
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
