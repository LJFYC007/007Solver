import { useRef, useState } from "react";
import { type SolverNode, querySolverNode } from "../solver";

export interface PathEntry {
    label: string;
    node: SolverNode;
}

export function useSolverNavigation(root: SolverNode) {
    const [path, setPath] = useState<PathEntry[]>([{ label: "Root", node: root }]);
    const [activeIndex, setActiveIndex] = useState(0);
    const [navigationError, setNavigationError] = useState<string>();
    const [navigationPending, setNavigationPending] = useState(false);
    const nodeCache = useRef(new Map<number, SolverNode>([[root.nodeId, root]]));
    const navigationRequest = useRef(0);
    const node = path[activeIndex].node;

    async function selectChild(nodeId: number, label: string) {
        if (navigationPending) return;

        const parentIndex = activeIndex;
        const request = ++navigationRequest.current;
        setNavigationError(undefined);
        setNavigationPending(true);
        try {
            let child = nodeCache.current.get(nodeId);
            if (!child) {
                child = await querySolverNode(nodeId);
                nodeCache.current.set(nodeId, child);
            }
            if (request !== navigationRequest.current) return;
            setPath((current) => {
                if (current[parentIndex + 1]?.node.nodeId === child.nodeId) return current;
                return [...current.slice(0, parentIndex + 1), { label, node: child }];
            });
            setActiveIndex(parentIndex + 1);
        } catch (error) {
            if (request === navigationRequest.current)
                setNavigationError(error instanceof Error ? error.message : String(error));
        } finally {
            if (request === navigationRequest.current) setNavigationPending(false);
        }
    }

    return {
        activeIndex,
        navigationError,
        navigationPending,
        node,
        path,
        selectChild,
        selectPath: (index: number) => {
            navigationRequest.current++;
            setNavigationError(undefined);
            setNavigationPending(false);
            setActiveIndex(index);
        },
    };
}
