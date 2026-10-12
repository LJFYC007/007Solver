import { useEffect, useMemo, useState } from "react";
import { type HandFilters, NO_FILTERS } from "../solver/display";
import type { SolutionCaches } from "./useNodeReport";

interface LockSession {
    /** The matrix classes the selection consists of; empty while the filters, or the whole range, make it. */
    cells: ReadonlySet<string>;
    /** The editor's own filters, apart from the study's, so they end with the editing. */
    filters: HandFilters;
    /** Actions, by node index, that keep their frequency while another one changes. */
    pinned: ReadonlySet<number>;
    /** Whether the toolbar's Show GTO is on. */
    gto: boolean;
}

const EMPTY = new Set<never>();

/**
 * The editing of the shown decision's node lock in the solution with `caches`, with its selection, filters and pins,
 * and whether the matrix shows GTO: while the toolbar's Show GTO is on or Alt is held.
 */
export function useLockEditor(caches?: SolutionCaches) {
    const [session, setSession] = useState<LockSession>();
    const [altHeld, setAltHeld] = useState(false);
    // A session belongs to the solution it was opened in.
    const [solution, setSolution] = useState(caches);
    if (solution !== caches) {
        setSolution(caches);
        setSession(undefined);
    }
    const editing = session !== undefined;
    useEffect(() => {
        if (!editing) return;
        function onKey(event: KeyboardEvent) {
            if (event.key !== "Alt") return;
            // Keeps the browser and window menus from taking focus.
            event.preventDefault();
            setAltHeld(event.type === "keydown");
        }
        const release = () => setAltHeld(false);
        window.addEventListener("keydown", onKey);
        window.addEventListener("keyup", onKey);
        window.addEventListener("blur", release);
        return () => {
            window.removeEventListener("keydown", onKey);
            window.removeEventListener("keyup", onKey);
            window.removeEventListener("blur", release);
            // The keyup of an Alt held while the editor closes would go unseen.
            release();
        };
    }, [editing]);
    // Stable, so callbacks built on them stay stable too.
    const actions = useMemo(
        () => ({
            open: () => setSession({ cells: EMPTY, filters: NO_FILTERS, pinned: EMPTY, gto: false }),
            close: () => setSession(undefined),
            /**
             * Replaces the selected classes, adds to them or toggles them; clicking the one selected class again
             * selects the whole range. Clicked classes replace the filters as the selection, and filters replace the
             * clicked classes.
             */
            select: (labels: string[], how: "replace" | "add" | "toggle") =>
                setSession((s) => {
                    if (!s) return s;
                    const again =
                        how === "replace" && labels.length === 1 && s.cells.size === 1 && s.cells.has(labels[0]);
                    const cells = new Set(how === "replace" ? [] : s.cells);
                    if (!again)
                        for (const label of labels)
                            if (how === "toggle" && cells.has(label)) cells.delete(label);
                            else cells.add(label);
                    return { ...s, cells, filters: NO_FILTERS };
                }),
            setFilters: (filters: HandFilters) => setSession((s) => s && { ...s, cells: EMPTY, filters }),
            togglePinned: (action: number) =>
                setSession((s) => {
                    if (!s) return s;
                    const pinned = new Set(s.pinned);
                    if (!pinned.delete(action)) pinned.add(action);
                    return { ...s, pinned };
                }),
            toggleGto: () => setSession((s) => s && { ...s, gto: !s.gto }),
        }),
        [],
    );
    return { session, showGto: !!session && (session.gto || altHeld), ...actions };
}
