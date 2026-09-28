import { useRef, useState } from "react";
import { useOutsidePress } from "../hooks/useOutsidePress";
import { DISPLAY_MODES, type DisplayMode, modeLabel } from "../solver/display";

/**
 * GTO Wizard's display dropdown. As a view switch, the first click selects the view and later clicks open
 * the list; otherwise every click opens it.
 */
export default function ModeMenu({
    mode,
    modes,
    enabled = modes,
    disabledReason,
    onMode,
    view,
}: {
    mode: DisplayMode;
    /** The listed modes, and those of them that can be chosen. */
    modes: readonly DisplayMode[];
    enabled?: readonly DisplayMode[];
    disabledReason?: string;
    onMode: (mode: DisplayMode) => void;
    view?: { selected: boolean; onSelect: () => void };
}) {
    const [open, setOpen] = useState(false);
    const root = useRef<HTMLDivElement>(null);
    const trigger = useRef<HTMLButtonElement>(null);
    useOutsidePress(root, open, setOpen);
    function close() {
        setOpen(false);
        trigger.current?.focus();
    }
    return (
        <div
            className="mode-menu"
            ref={root}
            onKeyDown={(event) => {
                if (event.key === "Escape" && open) close();
            }}
            onBlur={(event) => {
                if (!root.current?.contains(event.relatedTarget as Node | null)) setOpen(false);
            }}
        >
            <button
                type="button"
                ref={trigger}
                aria-expanded={open}
                className={`mode-menu-button${view?.selected ? " active" : ""}`}
                onClick={() => (view && !view.selected ? view.onSelect() : setOpen(!open))}
            >
                {modeLabel(mode)}
                <span aria-hidden="true">▾</span>
            </button>
            {open && (
                <div className="mode-menu-list" role="group" aria-label="Display">
                    {DISPLAY_MODES.filter((option) => modes.includes(option.id)).map((option) => (
                        <button
                            type="button"
                            key={option.id}
                            aria-pressed={option.id === mode}
                            disabled={!enabled.includes(option.id)}
                            title={enabled.includes(option.id) ? undefined : disabledReason}
                            onClick={() => {
                                onMode(option.id);
                                close();
                            }}
                        >
                            {option.label}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
