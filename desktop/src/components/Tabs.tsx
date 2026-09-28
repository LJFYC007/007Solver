import type { ReactNode } from "react";

/** A tab list and its panel. Arrow keys move between the enabled tabs; only the selected one is in the tab order. */
export default function Tabs<T extends string>({
    tabs,
    selected,
    onSelect,
    label,
    id,
    className,
    panelClassName,
    disabled = {},
    detail,
    children,
}: {
    tabs: readonly T[];
    selected: T;
    onSelect: (tab: T) => void;
    label: string;
    /** The panel's ID, which also prefixes the tabs' IDs. */
    id: string;
    className: string;
    panelClassName: string;
    /** Why a tab is unavailable, for each tab that is. */
    disabled?: Partial<Record<T, string>>;
    /** Shown after the tabs. */
    detail?: ReactNode;
    children: ReactNode;
}) {
    const tabId = (name: T) => `${id}-${name.replace(" ", "-")}`;
    const enabled = tabs.filter((name) => !disabled[name]);
    return (
        <>
            <div className={className} role="tablist" aria-label={label}>
                {tabs.map((name) => (
                    <button
                        id={tabId(name)}
                        aria-controls={id}
                        type="button"
                        key={name}
                        role="tab"
                        aria-selected={selected === name}
                        tabIndex={selected === name ? 0 : -1}
                        disabled={!!disabled[name]}
                        title={disabled[name]}
                        onClick={() => onSelect(name)}
                        onKeyDown={(event) => {
                            if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
                            event.preventDefault();
                            const step = event.key === "ArrowLeft" ? -1 : 1;
                            const next = enabled[(enabled.indexOf(name) + step + enabled.length) % enabled.length];
                            onSelect(next);
                            document.getElementById(tabId(next))?.focus();
                        }}
                    >
                        {name}
                    </button>
                ))}
                {detail}
            </div>
            <div className={panelClassName} id={id} role="tabpanel" aria-labelledby={tabId(selected)}>
                {children}
            </div>
        </>
    );
}
