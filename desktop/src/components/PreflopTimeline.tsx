import { formatNumber } from "../solver";
import type { PreflopChoice } from "../solver/catalog";
import type { PreflopEntry } from "../solver/study";

export default function PreflopTimeline({
    entries,
    history,
    activeIndex,
    disabled,
    onView,
    onAction,
}: {
    entries: PreflopEntry[];
    history: PreflopChoice[];
    activeIndex: number | null;
    disabled: boolean;
    onView: (index: number) => void;
    onAction: (entry: PreflopEntry, action: string) => void;
}) {
    return entries.map((entry, index) => (
        <section
            key={index}
            className={`preflop-node${activeIndex === index ? " active" : ""}${index > history.length ? " future" : ""}`}
        >
            <button type="button" className="preflop-node-title" disabled={disabled} onClick={() => onView(index)}>
                <strong>{entry.node.actor}</strong>
                <span>{formatNumber(entry.stack)}</span>
            </button>
            {entry.node.actions
                .slice()
                .reverse()
                .map((action) => (
                    <button
                        type="button"
                        key={action.code}
                        disabled={disabled}
                        className={history[index]?.action === action.label ? "chosen" : undefined}
                        onClick={() => onAction(entry, action.label)}
                    >
                        {action.label}
                    </button>
                ))}
        </section>
    ));
}
