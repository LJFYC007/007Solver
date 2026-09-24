import { formatNumber } from "../solver";
import type { PreflopChoice } from "../solver/catalog";
import type { PreflopEntry } from "../solver/study";
import TimelineChoice from "./TimelineChoice";

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
            className={`timeline-node${activeIndex === index ? " active" : ""}${index > history.length ? " future" : ""}`}
        >
            <button type="button" className="timeline-node-title" disabled={disabled} onClick={() => onView(index)}>
                <strong>{entry.node.actor}</strong>
                <span>{formatNumber(entry.stack)}</span>
            </button>
            {entry.node.actions
                .slice()
                .reverse()
                .map((action) => (
                    <TimelineChoice
                        key={action.code}
                        label={action.label}
                        color={action.color}
                        chosen={history[index]?.action === action.label}
                        disabled={disabled}
                        onClick={() => onAction(entry, action.label)}
                    />
                ))}
        </section>
    ));
}
