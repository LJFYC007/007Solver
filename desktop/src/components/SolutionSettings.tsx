import { useRef, useState } from "react";
import { useOutsidePress } from "../hooks/useOutsidePress";
import type { TableFormat } from "../solver/catalog";

const FORMATS: { value: TableFormat; label: string }[] = [
    { value: "6max", label: "6-max" },
    { value: "8max", label: "8-max" },
];

export default function SolutionSettings({
    format,
    disabled,
    onChange,
    onReset,
}: {
    format: TableFormat;
    disabled: boolean;
    onChange: (format: TableFormat) => void;
    onReset: () => void;
}) {
    const [open, setOpen] = useState(false);
    const card = useRef<HTMLElement>(null);
    useOutsidePress(card, open, setOpen);
    return (
        <section
            ref={card}
            className="preflop-settings"
            aria-label="Solution settings"
            onKeyDown={(event) => {
                if (event.key === "Escape") setOpen(false);
            }}
        >
            <div className="solution-heading">
                <strong>
                    Cash <span>100bb</span>
                </strong>
                <button
                    type="button"
                    disabled={disabled}
                    aria-label="Reset history"
                    title="Reset history"
                    onClick={onReset}
                >
                    ↻
                </button>
            </div>
            <ul title="GTO Wizard ranges · Cold calls enabled · No rake or ante">
                <li>{FORMATS.find((entry) => entry.value === format)?.label} · Chip EV</li>
                <li>2.5bb open · No rake</li>
            </ul>
            <button
                type="button"
                className="quiet-button"
                aria-expanded={open}
                aria-controls="solution-menu"
                disabled={disabled}
                onClick={() => setOpen(!open)}
            >
                Change
            </button>
            {open && (
                <div id="solution-menu" className="solution-menu">
                    <span>Players</span>
                    <div className="pill-group" role="group" aria-label="Table size">
                        {FORMATS.map(({ value, label }) => (
                            <button
                                type="button"
                                key={value}
                                aria-pressed={format === value}
                                onClick={() => {
                                    setOpen(false);
                                    if (value !== format) onChange(value);
                                }}
                            >
                                {label}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </section>
    );
}
