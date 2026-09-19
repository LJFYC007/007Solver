import type { TableFormat } from "../solver/catalog";

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
    return (
        <section className="preflop-settings" aria-label="Solution settings">
            <div className="solution-heading">
                <strong>
                    100bb <span>Chip EV</span>
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
            <div className="solution-options" role="group" aria-label="Table size">
                {(["6max", "8max"] as const).map((value) => (
                    <button
                        type="button"
                        key={value}
                        aria-pressed={format === value}
                        disabled={disabled}
                        onClick={() => {
                            if (value !== format) onChange(value);
                        }}
                    >
                        {value === "6max" ? "6-max" : "8-max"}
                    </button>
                ))}
            </div>
            <small title="GTO Wizard ranges · Cold calls enabled · No rake or ante">2.5bb open · No rake</small>
        </section>
    );
}
