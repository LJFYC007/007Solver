import { useEffect, useRef, useState } from "react";
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
    const [open, setOpen] = useState(false);
    const dialog = useRef<HTMLDialogElement>(null);
    useEffect(() => {
        if (!open) return;
        const element = dialog.current!;
        element.showModal();
        return () => element.close();
    }, [open]);
    return (
        <>
            <section className="preflop-settings" aria-label="Solution settings">
                <strong>
                    Cash <span>100bb</span>
                </strong>
                <span>• {format} cEV</span>
                <span>• With cold calls 2.5x</span>
                <div className="solution-settings-actions">
                    <button type="button" disabled={disabled} onClick={() => setOpen(true)}>
                        ⚙ Change
                    </button>
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
            </section>
            {open && (
                <dialog
                    ref={dialog}
                    className="solution-dialog"
                    aria-label="Choose solution"
                    onCancel={() => setOpen(false)}
                    onClick={(event) => {
                        if (event.target === event.currentTarget) setOpen(false);
                    }}
                >
                    <div className="solution-dialog-content">
                        <header>
                            <strong>Solutions</strong>
                            <button type="button" aria-label="Close solution settings" onClick={() => setOpen(false)}>
                                ×
                            </button>
                        </header>
                        <p>Cash · 100bb · Chip EV</p>
                        <span>Players</span>
                        <div className="solution-options" role="group" aria-label="Players">
                            {(["6max", "8max"] as const).map((value) => (
                                <button
                                    type="button"
                                    key={value}
                                    aria-pressed={format === value}
                                    disabled={disabled}
                                    onClick={() => {
                                        setOpen(false);
                                        if (value !== format) onChange(value);
                                    }}
                                >
                                    {value}
                                </button>
                            ))}
                        </div>
                        <small>With cold calls · 2.5x opens · No rake or ante</small>
                    </div>
                </dialog>
            )}
        </>
    );
}
