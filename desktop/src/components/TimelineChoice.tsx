import type { CSSProperties } from "react";

export default function TimelineChoice({
    label,
    size,
    color,
    chosen,
    disabled,
    onClick,
}: {
    label: string;
    size?: string;
    color: string;
    chosen: boolean;
    disabled: boolean;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            className={`timeline-choice${chosen ? " chosen" : ""}`}
            style={{ "--action": color } as CSSProperties}
            disabled={disabled}
            onClick={onClick}
            title={size ? `${label} · ${size} pot` : label}
        >
            <span>{label}</span>
            {size && <small>{size}</small>}
        </button>
    );
}
