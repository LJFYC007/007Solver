import { Fragment, useState, type CSSProperties } from "react";
import { sizePosition, spreadSizeColors } from "../solver";
import { POSITIONS, STREETS, parsePercentage, type BettingTreeDraft, type Position } from "../solver/bettingTree";

// One field per size, marked in the color its bet or raise will have in the solution.
function SizeList({ label, texts, onChange }: { label: string; texts: string[]; onChange: (texts: string[]) => void }) {
    const [added, setAdded] = useState<number>();
    const parsed = texts.map(parsePercentage);
    const sizes = [...new Set(parsed.filter((size) => size !== undefined))].sort((a, b) => a - b);
    const colors = spreadSizeColors(sizes.map(sizePosition));
    return (
        <div className="size-list">
            {texts.map((text, index) => {
                const size = parsed[index];
                return (
                    <span
                        key={index}
                        className="size-field"
                        style={
                            {
                                "--action": size === undefined ? "transparent" : colors[sizes.indexOf(size)],
                            } as CSSProperties
                        }
                    >
                        <input
                            aria-label={`${label} size ${index + 1}`}
                            aria-invalid={text.trim() !== "" && size === undefined}
                            inputMode="decimal"
                            autoFocus={index === added}
                            value={text}
                            onChange={(event) => onChange(texts.map((t, i) => (i === index ? event.target.value : t)))}
                        />
                        <button
                            type="button"
                            aria-label={`Remove ${label} size ${index + 1}`}
                            onClick={() => onChange(texts.filter((_, i) => i !== index))}
                        >
                            ×
                        </button>
                    </span>
                );
            })}
            <button
                type="button"
                className="size-add"
                aria-label={`Add ${label} size`}
                title="Add size"
                onClick={() => {
                    setAdded(texts.length);
                    onChange([...texts, ""]);
                }}
            >
                +
            </button>
        </div>
    );
}

export default function BettingTreeSettings({
    value,
    players,
    disabled,
    error,
    onChange,
}: {
    value: BettingTreeDraft;
    players?: Record<Position, string>;
    disabled: boolean;
    error?: string;
    onChange: (value: BettingTreeDraft) => void;
}) {
    return (
        <fieldset className="betting-tree-settings" disabled={disabled}>
            <div className="betting-players">
                {POSITIONS.map((position) => {
                    const player = value[position];
                    return (
                        <section key={position} className="betting-player" aria-label={`${position} bet sizes`}>
                            <h4>
                                {players?.[position]} <small>{position}</small>
                            </h4>
                            <div className="betting-size-grid">
                                <span />
                                <span>Bet · % pot</span>
                                <span>Raise · % pot after call</span>
                                {STREETS.map((street) => (
                                    <Fragment key={street}>
                                        <strong>{street}</strong>
                                        {(["bet", "raise"] as const).map((kind) => (
                                            <SizeList
                                                key={kind}
                                                label={`${position.toUpperCase()} ${street} ${kind}`}
                                                texts={player[street][kind]}
                                                onChange={(texts) =>
                                                    onChange({
                                                        ...value,
                                                        [position]: {
                                                            ...player,
                                                            [street]: { ...player[street], [kind]: texts },
                                                        },
                                                    })
                                                }
                                            />
                                        ))}
                                    </Fragment>
                                ))}
                            </div>
                        </section>
                    );
                })}
            </div>
            <div className="solve-targets betting-limits">
                <label>
                    Raises per street
                    <select
                        aria-label="Maximum raises per street"
                        title="Excludes the opening bet"
                        value={value.maxRaises}
                        onChange={(event) => onChange({ ...value, maxRaises: Number(event.target.value) })}
                    >
                        <option value={0}>None</option>
                        <option value={1}>1 · Raise</option>
                        <option value={2}>2 · Re-raise</option>
                    </select>
                </label>
                <label>
                    All-in at SPR ≤
                    <input
                        aria-label="All-in SPR threshold"
                        title="Replace a bet or raise with all-in when the SPR after a call is at or below this value. 0 disables merging."
                        type="number"
                        min="0"
                        step="0.01"
                        value={Number.isNaN(value.allInSpr) ? "" : value.allInSpr}
                        onChange={(event) => onChange({ ...value, allInSpr: event.target.valueAsNumber })}
                    />
                </label>
            </div>
            {error && (
                <small className="betting-size-error" role="alert">
                    {error}
                </small>
            )}
        </fieldset>
    );
}
