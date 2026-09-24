import { sizePosition, spreadSizeColors } from "../solver";
import { STREETS, parsePercentages, type BettingTreeDraft } from "../solver/bettingTree";

// Previews each size in the color its bet or raise will have in the solution.
export function SizePills({ sizes }: { sizes: number[] }) {
    if (!sizes.length) return <span className="size-pills empty">None</span>;
    const sorted = [...sizes].sort((a, b) => a - b);
    const colors = spreadSizeColors(sorted.map(sizePosition));
    return (
        <span className="size-pills">
            {sorted.map((size, index) => (
                <i key={index} style={{ background: colors[index] }}>
                    {size}%
                </i>
            ))}
        </span>
    );
}

function parsed(text: string) {
    try {
        return parsePercentages(text, "");
    } catch {
        return undefined;
    }
}

export default function BettingTreeSettings({
    value,
    disabled,
    error,
    onChange,
}: {
    value: BettingTreeDraft;
    disabled: boolean;
    error?: string;
    onChange: (value: BettingTreeDraft) => void;
}) {
    return (
        <section className="betting-tree-settings">
            <h3>Bet sizes</h3>
            <fieldset disabled={disabled}>
                <div className="betting-size-grid">
                    <span />
                    <span>Bet (% pot)</span>
                    <span>Raise (% pot after call)</span>
                    {STREETS.map((street) => (
                        <div className="betting-size-row" key={street}>
                            <strong>{street}</strong>
                            {(["bet", "raise"] as const).map((kind) => {
                                const sizes = parsed(value[street][kind]);
                                return (
                                    <label key={kind} className="betting-size-field">
                                        <input
                                            aria-label={`${street} ${kind} percentages`}
                                            aria-invalid={!sizes}
                                            placeholder="None"
                                            value={value[street][kind]}
                                            onChange={(event) =>
                                                onChange({
                                                    ...value,
                                                    [street]: { ...value[street], [kind]: event.target.value },
                                                })
                                            }
                                        />
                                        {sizes && <SizePills sizes={sizes} />}
                                    </label>
                                );
                            })}
                        </div>
                    ))}
                </div>
                {error ? (
                    <small className="betting-size-error" role="alert">
                        {error}
                    </small>
                ) : (
                    <small>Separate sizes with spaces or commas. Each size adds a branch to every matching node.</small>
                )}
                <details className="solve-advanced">
                    <summary>Advanced</summary>
                    <label>
                        Raises per street
                        <select
                            aria-label="Maximum raises per street"
                            value={value.maxRaises}
                            onChange={(event) => onChange({ ...value, maxRaises: Number(event.target.value) })}
                        >
                            <option value={0}>0 · No raises</option>
                            <option value={1}>1 · Raise</option>
                            <option value={2}>2 · Raise + re-raise</option>
                        </select>
                    </label>
                    <small>Excludes the opening bet.</small>
                    <label>
                        All-in SPR threshold
                        <input
                            aria-label="All-in SPR threshold"
                            type="number"
                            min="0"
                            step="0.01"
                            value={Number.isNaN(value.allInSpr) ? "" : value.allInSpr}
                            onChange={(event) => onChange({ ...value, allInSpr: event.target.valueAsNumber })}
                        />
                    </label>
                    <small>
                        Replace bet / raise when SPR after a call is at or below this value. 0 disables merging.
                    </small>
                </details>
            </fieldset>
        </section>
    );
}
