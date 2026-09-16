import { STREETS, type BettingTreeDraft } from "../solver/bettingTree";

export default function BettingTreeSettings({
    value,
    disabled,
    onChange,
}: {
    value: BettingTreeDraft;
    disabled: boolean;
    onChange: (value: BettingTreeDraft) => void;
}) {
    return (
        <details className="betting-tree-settings">
            <summary>Betting tree · {value.maxRaises} raises max per street</summary>
            <fieldset disabled={disabled}>
                <div className="betting-size-grid">
                    <span>Street</span>
                    <span>Bet (% pot)</span>
                    <span>Raise (%)</span>
                    {STREETS.map((street) => (
                        <div className="betting-size-row" key={street}>
                            <strong>{street}</strong>
                            {(["bet", "raise"] as const).map((kind) => (
                                <input
                                    key={kind}
                                    aria-label={`${street} ${kind} percentages`}
                                    placeholder="None"
                                    value={value[street][kind]}
                                    onChange={(event) =>
                                        onChange({
                                            ...value,
                                            [street]: { ...value[street], [kind]: event.target.value },
                                        })
                                    }
                                />
                            ))}
                        </div>
                    ))}
                </div>
                <small>Comma-separated sizes; empty means none. Raise % uses the pot after calling.</small>
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
                <small>Opening bet excluded. At the limit, only call / fold remain.</small>
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
                <small>Replace bet / raise when SPR after a call is at or below this value. 0 disables merging.</small>
            </fieldset>
        </details>
    );
}
