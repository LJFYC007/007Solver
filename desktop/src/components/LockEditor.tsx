import { type CSSProperties, useEffect, useRef, useState } from "react";
import { type ActionInfo, type DecisionNode, actionFrequency, aggregateActions, cardsKey } from "../solver";
import { type HandFilters, type HandRow, NO_FILTERS } from "../solver/display";
import {
    type LockedHands,
    type NodeLock,
    frequencyLimits,
    hasEarlierDecisions,
    lockFrequency,
    scopeHands,
    selectedHands,
    strategies,
} from "../solver/locks";
import FiltersPanel, { FilterChips } from "./FiltersPanel";
import LockGlyph from "./LockGlyph";
import PinGlyph from "./PinGlyph";

/** A percentage field that applies its value on Enter or when it loses focus. */
function PercentInput({
    value,
    label,
    disabled,
    onCommit,
}: {
    value: number;
    label: string;
    disabled: boolean;
    onCommit: (value: number) => void;
}) {
    const [text, setText] = useState<string>();
    function commit() {
        if (text === undefined) return;
        setText(undefined);
        const next = Number(text);
        if (text.trim() && Number.isFinite(next)) onCommit(next);
    }
    return (
        <span className="solve-input-unit lock-percent">
            <input
                type="number"
                min={0}
                max={100}
                step="any"
                aria-label={label}
                disabled={disabled}
                value={text ?? value.toFixed(1)}
                onChange={(event) => setText(event.target.value)}
                onBlur={commit}
                onKeyDown={(event) => {
                    if (event.key === "Enter") commit();
                    else if (event.key === "Escape") {
                        // It drops the typed value only, not the editor's selection.
                        event.stopPropagation();
                        setText(undefined);
                    }
                }}
            />
            <span>%</span>
        </span>
    );
}

const ActionName = ({ action }: { action: ActionInfo }) => (
    <span className="lock-action">
        <i style={{ background: action.color }} />
        {action.label}
        {action.size && <small>{action.size}</small>}
    </span>
);

/** The strategy toolbar while a lock is edited. */
export function LockToolbar({
    filters,
    showGto,
    gtoAvailable,
    onFilters,
    onToggleGto,
}: {
    filters: HandFilters;
    showGto: boolean;
    /** Whether the GTO strategies have loaded. */
    gtoAvailable: boolean;
    onFilters: (filters: HandFilters) => void;
    onToggleGto: () => void;
}) {
    return (
        <div className="lock-toolbar" role="toolbar" aria-label="Lock editing">
            <strong className="lock-toolbar-title">
                <LockGlyph />
                Node lock
            </strong>
            <FilterChips filters={filters} onChange={onFilters} />
            <span className="lock-keys">
                <kbd>Shift</kbd>+click to add/remove · Hold <kbd>Alt</kbd> for GTO
            </span>
            <button
                type="button"
                className="lock-show-gto"
                aria-pressed={showGto}
                disabled={!gtoAvailable}
                onClick={onToggleGto}
            >
                Show GTO
            </button>
        </div>
    );
}

/**
 * PioSolver's node lock editor in the inspector: the frequencies of the whole range or the selected hands, the hand
 * classes that select, then the lock's options.
 */
export default function LockEditor({
    node,
    gto,
    gtoError,
    rows,
    lock,
    selection,
    filters,
    equityStatus,
    pinnedActions,
    onTogglePin,
    applyDisabled,
    onFilters,
    onChange,
    onExit,
    onApply,
}: {
    node: DecisionNode;
    /** The decision in the GTO solution: the reference's, or the current one without a reference; undefined while it loads. */
    gto?: DecisionNode;
    /** Why the GTO decision could not be loaded, if it could not. */
    gtoError?: string;
    /** The actor's rows, with the lock's strategies. */
    rows: HandRow[];
    lock: NodeLock;
    /** The cards keys of the selected hands; without them, the whole range is selected. */
    selection?: ReadonlySet<string>;
    filters: HandFilters;
    equityStatus?: string;
    pinnedActions: ReadonlySet<number>;
    onTogglePin: (action: number) => void;
    applyDisabled: boolean;
    /** Sets the filters, which replace the clicked classes as the selection: no filters select the whole range. */
    onFilters: (filters: HandFilters) => void;
    onChange: (change: Partial<NodeLock>) => void;
    /** Leaves the editor; the lock stays as it is. */
    onExit: () => void;
    onApply: () => void;
}) {
    const section = useRef<HTMLElement>(null);
    useEffect(() => section.current?.querySelector<HTMLElement>(":enabled")?.focus(), []);
    const { hands, pinned: pinnedHands } = lock;
    const selecting = selection !== undefined;
    // The whole range's frequencies count the pinned hands, though whole-range moves hold them.
    const inScope = scopeHands(node, hands, selection);
    const inScopeKeys = inScope.map((hand) => cardsKey(hand.cards));
    const mix = aggregateActions(node, inScope);
    const limits = frequencyLimits(node, lock, selection, pinnedActions);
    const gtoMix = gto && aggregateActions(gto, selectedHands(gto, selection));
    // A slider move recomputes from the hands its drag started from, so dragging back restores the hands it moved. A
    // drag keeps to the selection it started with, which Escape can clear while the slider keeps focus.
    const drag = useRef<{ from: LockedHands; selecting: boolean }>(undefined);
    const earlier = hasEarlierDecisions(lock);
    /** Moves an action's frequency over the selection, or the whole range, to `target`, from `base` if given. */
    function setFrequency(action: number, target: number, base?: LockedHands) {
        // Hands set through a selection are pinned: whole-range moves hold them.
        const pins = selecting ? inScopeKeys.filter((key) => !pinnedHands.has(key)) : [];
        const from = base ? { ...lock, hands: base } : lock;
        onChange({
            hands: lockFrequency(node, from, selection, action, target, pinnedActions),
            pinned: pins.length ? new Set([...pinnedHands, ...pins]) : pinnedHands,
        });
    }
    /**
     * Returns the selection's hands, or every hand, pinned ones included, to their GTO strategies and unpins them. Hands
     * without one, which reach the decision only in the locked solution, keep their strategies and pins.
     */
    function revert() {
        // Revert to GTO is disabled until the GTO decision has loaded.
        const gtoStrategies = strategies(gto!);
        const keys = (selecting ? inScopeKeys : Object.keys(hands)).filter((key) => gtoStrategies.has(key));
        const pinned = new Set(pinnedHands);
        for (const key of keys) pinned.delete(key);
        const reverted = Object.fromEntries(keys.map((key) => [key, gtoStrategies.get(key)!]));
        onChange({ hands: { ...hands, ...reverted }, pinned });
    }
    return (
        <section ref={section} className="lock-editor" aria-label="Node lock editor">
            <div className="lock-body">
                <section className="lock-group">
                    <div className="lock-selection">
                        <div className="filters-mode lock-scope" role="group" aria-label="Hands the sliders set">
                            <button type="button" aria-pressed={!selecting} onClick={() => onFilters(NO_FILTERS)}>
                                Whole range
                            </button>
                            <button type="button" aria-pressed={selecting} disabled={!selecting}>
                                Selected hands
                            </button>
                        </div>
                        <button
                            type="button"
                            className="quiet-button lock-revert"
                            disabled={(selecting && !inScope.length) || !gto}
                            onClick={revert}
                        >
                            Revert to GTO
                        </button>
                    </div>
                    <div className="lock-sliders">
                        {mix.map((action) => {
                            const pinned = pinnedActions.has(action.index);
                            const [min, max] = limits[action.index];
                            // Only another unpinned action, in hands that a move changes, can give or take frequency.
                            const fixed = pinned || min === max;
                            // At the slider's step, so the release's change event finds no new value to apply.
                            const value = Number((action.probability * 100).toFixed(1));
                            const gtoValue = actionFrequency(gtoMix, action.index);
                            // Targets beyond the limits move as far as the limits allow.
                            const setPercent = (target: number, base?: LockedHands) =>
                                setFrequency(action.index, target / 100, base);
                            return (
                                <div
                                    key={action.index}
                                    className="lock-slider"
                                    style={
                                        {
                                            "--action": action.color,
                                            "--fill": value / 100,
                                            // Below the minimum, the share that pinned hands hold.
                                            "--min": fixed ? 0 : min,
                                            "--max": fixed ? value / 100 : max,
                                            "--gto": gtoValue === undefined ? -1 : gtoValue,
                                        } as CSSProperties
                                    }
                                >
                                    <ActionName action={action} />
                                    <input
                                        type="range"
                                        min={0}
                                        max={100}
                                        step={0.1}
                                        value={value}
                                        disabled={fixed}
                                        aria-label={`${action.label} frequency`}
                                        aria-valuemin={fixed ? value : Number((min * 100).toFixed(1))}
                                        aria-valuemax={fixed ? value : Number((max * 100).toFixed(1))}
                                        onPointerDown={() => {
                                            // A value being typed applies first, so the drag builds on it.
                                            if (document.activeElement instanceof HTMLElement)
                                                document.activeElement.blur();
                                            drag.current = undefined;
                                        }}
                                        onChange={(event) => {
                                            if (drag.current?.selecting !== selecting)
                                                drag.current = { from: hands, selecting };
                                            setPercent(event.target.valueAsNumber, drag.current.from);
                                        }}
                                        onPointerUp={() => (drag.current = undefined)}
                                        onBlur={() => (drag.current = undefined)}
                                    />
                                    <PercentInput
                                        value={value}
                                        label={`${action.label} frequency (%)`}
                                        disabled={fixed}
                                        onCommit={setPercent}
                                    />
                                    <button
                                        type="button"
                                        className="lock-pin"
                                        aria-pressed={pinned}
                                        aria-label={`Pin ${action.label} frequency`}
                                        onClick={() => onTogglePin(action.index)}
                                    >
                                        <PinGlyph />
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                </section>
                <section className="lock-group lock-classes">
                    <h3>Hand classes</h3>
                    <FiltersPanel
                        node={node}
                        rows={rows}
                        filters={filters}
                        equityStatus={equityStatus}
                        onChange={onFilters}
                    />
                </section>
            </div>
            <footer className="lock-footer">
                {gtoError && (
                    <p className="solve-error" role="alert">
                        {gtoError}
                    </p>
                )}
                <div className="lock-options">
                    <label>
                        <input
                            type="checkbox"
                            checked={lock.keepEarlier}
                            disabled={!earlier}
                            onChange={(event) => onChange({ keepEarlier: event.target.checked })}
                        />
                        Lock previous nodes
                    </label>
                    <button type="button" className="quiet-button lock-exit" onClick={onExit}>
                        Exit
                    </button>
                    <button type="button" className="solve-button" disabled={applyDisabled} onClick={onApply}>
                        Apply & solve
                    </button>
                </div>
            </footer>
        </section>
    );
}
