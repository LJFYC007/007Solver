import type { CSSProperties } from "react";
import { strategyGradient } from "../solver";
import {
    type DisplayMode,
    type EvComparison,
    RANGE_BACKGROUND,
    evShare,
    frequency,
    handColor,
    modeLabel,
    percent,
} from "../solver/display";
import type { DetailAction, DetailCombo } from "../solver/study";
import { Card } from "./PlayingCards";

// Equity, EQR and Range show the value large, so the heading only names it.
const showsLargeValue = (mode: DisplayMode): mode is "equity" | "eqr" | "range" =>
    mode === "equity" || mode === "eqr" || mode === "range";

function comboValue(combo: DetailCombo, mode: DisplayMode) {
    if (mode === "strategyEv" || mode === "ev")
        return combo.ev === undefined ? undefined : combo.ev === null ? "EV —" : `EV ${combo.ev.toFixed(2)}`;
    if (mode === "strategyEq") return combo.equity === undefined ? undefined : `EQ ${percent(combo.equity)}`;
    if (mode === "compareEv") return "EV/POT";
    if (showsLargeValue(mode)) return modeLabel(mode);
    return `${(combo.weight * 100).toFixed(combo.weight < 1 ? 1 : 0)}%`;
}

/** What the action rows show: action EVs in the EV modes, their pot shares in Compare EV, else frequencies. */
function actionValue(mode: DisplayMode, pot: number) {
    if (mode === "strategyEv" || mode === "ev")
        return (a: DetailAction) => (a.ev === undefined ? "—" : a.ev.toFixed(2));
    if (mode === "compareEv") return (a: DetailAction) => percent(evShare(a.ev, pot));
    return (a: DetailAction) => `${frequency(a.probability)}%`;
}

const largeValue = (combo: DetailCombo, mode: "equity" | "eqr" | "range") =>
    mode === "equity" ? percent(combo.equity) : mode === "eqr" ? percent(combo.eqr) : combo.weight.toFixed(2);

function ActionList({ actions, value }: { actions: DetailAction[]; value: (action: DetailAction) => string }) {
    return (
        <ul className="combo-action-list">
            {actions.map((a) => (
                <li
                    key={a.id}
                    className={a.probability > 0 ? undefined : "unused"}
                    title={`${a.label}${a.size ? ` (${a.size} pot)` : ""}: ${(a.probability * 100).toFixed(2)}%${a.ev === undefined ? "" : `, EV ${a.ev.toFixed(2)}`}`}
                >
                    <i style={{ background: a.color }} />
                    <b>{a.label}</b>
                    <strong>{value(a)}</strong>
                </li>
            ))}
        </ul>
    );
}

export default function HandDetails({
    label,
    combos,
    mode,
    pot,
    comparison,
}: {
    label: string;
    combos: DetailCombo[];
    mode: DisplayMode;
    /** Postflop only: the pot that scales metric colors, EV differences and EV/POT. */
    pot?: number;
    /** Compare EV's choice. */
    comparison?: EvComparison;
}) {
    const columns = label.endsWith("s") ? 2 : 3;
    const color = pot === undefined ? undefined : handColor(mode, pot, comparison);
    const value = actionValue(mode, pot ?? 0);
    return (
        <div
            className={`combo-grid ${columns === 2 ? "two-columns" : "three-columns"}`}
            role="group"
            aria-label={`${label} hand combinations`}
        >
            {combos.map((combo) => {
                const active = combo.weight > 0 && combo.matches;
                const fill = color?.(combo.ev, combo.equity, combo.actionEvs);
                const background =
                    mode === "range"
                        ? RANGE_BACKGROUND
                        : color
                          ? fill
                              ? `linear-gradient(${fill}, ${fill})`
                              : "none"
                          : (strategyGradient(combo.actions) ?? "none");
                const heading = active ? comboValue(combo, mode) : undefined;
                const description = combo.matches ? combo.description : "Filtered out";
                return (
                    <article
                        key={combo.cards.join("")}
                        className={`combo-tile${active ? "" : " inactive"}`}
                        style={
                            {
                                "--strategy-background": background,
                                "--strategy-height": `${Math.min(1, Math.max(0, combo.weight)) * 100}%`,
                            } as CSSProperties
                        }
                        title={description}
                    >
                        <div className="combo-tile-heading">
                            <div className="combo-cards" aria-label={combo.cards.join(" ")}>
                                {combo.cards.map((card) => (
                                    <Card key={card} card={card} />
                                ))}
                            </div>
                            {heading && <span>{heading}</span>}
                        </div>
                        {!active || !combo.actions.length ? (
                            <small className="combo-reach">{description}</small>
                        ) : showsLargeValue(mode) ? (
                            <strong className="combo-metric">{largeValue(combo, mode)}</strong>
                        ) : mode === "all" ? (
                            <>
                                <ActionList actions={combo.actions} value={value} />
                                <dl className="combo-stats">
                                    <dt>Range</dt>
                                    <dd>{combo.weight.toFixed(2)}</dd>
                                    {combo.evPot !== undefined && (
                                        <>
                                            <dt>EV/POT</dt>
                                            <dd>{percent(combo.evPot)}</dd>
                                        </>
                                    )}
                                    {combo.equity !== undefined && (
                                        <>
                                            <dt>EQ</dt>
                                            <dd>{percent(combo.equity)}</dd>
                                            <dt>EQR</dt>
                                            <dd>{percent(combo.eqr)}</dd>
                                        </>
                                    )}
                                </dl>
                            </>
                        ) : (
                            <ActionList actions={combo.actions} value={value} />
                        )}
                    </article>
                );
            })}
        </div>
    );
}

export function ActionSummary({
    actions,
    actor,
}: {
    actions: (DetailAction & { combos: number; onSelect?: () => void })[];
    actor?: string;
}) {
    return (
        <section className="action-summary-panel">
            <div className="panel-strip-title">
                <strong>Actions</strong>
                <span>{actor}</span>
            </div>
            {actions.length ? (
                <>
                    <div className="action-cards">
                        {actions.map((a) => (
                            <button
                                type="button"
                                key={a.id}
                                data-kind={a.kind}
                                style={{ "--action": a.color } as CSSProperties}
                                onClick={a.onSelect}
                                disabled={!a.onSelect}
                                title={
                                    a.locked
                                        ? "This branch has not been downloaded"
                                        : a.size
                                          ? `${a.label} · ${a.size} pot`
                                          : a.label
                                }
                            >
                                <span className="action-card-name">
                                    <h3>
                                        {a.locked && "🔒 "}
                                        {a.label}
                                    </h3>
                                    {a.size && <small>{a.size} pot</small>}
                                </span>
                                <span className="action-card-stats">
                                    <strong>{frequency(a.probability)}%</strong>
                                    <small>
                                        {a.combos > 0 && a.combos < 0.01 ? "<0.01" : a.combos.toFixed(2)} combos
                                    </small>
                                </span>
                            </button>
                        ))}
                    </div>
                    <div className="action-bar" aria-label="Action mix">
                        {actions.map((a) => (
                            <span
                                key={a.id}
                                style={{ width: `${a.probability * 100}%`, backgroundColor: a.color }}
                                title={`${a.label} ${(a.probability * 100).toFixed(2)}%`}
                            />
                        ))}
                    </div>
                </>
            ) : (
                <p className="muted-copy">Choose an action in the history to inspect its strategy.</p>
            )}
        </section>
    );
}
