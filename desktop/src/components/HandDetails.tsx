import type { CSSProperties } from "react";
import { strategyGradient } from "../solver";
import type { DetailAction, DetailCombo } from "../solver/study";
import { Card } from "./PlayingCards";

function frequency(probability: number) {
    return probability > 0 && probability < 0.001 ? "<0.1" : (probability * 100).toFixed(1);
}

function comboValue(combo: DetailCombo) {
    if (combo.ev === undefined) return `${(combo.weight * 100).toFixed(combo.weight < 1 ? 1 : 0)}%`;
    return combo.ev === null ? "EV —" : `EV ${combo.ev.toFixed(2)}`;
}

export default function HandDetails({ label, combos }: { label: string; combos: DetailCombo[] }) {
    const columns = label.endsWith("s") ? 2 : 3;
    return (
        <section className="hand-panel" aria-label={`${label} hand combinations`}>
            <div className="panel-strip-title">
                <strong>Hands</strong>
                <span>
                    {label} · {combos.length} combinations
                </span>
            </div>
            <div className={`combo-grid ${columns === 2 ? "two-columns" : "three-columns"}`}>
                {combos.map((combo) => (
                    <article
                        key={combo.cards.join("")}
                        className={`combo-tile${combo.weight > 0 ? "" : " inactive"}`}
                        style={
                            {
                                "--strategy-background": strategyGradient(combo.actions) ?? "none",
                                "--strategy-height": `${Math.min(1, Math.max(0, combo.weight)) * 100}%`,
                            } as CSSProperties
                        }
                        title={combo.description}
                    >
                        <div className="combo-tile-heading">
                            <div className="combo-cards" aria-label={combo.cards.join(" ")}>
                                {combo.cards.map((card) => (
                                    <Card key={card} card={card} />
                                ))}
                            </div>
                            {combo.weight > 0 && (
                                <span title={combo.ev === undefined ? "Range weight" : "Node strategy EV"}>
                                    {comboValue(combo)}
                                </span>
                            )}
                        </div>
                        {combo.actions.length ? (
                            <ul className="combo-action-list">
                                {combo.actions.map((a) => (
                                    <li
                                        key={a.id}
                                        className={a.probability > 0 ? undefined : "unused"}
                                        title={`${a.label}${a.size ? ` (${a.size} pot)` : ""}: ${(a.probability * 100).toFixed(2)}%`}
                                    >
                                        <i style={{ background: a.color }} />
                                        <b>{a.label}</b>
                                        <strong>{frequency(a.probability)}%</strong>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <small className="combo-reach">{combo.description}</small>
                        )}
                    </article>
                ))}
            </div>
        </section>
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
                                title={a.size ? `${a.label} · ${a.size} pot` : a.label}
                            >
                                <span className="action-card-name">
                                    <h3>{a.label}</h3>
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
