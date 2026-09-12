import type { CSSProperties } from "react";
import { strategyGradient } from "../solver";
import { Card } from "./PlayingCards";

function frequency(probability: number) {
    return probability > 0 && probability < 0.001 ? "<0.1" : (probability * 100).toFixed(1);
}

export interface DetailAction {
    id: string;
    label: string;
    color: string;
    probability: number;
}
export interface DetailCombo {
    cards: string[];
    weight: number;
    actions: DetailAction[];
    description?: string;
}
export default function HandDetails({ label, combos }: { label: string; combos: DetailCombo[] }) {
    const columns = label.endsWith("s") ? 2 : 3;
    const actionCount = Math.max(1, ...combos.map((c) => c.actions.length));
    return (
        <section className="hand-panel" aria-label={`${label} hand combinations`}>
            <div className="panel-strip-title">
                <strong>Hands</strong>
                <span>
                    {label} · {combos.length} combinations
                </span>
            </div>
            <div
                className={`combo-grid ${columns === 2 ? "two-columns" : "three-columns"}`}
                style={
                    {
                        "--action-count": actionCount,
                    } as CSSProperties
                }
            >
                {combos.map((combo) => {
                    return (
                        <article
                            key={combo.cards.join("")}
                            className={`combo-tile${combo.weight > 0 ? "" : " inactive"}`}
                            style={
                                {
                                    "--combo-background": strategyGradient(combo.actions) ?? "none",
                                    "--combo-height": `${Math.min(1, Math.max(0, combo.weight)) * 100}%`,
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
                                <span>%</span>
                            </div>
                            {combo.actions.length ? (
                                <div className="combo-action-list">
                                    {combo.actions.map((a) => (
                                        <span key={a.id} title={`${a.label}: ${(a.probability * 100).toFixed(2)}%`}>
                                            <b>{a.label}</b>
                                            <strong>{frequency(a.probability)}</strong>
                                        </span>
                                    ))}
                                </div>
                            ) : (
                                <small className="combo-reach">{combo.description ?? "No strategy"}</small>
                            )}
                        </article>
                    );
                })}
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
                                style={{ backgroundColor: a.color }}
                                onClick={a.onSelect}
                                disabled={!a.onSelect}
                            >
                                <h3 title={a.label}>{a.label}</h3>
                                <div>
                                    <strong>{frequency(a.probability)}%</strong>
                                    <small>
                                        {a.combos > 0 && a.combos < 0.01 ? "<0.01" : a.combos.toFixed(2)}
                                        <br />
                                        {" combos"}
                                    </small>
                                </div>
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
