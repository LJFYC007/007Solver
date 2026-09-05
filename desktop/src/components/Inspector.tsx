import type { CSSProperties } from "react";
import {
    type AggregatedAction,
    type HandGroup,
    type HandStrategy,
    type SolverNode,
    formatEv,
    formatNumber,
    formatPercent,
    handClassCombos,
    orderedActions,
    playerLabel,
    strategyGradient,
} from "../solver";
import { Board, Card } from "./PlayingCards";

function ActionBar({ actions }: { actions: AggregatedAction[] }) {
    const visible = actions.filter((action) => action.probability > 0);
    if (visible.length === 0) return <div className="action-bar empty" />;

    return (
        <div className="action-bar" aria-label="Action mix">
            {visible.map((action) => (
                <span
                    key={action.index}
                    style={{ backgroundColor: action.color, width: `${action.probability * 100}%` }}
                    title={`${action.label}: ${formatPercent(action.probability)}`}
                />
            ))}
        </div>
    );
}

function SpotOverview({ node }: { node: SolverNode }) {
    const activity =
        node.kind === "decision"
            ? `${playerLabel(node.actor)} to act`
            : node.kind === "chance"
              ? "Next card"
              : "Line complete";

    return (
        <section className="overview-panel">
            <div className="panel-strip-title">
                <strong>Overview</strong>
            </div>
            <div className="overview-row">
                <div className="stack-chip hero">
                    <span>Hero</span>
                    <strong>{formatNumber(node.state.stacks.hero)}</strong>
                </div>
                <div className="stack-chip villain">
                    <span>Villain</span>
                    <strong>{formatNumber(node.state.stacks.villain)}</strong>
                </div>
                <div className="overview-copy">
                    <strong>Pot {formatNumber(node.state.pot)}</strong>
                    <span>
                        {activity} · {node.state.street}
                    </span>
                </div>
                <div className="overview-board">
                    <Board board={node.state.board} />
                </div>
            </div>
        </section>
    );
}

function ActionSummary({ actions }: { actions: AggregatedAction[] }) {
    const displayActions = orderedActions(actions);

    return (
        <section className="action-summary-panel">
            <div className="panel-strip-title">
                <strong title="Weighted by input range and this player's actions; excludes opponent actions, chance, and hands without an available strategy.">
                    Actions · own reach weighted
                </strong>
            </div>
            {displayActions.length > 0 ? (
                <>
                    <div className="action-cards">
                        {displayActions.map((action) => (
                            <article key={action.index} style={{ backgroundColor: action.color }}>
                                <div className="action-card-title">
                                    <h3>{action.label}</h3>
                                    {action.chipsCommitted > 0 && (
                                        <small>({formatNumber(action.chipsCommitted)} committed)</small>
                                    )}
                                </div>
                                <div>
                                    <strong>{formatPercent(action.probability)}</strong>
                                </div>
                            </article>
                        ))}
                    </div>
                    <ActionBar actions={displayActions} />
                </>
            ) : (
                <p className="muted-copy">No strategy is available for this node.</p>
            )}
        </section>
    );
}

function comboKey(cards: readonly string[]): string {
    return [...cards].sort().join("-");
}

function comboFrequency(probability: number): string {
    const percentage = Math.max(0, probability) * 100;
    if (percentage > 0 && percentage < 0.1) return "<0.1";
    return String(Number(percentage.toFixed(1)));
}

function HandInspector({ board, group }: { board: string[]; group?: HandGroup }) {
    const handByCards = new Map<string, HandStrategy>((group?.hands ?? []).map((hand) => [comboKey(hand.cards), hand]));
    const combos = group
        ? handClassCombos(group.label).map((cards) => ({ cards, hand: handByCards.get(comboKey(cards)) }))
        : [];
    const columns = group?.label.endsWith("s") ? "two-columns" : "three-columns";

    return (
        <section className="hand-panel">
            <div className="panel-strip-title">
                <strong>Hands</strong>
            </div>

            {!group && <p className="muted-copy">Choose a populated cell in the range matrix.</p>}
            {group && (
                <div className={`combo-grid ${columns}`}>
                    {combos.map(({ cards, hand }) => {
                        const handActions = hand
                            ? group.actions.map((action) => ({
                                  ...action,
                                  probability: hand.strategy[action.index] ?? 0,
                              }))
                            : [];
                        const background = strategyGradient(handActions) ?? "none";
                        const ownReachWeight = Math.min(1, Math.max(0, hand?.ownReachWeight ?? 0));
                        const showsStrategy =
                            ownReachWeight > 0 && handActions.some((action) => action.probability > 0);
                        const reachDescription = cards.some((card) => board.includes(card))
                            ? "Blocked by board"
                            : !hand
                              ? "Not in input range"
                              : ownReachWeight === 0
                                ? "Zero own reach"
                                : hand.marginalReachMass === 0
                                  ? "Zero joint reach"
                                  : undefined;

                        return (
                            <article
                                className={`combo-tile${showsStrategy ? "" : " inactive"}`}
                                key={cards.join("-")}
                                style={
                                    {
                                        "--combo-background": background,
                                        "--combo-height": `${ownReachWeight * 100}%`,
                                    } as CSSProperties
                                }
                                title={
                                    hand
                                        ? `${reachDescription ? `${reachDescription} · ` : ""}Own reach weight ${formatPercent(ownReachWeight)} · Node strategy EV ${formatEv(hand.nodeStrategyEv)}`
                                        : reachDescription
                                }
                            >
                                <div className="combo-tile-heading">
                                    <div className="combo-cards" aria-label={cards.join(" ")}>
                                        {cards.map((card) => (
                                            <Card card={card} compact key={card} />
                                        ))}
                                    </div>
                                    {showsStrategy && <span>%</span>}
                                </div>
                                {reachDescription && <small className="combo-reach">{reachDescription}</small>}
                                {showsStrategy && (
                                    <div className="combo-action-list">
                                        {orderedActions(handActions).map((action) => (
                                            <span key={action.index}>
                                                <b>{action.label}</b>
                                                <strong>{comboFrequency(action.probability)}</strong>
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </article>
                        );
                    })}
                </div>
            )}
        </section>
    );
}

export default function Inspector({
    actions,
    node,
    selectedGroup,
}: {
    actions: AggregatedAction[];
    node: SolverNode;
    selectedGroup?: HandGroup;
}) {
    return (
        <aside className="inspector">
            <SpotOverview node={node} />
            <ActionSummary actions={actions} />
            {node.kind === "decision" && <HandInspector board={node.state.board} group={selectedGroup} />}
        </aside>
    );
}
