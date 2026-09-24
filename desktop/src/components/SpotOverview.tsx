import { useState } from "react";
import { type EquityReport, type Player, formatNumber } from "../solver";
import { postflopOrder } from "../solver/preflop";
import type { SpotSeat } from "../solver/study";
import { useSolverEquity } from "../hooks/useSolverEquity";
import EquityChart from "./EquityChart";
import { BoardCard } from "./PlayingCards";

export interface SpotSummary {
    seats: SpotSeat[];
    pot: number;
    basePot?: number;
    board: string[];
    toCall?: number;
    players?: Record<Player, string>;
    nodeId?: number;
    showdown?: boolean;
    selectedHand?: string;
    onSeat?: (seat: string) => void;
    onBoard?: () => void;
}
const players: Player[] = ["villain", "hero"];
const tabs = ["Overview", "Table", "Equity chart"] as const;

function PlayerStats({ spot, data }: { spot: SpotSummary; data?: EquityReport }) {
    const seats = spot.players
        ? players.map((player) => ({
              player,
              seat: spot.seats.find((seat) => seat.position === spot.players![player]),
          }))
        : spot.seats
              .filter((seat) => !seat.folded)
              .slice(0, 2)
              .map((seat) => ({ seat, player: undefined }));
    return (
        <div className="player-stats">
            {seats.map(({ seat, player }) => {
                if (!seat) return null;
                const equity = player ? data?.players[player].equity : undefined;
                const ev = spot.showdown && equity != null ? equity * spot.pot : seat.ev;
                const eqr =
                    ev != null && equity != null && equity > 0 && spot.pot > 0 ? ev / (equity * spot.pot) : undefined;
                return (
                    <article key={seat.position}>
                        <header>
                            <strong>{seat.position}</strong>
                            <small>{player ? (player === "hero" ? "IP" : "OOP") : ""}</small>
                        </header>
                        <div className="player-stat-grid">
                            <div title="Expected chips won minus future contributions from this node">
                                <span>EV</span>
                                <b>{ev == null ? "—" : ev.toFixed(2)}</b>
                            </div>
                            <div title="Exact showdown equity over all legal runouts">
                                <span>Equity</span>
                                <b>{equity == null ? "—" : `${(equity * 100).toFixed(2)}%`}</b>
                            </div>
                            <div title="Current-node EV / (equity × current pot)">
                                <span>EQR</span>
                                <b>{eqr == null ? "—" : `${(eqr * 100).toFixed(0)}%`}</b>
                            </div>
                            <div title="Board-filtered own range weight">
                                <span>Combos</span>
                                <b>{seat.combos == null ? "—" : seat.combos.toFixed(1)}</b>
                            </div>
                        </div>
                    </article>
                );
            })}
        </div>
    );
}

export default function SpotOverview({ spot, generation }: { spot: SpotSummary; generation?: number }) {
    const [tab, setTab] = useState<(typeof tabs)[number]>("Overview");
    const result = useSolverEquity(generation, spot.nodeId, tab !== "Overview");
    const data = result?.data;
    const potOdds = spot.toCall && spot.toCall > 0 ? spot.toCall / (spot.pot + spot.toCall) : undefined;
    const blindOrder = postflopOrder(spot.seats.map((s) => s.position));
    const board = (
        <button
            type="button"
            className="overview-board-button"
            aria-label="Select public cards"
            disabled={!spot.onBoard}
            onClick={spot.onBoard}
        >
            {spot.board.map((card, index) => (
                <BoardCard key={index} card={card} />
            ))}
            {spot.board.length === 0 && <span>No flop</span>}
        </button>
    );
    const basePot = spot.basePot ?? spot.pot;
    return (
        <section className="overview-panel">
            <div className="inspector-tabs" role="tablist" aria-label="Spot information">
                {tabs.map((name, index) => (
                    <button
                        id={`spot-tab-${name.replace(" ", "-")}`}
                        aria-controls="spot-information"
                        type="button"
                        key={name}
                        role="tab"
                        aria-selected={tab === name}
                        tabIndex={tab === name ? 0 : -1}
                        onClick={() => setTab(name)}
                        onKeyDown={(event) => {
                            if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
                            event.preventDefault();
                            const next = (index + (event.key === "ArrowLeft" ? -1 : 1) + tabs.length) % tabs.length;
                            setTab(tabs[next]);
                            (event.currentTarget.parentElement?.children[next] as HTMLButtonElement)?.focus();
                        }}
                    >
                        {name}
                    </button>
                ))}
            </div>
            <div
                className={`spot-body${tab === "Overview" ? "" : " expanded"}`}
                id="spot-information"
                role="tabpanel"
                aria-labelledby={`spot-tab-${tab.replace(" ", "-")}`}
            >
                {tab === "Overview" ? (
                    <div className="overview-row">
                        <div className="overview-seats">
                            {blindOrder
                                .map((position) => spot.seats.find((s) => s.position === position)!)
                                .map((seat) => (
                                    <button
                                        type="button"
                                        key={seat.position}
                                        onClick={() => spot.onSeat?.(seat.position)}
                                        title={`Review ${seat.position}`}
                                        className={`overview-seat${seat.folded ? " folded" : ""}${seat.acting ? " acting" : ""}`}
                                    >
                                        <b>{seat.position}</b>
                                        <span>{formatNumber(seat.stack)}</span>
                                    </button>
                                ))}
                        </div>
                        <div className="overview-copy">
                            <div>
                                <span>Pot</span>
                                <strong>{formatNumber(spot.pot)}</strong>
                            </div>
                            <div>
                                <span>Street start</span>
                                <strong>{formatNumber(basePot)}</strong>
                            </div>
                            <div
                                title={
                                    potOdds === undefined
                                        ? "No call to make"
                                        : `Call ${formatNumber(spot.toCall!)} / ${formatNumber(spot.pot + spot.toCall!)} pot after calling. Break-even equity if no further betting.`
                                }
                            >
                                <span>Pot odds ⓘ</span>
                                <strong>{potOdds === undefined ? "—" : `${(potOdds * 100).toFixed(1)}%`}</strong>
                            </div>
                        </div>
                        {board}
                    </div>
                ) : (
                    <div className="spot-detail-view">
                        {tab === "Table" ? (
                            <div className="poker-table" aria-label="Table">
                                <div className="table-center">
                                    <div>
                                        <strong>{formatNumber(spot.pot)} pot</strong>
                                        {potOdds !== undefined && <small>{(potOdds * 100).toFixed(0)}%</small>}
                                    </div>
                                    {board}
                                    <small>
                                        <i className="pot-chip" />
                                        {formatNumber(basePot)} start
                                    </small>
                                </div>
                                {spot.seats.map((seat, index) => {
                                    const angle = ((index + 1) / spot.seats.length) * Math.PI * 2 - Math.PI;
                                    const left = 50 + 43 * Math.cos(angle),
                                        top = 50 + 40 * Math.sin(angle);
                                    return (
                                        <div key={seat.position}>
                                            <button
                                                type="button"
                                                className={`table-seat${seat.folded ? " folded" : ""}${seat.acting ? " acting" : ""}`}
                                                style={{ left: `${left}%`, top: `${top}%` }}
                                                onClick={() => spot.onSeat?.(seat.position)}
                                            >
                                                <b>{seat.position}</b>
                                                <span>{formatNumber(seat.stack)}</span>
                                                {seat.position === "BTN" && <i className="dealer-marker">D</i>}
                                            </button>
                                            {!!seat.committed && !seat.folded && (
                                                <span
                                                    className="table-contribution"
                                                    style={{
                                                        left: `${50 + (left - 50) * 0.64}%`,
                                                        top: `${50 + (top - 50) * 0.64}%`,
                                                    }}
                                                >
                                                    <i className="pot-chip" />
                                                    {formatNumber(seat.committed)}
                                                </span>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        ) : data && spot.players ? (
                            <EquityChart data={data} names={spot.players} selected={spot.selectedHand} />
                        ) : (
                            <div className="equity-empty" role="status">
                                {spot.nodeId === undefined
                                    ? "Equity is available after solving a heads-up flop."
                                    : result?.error
                                      ? result.error
                                      : "Calculating equity…"}
                            </div>
                        )}
                        <PlayerStats spot={spot} data={data} />
                    </div>
                )}
            </div>
        </section>
    );
}
