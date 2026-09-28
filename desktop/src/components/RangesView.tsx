import { type CSSProperties, useMemo, useState } from "react";
import { type DecisionNode, type EquityReport, type Player, nodeEv, strategyGradient } from "../solver";
import { type ReportCache, useNodeReport } from "../hooks/useNodeReport";
import {
    PLAYERS,
    PLAYER_COLORS,
    postflopCells,
    type DisplayMode,
    type HandRow,
    categoryGroups,
    equityRealization,
    isEquitySection,
    opponentRows,
    percent,
} from "../solver/display";
import { FILTER_SECTIONS } from "../solver/handCategories";
import EquityChart from "./EquityChart";
import ModeMenu from "./ModeMenu";
import { RangeMatrix, type Selection } from "./RangeMatrix";

// The waiting player has no strategy at the node, and its EVs take a separate, slower query.
const ACTOR_MODES: DisplayMode[] = ["strategy", "ev", "equity", "eqr", "range"];
const WAITING_MODES: DisplayMode[] = ["ev", "equity", "eqr", "range"];
// The waiting player's matrix shows its range only; the Hands panel follows the acting player.
const ignore = () => {};
const BUCKET_SECTIONS = FILTER_SECTIONS.filter((section) => isEquitySection(section.id));

function StatRow({ name, values, shares }: { name: string; values: string[]; shares: number[] }) {
    const mix = shares.map((share, i) => ({ color: PLAYER_COLORS[PLAYERS[i]], probability: share }));
    return (
        <div className="range-stat">
            <b>{values[0]}</b>
            <span>{name}</span>
            <b>{values[1]}</b>
            <div className="range-stat-bar" aria-hidden="true" style={{ background: strategyGradient(mix) }} />
        </div>
    );
}

export default function RangesView({
    node,
    rows,
    report,
    equityStatus,
    names,
    opponentEvCache,
    selection,
}: {
    node: DecisionNode;
    rows: HandRow[];
    report?: EquityReport;
    equityStatus?: string;
    names: Record<Player, string>;
    opponentEvCache: ReportCache<"opponentEv">;
    selection: Selection;
}) {
    const [actorMode, setActorMode] = useState<DisplayMode>("strategy");
    const [waitingMode, setWaitingMode] = useState<DisplayMode>("equity");
    const [buckets, setBuckets] = useState<"eqSimple" | "eqAdvanced">("eqSimple");
    const { pot, rangeCombos } = node.state;
    const waitingEvs = waitingMode === "ev" || waitingMode === "eqr";
    const opponentEv = useNodeReport(opponentEvCache, node.nodeId, waitingEvs);
    const evStatus = waitingEvs && !opponentEv?.data ? (opponentEv?.error ?? "Calculating EV…") : undefined;
    const sides = useMemo(
        () =>
            PLAYERS.map((player) => {
                const acting = player === node.actor;
                return {
                    player,
                    acting,
                    // Other modes keep own-reach weights, whatever EV modes were visited before.
                    rows: acting
                        ? rows
                        : report
                          ? opponentRows(report, player, waitingEvs ? opponentEv?.data : undefined)
                          : [],
                };
            }),
        [node, rows, report, waitingEvs, opponentEv?.data],
    );
    const cells = useMemo(
        () =>
            sides.map((side) => postflopCells({ node, rows: side.rows, mode: side.acting ? actorMode : waitingMode })),
        [node, sides, actorMode, waitingMode],
    );
    const evs = useMemo(() => PLAYERS.map((player) => nodeEv(node, player)), [node]);
    const equities = PLAYERS.map((player) => report?.players[player].equity);
    const eqrs = PLAYERS.map((_, i) => equityRealization(evs[i], equities[i], pot));
    const bucketSection = BUCKET_SECTIONS.find((section) => section.id === buckets)!;
    const shares = useMemo(() => sides.map((side) => categoryGroups(side.rows, buckets).share), [sides, buckets]);
    return (
        <div className="ranges-view">
            <div className="ranges-top">
                {sides.map((side, index) => {
                    const mode = side.acting ? actorMode : waitingMode;
                    return (
                        <section
                            key={side.player}
                            className="ranges-side"
                            style={{ order: index * 2, "--player": PLAYER_COLORS[side.player] } as CSSProperties}
                            aria-label={`${names[side.player]} range`}
                        >
                            <header>
                                <strong>
                                    <i aria-hidden="true" />
                                    {names[side.player]}
                                </strong>
                                {side.acting ? <small>to act</small> : evStatus && <small>{evStatus}</small>}
                                <ModeMenu
                                    mode={mode}
                                    modes={side.acting ? ACTOR_MODES : WAITING_MODES}
                                    onMode={side.acting ? setActorMode : setWaitingMode}
                                />
                            </header>
                            {side.rows.length ? (
                                <RangeMatrix
                                    cells={cells[index]}
                                    compact
                                    {...selection}
                                    {...(side.acting ? {} : { onHover: ignore, onPick: ignore })}
                                />
                            ) : (
                                <p className="muted-copy">{equityStatus}</p>
                            )}
                        </section>
                    );
                })}
                <div className="range-stats" style={{ order: 1 }}>
                    <StatRow
                        name="Combos"
                        values={PLAYERS.map((player) => rangeCombos[player].toFixed(1))}
                        shares={PLAYERS.map((player) => rangeCombos[player])}
                    />
                    <StatRow
                        name="EV"
                        values={evs.map((ev) => (ev === undefined ? "—" : ev.toFixed(2)))}
                        shares={evs.map((ev) => ev ?? 0)}
                    />
                    <StatRow name="Equity" values={equities.map(percent)} shares={equities.map((e) => e ?? 0)} />
                    <StatRow name="EQR" values={eqrs.map(percent)} shares={eqrs.map((e) => e ?? 0)} />
                    <label className="range-buckets">
                        <select value={buckets} onChange={(event) => setBuckets(event.target.value as typeof buckets)}>
                            {BUCKET_SECTIONS.map((section) => (
                                <option key={section.id} value={section.id}>
                                    {section.title}
                                </option>
                            ))}
                        </select>
                    </label>
                    {bucketSection.categories.map((bucket) => {
                        const values = shares.map((share) => share(bucket.id));
                        return (
                            <StatRow
                                key={bucket.id}
                                name={bucket.label}
                                values={values.map(percent)}
                                shares={values.map((v) => v ?? 0)}
                            />
                        );
                    })}
                </div>
            </div>
            <div className="ranges-chart">
                {report ? (
                    <EquityChart data={report} names={names} selected={selection.selected} />
                ) : (
                    <div className="equity-empty" role="status">
                        {equityStatus}
                    </div>
                )}
            </div>
        </div>
    );
}
