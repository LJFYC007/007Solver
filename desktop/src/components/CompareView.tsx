import { useMemo } from "react";
import {
    type ActionInfo,
    type AggregatedAction,
    type DecisionNode,
    type Player,
    actionFrequency,
    describeActions,
    strategyGradient,
} from "../solver";
import {
    type HandFilters,
    type HandRow,
    PLAYERS,
    actorRows,
    filterRows,
    filtersUseEquity,
    formatMetric,
    hasFilters,
    percent,
    postflopCells,
    rowActions,
    withEquity,
} from "../solver/display";
import { type ReportCache, useNodeReport } from "../hooks/useNodeReport";
import { RangeMatrix, type Selection } from "./RangeMatrix";

const chips = (value: number | null | undefined) => formatMetric("ev", value ?? undefined) ?? "—";

/**
 * One solution at the decision: the cells and action mix of the actor's rows, with equities, that the filters keep;
 * the mix is empty when none of them has a strategy.
 */
function side(node: DecisionNode, all: HandRow[], filters: HandFilters) {
    const filtered = hasFilters(filters);
    const rows = filtered ? filterRows(all, filters) : all;
    return { cells: postflopCells({ node, rows, mode: "strategy", filtered }), mix: rowActions(node, rows) };
}

/**
 * GTO Wizard's node comparison: the locked strategy beside the baseline's, overall and in EV. Both solve the same
 * tree, so they share node ids and list the same actions.
 */
export default function CompareView({
    node,
    rows,
    gto: gtoNode,
    gtoError,
    equityStatus,
    gtoEquityCache,
    filters,
    names,
    selection,
}: {
    /** The locked solution's decision and its actor's rows, with equities. */
    node: DecisionNode;
    rows: HandRow[];
    /** The decision in the GTO solution the locked one is compared with, once it has loaded. */
    gto?: DecisionNode;
    /** Why the GTO decision could not be loaded, if it could not. */
    gtoError?: string;
    /** Why the locked solution's equity report at the decision is missing, while it is. */
    equityStatus?: string;
    /** The GTO solution's equity reports. */
    gtoEquityCache: ReportCache<"equity">;
    filters: HandFilters;
    names: Record<Player, string>;
    selection: Selection;
}) {
    // Classified once, not again as equities arrive or filters change.
    const gtoHands = useMemo(() => gtoNode && actorRows(gtoNode), [gtoNode]);
    const needsEquity = filtersUseEquity(filters);
    const gtoEquity = useNodeReport(gtoEquityCache, node.nodeId, needsEquity);
    const status = equityStatus ?? (gtoEquity?.data ? undefined : (gtoEquity?.error ?? "Calculating equity…"));
    const locked = useMemo(() => side(node, rows, filters), [node, rows, filters]);
    const gto = useMemo(
        () => gtoNode && gtoHands && side(gtoNode, withEquity(gtoHands, gtoNode.actor, gtoEquity?.data), filters),
        [gtoNode, gtoHands, gtoEquity?.data, filters],
    );
    if (needsEquity && status)
        return (
            <div className="study-placeholder">
                <h2>Filtering by equity</h2>
                <p>{status}</p>
            </div>
        );
    const actions = describeActions(node);
    return (
        <div className="ranges-view compare-view">
            <div className="ranges-top">
                <section className="ranges-side" aria-label="GTO strategy">
                    <header>
                        <strong>GTO</strong>
                    </header>
                    {gto ? (
                        <RangeMatrix cells={gto.cells} compact {...selection} />
                    ) : (
                        <p className="muted-copy">{gtoError ?? "Loading the GTO strategies…"}</p>
                    )}
                </section>
                <div className="range-stats compare-stats">
                    <ActionTable actions={actions} gto={gto?.mix} locked={locked.mix} />
                    <div className="compare-mix" aria-hidden="true">
                        <span>GTO</span>
                        <i style={{ background: strategyGradient(gto?.mix ?? []) }} />
                        <span>Locked</span>
                        <i style={{ background: strategyGradient(locked.mix) }} />
                    </div>
                    <EvTable node={node} gto={gtoNode} names={names} />
                </div>
                <section className="ranges-side" aria-label="Locked strategy">
                    <header>
                        <strong>Locked</strong>
                    </header>
                    <RangeMatrix cells={locked.cells} compact {...selection} />
                </section>
            </div>
        </div>
    );
}

function ActionTable({
    actions,
    gto,
    locked,
}: {
    actions: readonly ActionInfo[];
    gto?: AggregatedAction[];
    locked: AggregatedAction[];
}) {
    return (
        <table className="compare-table" aria-label="Action frequencies">
            <thead>
                <tr>
                    <th scope="col">Action</th>
                    <th scope="col">GTO</th>
                    <th scope="col">Locked</th>
                </tr>
            </thead>
            <tbody>
                {actions.map((action) => (
                    <tr key={action.index}>
                        <th scope="row" title={action.size && `${action.label} · ${action.size} pot`}>
                            <i style={{ background: action.color }} />
                            {action.label}
                        </th>
                        <td>{percent(actionFrequency(gto, action.index))}</td>
                        <td>{percent(actionFrequency(locked, action.index))}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

function EvTable({ node, gto, names }: { node: DecisionNode; gto?: DecisionNode; names: Record<Player, string> }) {
    return (
        <table className="compare-table" aria-label="Range EVs">
            <thead>
                <tr>
                    <th scope="col">EV</th>
                    <th scope="col">GTO</th>
                    <th scope="col">Locked</th>
                </tr>
            </thead>
            <tbody>
                {PLAYERS.map((player) => (
                    <tr key={player}>
                        <th scope="row">{names[player]}</th>
                        <td>{chips(gto?.rangeEvs[player])}</td>
                        <td>{chips(node.rangeEvs[player])}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}
