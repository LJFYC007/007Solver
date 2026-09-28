import { useMemo } from "react";
import { type DecisionNode, HAND_CLASSES } from "../solver";
import type { DisplayMode, EvComparison, HandFilters, HandRow } from "../solver/display";
import type { DetailCombo } from "../solver/study";
import FiltersPanel from "./FiltersPanel";
import HandDetails from "./HandDetails";
import SummaryTable from "./SummaryTable";
import Tabs from "./Tabs";

const tabs = ["Hands", "Summary", "Filters"] as const;
export type HandsTab = (typeof tabs)[number];

export interface PostflopHands {
    node: DecisionNode;
    /** The actor's hands at the node. */
    rows: HandRow[];
    /** Whether the filters currently apply. */
    filtered: boolean;
    filters: HandFilters;
    onFilters: (filters: HandFilters) => void;
    /** Why equity is missing, if it is. */
    equityStatus?: string;
}

export default function HandsPanel({
    tab,
    onTab,
    selected,
    locked,
    mode,
    comparison,
    details,
    postflop,
    onPick,
}: {
    tab: HandsTab;
    onTab: (tab: HandsTab) => void;
    selected: string;
    locked: boolean;
    mode: DisplayMode;
    comparison?: EvComparison;
    details: (label: string) => DetailCombo[];
    postflop?: PostflopHands;
    onPick: (label: string) => void;
}) {
    const summary = useMemo(
        () =>
            tab === "Summary"
                ? HAND_CLASSES.flatMap((label) => details(label).filter((combo) => combo.weight > 0 && combo.matches))
                : [],
        [tab, details],
    );
    const combos = details(selected);
    const active = tab === "Filters" && !postflop ? "Hands" : tab;
    return (
        <section className="hand-panel">
            <Tabs
                tabs={tabs}
                selected={active}
                onSelect={onTab}
                label="Hand information"
                id="hand-information"
                className="inspector-tabs hand-tabs"
                panelClassName="hand-body"
                disabled={postflop ? undefined : { Filters: "Filters need a postflop decision" }}
                detail={
                    <span className="hand-tabs-detail">
                        {active === "Summary"
                            ? `${summary.length} combinations`
                            : `${selected} · ${combos.length} combinations${locked ? " · locked" : ""}`}
                    </span>
                }
            >
                {active === "Hands" ? (
                    <HandDetails
                        label={selected}
                        combos={combos}
                        mode={mode}
                        pot={postflop?.node.state.pot}
                        comparison={comparison}
                    />
                ) : active === "Summary" ? (
                    <SummaryTable
                        combos={summary}
                        postflop={!!postflop}
                        filtered={!!postflop?.filtered}
                        selected={selected}
                        onPick={onPick}
                    />
                ) : (
                    postflop && (
                        <FiltersPanel
                            node={postflop.node}
                            rows={postflop.rows}
                            filters={postflop.filters}
                            equityStatus={postflop.equityStatus}
                            onChange={postflop.onFilters}
                        />
                    )
                )}
            </Tabs>
        </section>
    );
}
