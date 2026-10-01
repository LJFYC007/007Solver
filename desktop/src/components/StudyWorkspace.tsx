import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { describeActions, formatNumber, isDesktopApp } from "../solver";
import {
    type ComparisonChoice,
    DISPLAY_MODES,
    type DisplayMode,
    type EvComparison,
    type HandFilters,
    NO_FILTERS,
    actorRows,
    filterRows,
    filtersUseEquity,
    hasFilters,
    resolveChoice,
    rowActions,
    totalWeight,
    usesEquity,
    withEquity,
} from "../solver/display";
import { postflopHandDetails } from "../solver/study";
import { type ReportCache, useNodeReport } from "../hooks/useNodeReport";
import { useStudyWorkspace, type StudyWorkspaceProps } from "../hooks/useStudyWorkspace";
import BoardPicker from "./BoardPicker";
import CompareEvControls from "./CompareEvControls";
import { FilterChips } from "./FiltersPanel";
import { BoardCard } from "./PlayingCards";
import { ActionSummary } from "./HandDetails";
import HandsPanel, { type HandsTab } from "./HandsPanel";
import ModeMenu from "./ModeMenu";
import RangesView from "./RangesView";
import SpotOverview from "./SpotOverview";
import { PreflopRangeMatrix, PostflopRangeMatrix } from "./RangeMatrix";
import SolvePanel from "./SolvePanel";
import SolutionSettings from "./SolutionSettings";
import PreflopTimeline from "./PreflopTimeline";
import { PostflopTimeline } from "./PostflopTimeline";
import WindowControls from "./WindowControls";
import { capturedSolutions } from "../solver/catalog";

export default function StudyWorkspace(props: StudyWorkspaceProps) {
    if (!capturedSolutions.length)
        return (
            <main className={`app-shell${isDesktopApp ? "" : " browser"}`}>
                <StudyHeader />
                <div className="study-browser">
                    <SolutionSettings disabled={false} onChange={() => {}} onReset={() => {}} />
                    <nav className="spot-timeline" aria-label="Action history" />
                </div>
                <div className="workspace">
                    <section className="strategy-panel">
                        <div className="panel-strip-title strategy-toolbar">Strategy</div>
                        <div className="study-placeholder">
                            <h2>🔒 No downloaded solutions</h2>
                            <p>Open Change to browse the solutions library.</p>
                            <p>Saved branches unlock as the daily capture progresses.</p>
                        </div>
                    </section>
                    <aside className="inspector">
                        <section className="line-complete">Select a downloaded solution to study.</section>
                    </aside>
                </div>
            </main>
        );
    return <LoadedStudyWorkspace {...props} />;
}

function StudyHeader() {
    return (
        isDesktopApp && (
            <header className="app-header" data-tauri-drag-region>
                <div className="brand" data-tauri-drag-region>
                    <div className="brand-mark" data-tauri-drag-region>
                        007
                    </div>
                    <strong data-tauri-drag-region>007 Solver</strong>
                    <span className="preflop-nav-label" data-tauri-drag-region>
                        Study
                    </span>
                </div>
                <WindowControls />
            </header>
        )
    );
}

function LoadedStudyWorkspace(props: StudyWorkspaceProps) {
    const { root, status, generation, changing } = props;
    const {
        format,
        loading,
        history,
        preIndex,
        board,
        iterations,
        setIterations,
        accuracyPercent,
        setAccuracyPercent,
        bettingTree,
        changeBettingTree,
        solveSettingsChanged,
        resetSolveSettings,
        error,
        picker,
        navigation,
        view,
        actionCards,
        boardAction,
        viewPre,
        viewPreTimeline,
        openFlop,
        choose,
        reset,
        solve,
        viewPost,
        actPost,
        pickRunout,
        selectSeat,
        confirmBoard,
        cancel,
        closePicker,
    } = useStudyWorkspace(props);
    // Hovering shows a hand class until a click locks it; clicking the locked class unlocks it.
    const [selection, setSelection] = useState({ label: "AA", locked: false });
    const hover = useCallback(
        (label: string) => setSelection((s) => (s.locked || s.label === label ? s : { label, locked: false })),
        [],
    );
    const pick = useCallback(
        (label: string) => setSelection((s) => ({ label, locked: !(s.locked && s.label === label) })),
        [],
    );
    const lock = useCallback((label: string) => setSelection({ label, locked: true }), []);
    const [mode, setMode] = useState<DisplayMode>("strategy");
    const [matrixTab, setMatrixTab] = useState<"Strategy" | "Ranges">("Strategy");
    const [handsTab, setHandsTab] = useState<HandsTab>("Hands");
    const [filters, setFilters] = useState<HandFilters>(NO_FILTERS);
    const [choice, setChoice] = useState<ComparisonChoice>({ action: 0, versus: "best" });
    const scroll = useRef<HTMLElement>(null);
    const { preflop, strategy, busy, showPostflop } = view;
    const solveLabel =
        status.state === "queued"
            ? "Waiting for GPU"
            : busy
              ? "Solving…"
              : status.state === "ready"
                ? `Solved · ${Math.round(status.elapsedSeconds)}s`
                : status.state === "failed"
                  ? "Solve failed"
                  : "Not solved";
    const decision = strategy.matrix.kind === "postflop" ? strategy.matrix.node : undefined;
    const modes = DISPLAY_MODES.filter((option) => decision || option.preflop).map((option) => option.id);
    const shownMode = modes.includes(mode) ? mode : "strategy";
    const shownTab = decision ? matrixTab : "Strategy";
    const displayedActions = decision && describeActions(decision);
    // Compare EV's choice as node action indices at this decision.
    const comparison = useMemo((): EvComparison | undefined => {
        if (!displayedActions?.length) return undefined;
        const { action, versus } = resolveChoice(choice, displayedActions.length);
        return {
            action: displayedActions[action].index,
            versus: versus === "best" ? "best" : displayedActions[versus].index,
        };
    }, [displayedActions, choice]);
    // Every generation change starts a new cache: the app renders without a generation between solves, and a
    // new web-server session numbers its solves from the start again.
    const equityCache = useMemo<ReportCache<"equity">>(
        () => ({ generation, kind: "equity", reports: new Map() }),
        [generation],
    );
    const opponentEvCache = useMemo<ReportCache<"opponentEv">>(
        () => ({ generation, kind: "opponentEv", reports: new Map() }),
        [generation],
    );
    const equity = useNodeReport(
        equityCache,
        decision?.nodeId,
        shownTab === "Ranges" || usesEquity(shownMode) || handsTab !== "Hands" || filtersUseEquity(filters),
    );
    const report = equity?.data;
    const equityStatus = report ? undefined : (equity?.error ?? "Calculating equity…");
    const handRows = useMemo(() => (decision ? actorRows(decision) : []), [decision]);
    const rows = useMemo(
        () => (decision ? withEquity(handRows, decision.actor, report) : handRows),
        [decision, handRows, report],
    );
    // Filters apply to the strategy view; the Ranges tab compares whole ranges. Equity buckets wait for equity.
    const filtering = !!decision && shownTab === "Strategy" && hasFilters(filters);
    const equityPending = filtering && filtersUseEquity(filters) && !report;
    const filtered = filtering && !equityPending;
    const shown = useMemo(() => (filtered ? filterRows(rows, filters) : rows), [filtered, rows, filters]);
    const filteredMix = useMemo(
        () => (filtered && decision ? rowActions(decision, shown) : undefined),
        [filtered, decision, shown],
    );
    const actions = filteredMix
        ? actionCards.map((card) => {
              const mix = filteredMix.find((action) => action.index === card.index);
              return { ...card, probability: mix?.probability ?? 0, combos: mix?.combos ?? 0 };
          })
        : actionCards;
    const combos = filtered ? totalWeight(shown) : strategy.combos;
    const matrixSelection = { selected: selection.label, locked: selection.locked, onHover: hover, onPick: pick };
    const postflopHands = decision && { node: decision, rows, filtered, filters, onFilters: setFilters, equityStatus };
    const rowsByKey = useMemo(() => new Map(rows.map((row) => [row.key, row])), [rows]);
    const shownKeys = useMemo(() => (filtered ? new Set(shown.map((row) => row.key)) : undefined), [filtered, shown]);
    const handDetails = useMemo(
        () =>
            strategy.handDetails ??
            ((label: string) => postflopHandDetails(label, view.spot.board, rowsByKey, shownKeys, decision)),
        [strategy.handDetails, view.spot.board, rowsByKey, shownKeys, decision],
    );
    useEffect(() => {
        const frame = window.requestAnimationFrame(() =>
            scroll.current
                ?.querySelector(".active")
                ?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" }),
        );
        return () => window.cancelAnimationFrame(frame);
    }, [preIndex, navigation.activeIndex, navigation.path.length, history.length, root]);
    return (
        <main className={`app-shell${isDesktopApp ? "" : " browser"}`}>
            {/* Browsers have no window for a title bar to move or control. */}
            <StudyHeader />
            <div className="study-browser">
                <SolutionSettings
                    format={format}
                    disabled={changing || loading}
                    onChange={(value) => void reset(value)}
                    onReset={() => void reset()}
                />
                <nav ref={scroll} className="spot-timeline" aria-label="Action history">
                    <PreflopTimeline
                        entries={preflop.timeline}
                        history={history}
                        activeIndex={preIndex}
                        disabled={changing || loading}
                        onView={viewPreTimeline}
                        onAction={(entry, action) => void choose(entry.history, entry.node.actor, action)}
                    />
                    {preflop.complete && (
                        <section className={`board-stage${preIndex === history.length ? " active" : ""}`}>
                            <button
                                type="button"
                                className="timeline-node-title"
                                onClick={() => viewPre(history.length)}
                            >
                                <strong>{preflop.canPlayPostflop ? "FLOP" : "Result"}</strong>
                                <span>{formatNumber(preflop.pot)}</span>
                            </button>
                            {preflop.canPlayPostflop ? (
                                <>
                                    <button
                                        type="button"
                                        className="stage-cards"
                                        disabled={busy || changing}
                                        aria-label={board.length ? `Change flop ${board.join(" ")}` : "Select flop"}
                                        onClick={openFlop}
                                    >
                                        {Array.from({ length: 3 }, (_, i) => (
                                            <BoardCard card={board[i]} key={i} />
                                        ))}
                                    </button>
                                    {board.length === 3 && (
                                        <small className={`stage-status ${status.state}`}>{solveLabel}</small>
                                    )}
                                </>
                            ) : (
                                <span>{preflop.resultLabel}</span>
                            )}
                        </section>
                    )}
                    <PostflopTimeline
                        path={navigation.path}
                        activeIndex={navigation.activeIndex}
                        visible={showPostflop}
                        disabled={navigation.navigationPending || changing}
                        players={view.players}
                        onPath={viewPost}
                        onAction={(...args) => void actPost(...args)}
                        onBoard={pickRunout}
                    />
                </nav>
            </div>
            <div className="workspace">
                <section className="strategy-panel">
                    <div className="panel-strip-title strategy-toolbar">
                        <div className="matrix-tabs" role="group" aria-label="Matrix view">
                            <ModeMenu
                                mode={shownMode}
                                modes={DISPLAY_MODES.map((option) => option.id)}
                                enabled={modes}
                                disabledReason="Needs a postflop solution"
                                onMode={(next) => {
                                    setMode(next);
                                    setMatrixTab("Strategy");
                                }}
                                view={{ selected: shownTab === "Strategy", onSelect: () => setMatrixTab("Strategy") }}
                            />
                            <button
                                type="button"
                                aria-pressed={shownTab === "Ranges"}
                                disabled={!decision}
                                title={decision ? undefined : "Ranges need a postflop decision"}
                                onClick={() => setMatrixTab("Ranges")}
                            >
                                Ranges
                            </button>
                        </div>
                        {shownMode === "compareEv" && !!displayedActions?.length && (
                            <CompareEvControls actions={displayedActions} choice={choice} onChange={setChoice} />
                        )}
                        {filtering && <FilterChips filters={filters} onChange={setFilters} />}
                        <div className="strategy-title">
                            <strong>{strategy.title}</strong>
                            <span>{combos === undefined ? "" : `${combos.toFixed(2)} combos`}</span>
                        </div>
                    </div>
                    <div className="strategy-content">
                        {shownTab === "Ranges" && decision ? (
                            <RangesView
                                node={decision}
                                rows={rows}
                                report={report}
                                equityStatus={equityStatus}
                                names={view.players}
                                opponentEvCache={opponentEvCache}
                                selection={matrixSelection}
                            />
                        ) : equityPending ? (
                            <div className="study-placeholder">
                                <h2>Filtering by equity</h2>
                                <p>{equityStatus}</p>
                            </div>
                        ) : strategy.matrix.kind === "postflop" ? (
                            <PostflopRangeMatrix
                                node={strategy.matrix.node}
                                rows={shown}
                                mode={shownMode}
                                comparison={comparison}
                                filtered={filtered}
                                {...matrixSelection}
                            />
                        ) : strategy.matrix.kind === "empty" ? (
                            <div className="study-placeholder">
                                <h2>{strategy.matrix.title}</h2>
                                <p>{strategy.matrix.description}</p>
                            </div>
                        ) : (
                            <PreflopRangeMatrix
                                node={strategy.matrix.node}
                                range={strategy.matrix.range}
                                mode={shownMode}
                                {...matrixSelection}
                            />
                        )}
                    </div>
                </section>
                <aside className={`inspector${view.panel === "solve" ? " without-actions" : ""}`}>
                    <SpotOverview
                        equityCache={equityCache}
                        spot={{
                            ...view.spot,
                            selectedHand: selection.label,
                            onSeat: selectSeat,
                            onBoard: boardAction,
                        }}
                    />
                    {strategy.unavailable || loading ? (
                        <section className="line-complete" role="status">
                            <strong>{loading ? "Loading strategy…" : strategy.unavailable}</strong>
                        </section>
                    ) : view.panel === "solve" ? (
                        <SolvePanel
                            board={board}
                            status={status}
                            busy={busy}
                            changing={changing}
                            iterations={iterations}
                            accuracyPercent={accuracyPercent}
                            settingsChanged={solveSettingsChanged}
                            onIterations={setIterations}
                            onAccuracyPercent={setAccuracyPercent}
                            bettingTree={bettingTree}
                            onBettingTree={changeBettingTree}
                            onSolve={() => void solve()}
                            onCancel={cancel}
                            onReset={resetSolveSettings}
                            scenario={props.scenario}
                            matchup={preflop.matchup}
                            error={error}
                        />
                    ) : (
                        <>
                            {view.panel === "complete" ? (
                                <section className="line-complete" role="status">
                                    <strong>Betting complete</strong>
                                    <span>No more actions to choose.</span>
                                </section>
                            ) : (
                                <ActionSummary actions={actions} actor={strategy.actor} />
                            )}
                            <HandsPanel
                                tab={handsTab}
                                onTab={setHandsTab}
                                selected={selection.label}
                                locked={selection.locked}
                                mode={shownMode}
                                comparison={comparison}
                                details={handDetails}
                                postflop={postflopHands}
                                onPick={lock}
                            />
                        </>
                    )}
                </aside>
            </div>
            {(error || navigation.navigationError) && (
                <div className="study-error" role="alert">
                    {error ?? navigation.navigationError}
                </div>
            )}
            {picker && (
                <BoardPicker
                    key={picker.kind === "flop" ? "flop" : picker.node.nodeId}
                    title={
                        picker.kind === "flop"
                            ? "Select flop"
                            : `Select ${picker.node.state.street === "flop" ? "turn" : "river"}`
                    }
                    board={picker.kind === "flop" ? board : picker.node.state.board}
                    count={picker.kind === "flop" ? 3 : picker.node.state.board.length + 1}
                    locked={picker.kind === "flop" ? 0 : picker.node.state.board.length}
                    available={picker.kind === "runout" ? picker.node.outcomes.map((o) => o.card) : undefined}
                    onClose={closePicker}
                    onConfirm={confirmBoard}
                />
            )}
        </main>
    );
}
