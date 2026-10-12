import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type DecisionNode, describeActions, formatNumber, isDesktopApp } from "../solver";
import {
    type ComparisonChoice,
    DISPLAY_MODES,
    type DisplayMode,
    type EvComparison,
    type HandFilters,
    NO_FILTERS,
    actorRows,
    type HandRow,
    filterRows,
    filtersUseEquity,
    hasFilters,
    resolveChoice,
    rowActions,
    selectRows,
    totalWeight,
    usesEquity,
    withEquity,
} from "../solver/display";
import {
    type NodeLock,
    editableHands,
    editableKeys,
    lockedNodeId,
    newLock,
    sameLock,
    strategies,
    withStrategies,
} from "../solver/locks";
import { postflopHandDetails } from "../solver/study";
import { useLockEditor } from "../hooks/useLockEditor";
import { requestReport, useNodeReport } from "../hooks/useNodeReport";
import { useStudyWorkspace, type StudyWorkspaceProps } from "../hooks/useStudyWorkspace";
import BoardPicker from "./BoardPicker";
import CompareEvControls from "./CompareEvControls";
import CompareView from "./CompareView";
import { FilterChips } from "./FiltersPanel";
import { BoardCard } from "./PlayingCards";
import { ActionSummary } from "./HandDetails";
import HandsPanel, { type HandsTab } from "./HandsPanel";
import LockEditor, { LockToolbar } from "./LockEditor";
import LockGlyph from "./LockGlyph";
import ModeMenu from "./ModeMenu";
import NodeLocks from "./NodeLocks";
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

// The decision's lock button, which gets focus back when the lock editor closes if its opener is gone.
const LOCK_BUTTON_ID = "lock-node-button";

const NO_CELLS: ReadonlySet<string> = new Set();

/** The matrix classes of the rows with these cards keys. */
const classesOf = (rows: readonly HandRow[], keys: ReadonlySet<string>) =>
    new Set(rows.filter((row) => keys.has(row.key)).map((row) => row.label));

function LoadedStudyWorkspace(props: StudyWorkspaceProps) {
    const { root, status, caches, changing, reference, appliedLocks, applying } = props;
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
        locks,
        setLocks,
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
        applyLocks,
        restore,
        showLock,
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
    const lockSelection = useCallback((label: string) => setSelection({ label, locked: true }), []);
    const [mode, setMode] = useState<DisplayMode>("strategy");
    const [matrixTab, setMatrixTab] = useState<"Strategy" | "Ranges" | "Compare">("Strategy");
    const [handsTab, setHandsTab] = useState<HandsTab>("Hands");
    const [filters, setFilters] = useState<HandFilters>(NO_FILTERS);
    const [choice, setChoice] = useState<ComparisonChoice>({ action: 0, versus: "best" });
    const editor = useLockEditor(caches);
    const { setFilters: filterInEditor, close: closeEditor } = editor;
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
    // The decision's node lock; the decision shows its strategies, also before it is applied.
    const nodeLock = locks.find((saved) => lockedNodeId(saved) === decision?.nodeId);
    // The study cannot move while a lock is edited, so the shown decision is the one the editor opened at.
    const { session } = editor;
    const editing = !!session;
    // The editor's filters select hands in it; the study's own wait until it closes.
    const activeFilters = session?.filters ?? filters;
    const modes = DISPLAY_MODES.filter((option) => decision || option.preflop).map((option) => option.id);
    const shownMode = modes.includes(mode) ? mode : "strategy";
    const shownTab = !decision || (matrixTab === "Compare" && reference === undefined) ? "Strategy" : matrixTab;
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
    const equity = useNodeReport(
        caches?.equity,
        decision?.nodeId,
        editing || shownTab === "Ranges" || usesEquity(shownMode) || handsTab !== "Hands" || filtersUseEquity(filters),
    );
    const report = equity?.data;
    const equityStatus = report ? undefined : (equity?.error ?? "Calculating equity…");
    const handRows = useMemo(() => (decision ? actorRows(decision) : []), [decision]);
    const equityRows = useMemo(
        () => (decision ? withEquity(handRows, decision.actor, report) : handRows),
        [decision, handRows, report],
    );
    const hands = nodeLock?.hands;
    const pinnedHands = nodeLock?.pinned ?? NO_CELLS;
    // Lock edits change only the strategies, not the equities.
    const rows = useMemo(
        () => (hands ? withStrategies(equityRows, (key) => hands[key]) : equityRows),
        [equityRows, hands],
    );
    // Classes with pinned hands show the pin in the lock editor, the only place where pins matter.
    const pinnedClasses = useMemo(() => classesOf(handRows, pinnedHands), [handRows, pinnedHands]);
    // Filters apply to the strategy view; the Ranges tab compares whole ranges. Equity buckets wait for equity.
    const filtering = !!decision && shownTab === "Strategy" && hasFilters(activeFilters);
    const equityPending = filtering && filtersUseEquity(activeFilters) && !report;
    const filtered = filtering && !equityPending;
    // The editor's filters each add hands to its selection; the study's narrow the hands shown.
    const shown = useMemo(
        () => (filtered ? (editing ? selectRows : filterRows)(rows, activeFilters) : rows),
        [filtered, editing, rows, activeFilters],
    );
    // A decision's actions show the mix of the rows the matrix shows, such as the filtered ones or the lock's.
    const shownMix = useMemo(() => (decision ? rowActions(decision, shown) : undefined), [decision, shown]);
    const actions = shownMix
        ? actionCards.map((card) => {
              const mix = shownMix.find((action) => action.index === card.index);
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
    // Node locks. While a locked solution is current, GTO means the reference solution.
    const referenceNode = useNodeReport(reference?.node, decision?.nodeId, editing || shownTab === "Compare");
    const gtoNode = reference === undefined ? decision : (referenceNode?.data as DecisionNode | undefined);
    const gtoError = referenceNode?.error && `Could not load the GTO strategies: ${referenceNode.error}`;
    // GTO shows once the reference report has loaded, never the current strategies in its place: hands without a GTO
    // strategy, which reach the decision only in the locked solution, show none.
    const gtoStrategies = useMemo(
        () => (editor.showGto && gtoNode ? strategies(gtoNode) : undefined),
        [editor.showGto, gtoNode],
    );
    const showGto = !!gtoStrategies;
    const gtoRows = useMemo(
        () => (gtoStrategies ? withStrategies(rows, (key) => gtoStrategies.get(key) ?? []) : rows),
        [gtoStrategies, rows],
    );
    // The editor's selection: the clicked classes, else the hands the filters select, else the whole range. Clicked
    // classes without a hand to set, such as ones outside the range, select the whole range too.
    const selectionKeys = useMemo(() => {
        if (!session || !decision) return undefined;
        const { cells } = session;
        if (!cells.size) return shownKeys;
        const keys = editableKeys(
            decision,
            handRows.filter((row) => cells.has(row.label)),
        );
        return keys.length ? new Set(keys) : undefined;
    }, [session, decision, handRows, shownKeys]);
    const selectedClasses = useMemo(
        () => selectionKeys && classesOf(handRows, selectionKeys),
        [selectionKeys, handRows],
    );
    // The hook's select is stable, so a selection change re-renders only the matrix cells it changes.
    const classSelection = session && { classes: selectedClasses, onSelect: editor.select };
    const updateLock = (change: Partial<NodeLock>) =>
        setLocks(locks.map((other) => (other === nodeLock ? { ...other, ...change } : other)));
    // Applying needs a GTO solution: the current one, or the locked solve's reference.
    const applyDisabled = changing || busy || applying || (status.state !== "ready" && reference === undefined);
    // Editing also needs the current solution.
    const editDisabled = applyDisabled || status.state !== "ready" || !root;
    const lockable =
        !editDisabled &&
        !editing &&
        showPostflop &&
        !navigation.navigationPending &&
        navigation.node?.kind === "decision" &&
        editableHands(navigation.node).length > 0;
    const isApplied = (saved: NodeLock) => !!appliedLocks?.some((other) => sameLock(other, saved));
    const lockMarks = new Map(locks.map((saved) => [lockedNodeId(saved), { applied: isApplied(saved) }]));
    // The locks differ from those the current solution was solved with, by value: added, edited or removed.
    const locksChanged = locks.some((saved) => !isApplied(saved)) || locks.length !== (appliedLocks?.length ?? 0);
    const applyAll = () => void applyLocks();
    // Closing the lock editor returns focus to the control that opened it.
    const opener = useRef<Element>(null);
    useEffect(() => {
        if (editing || !opener.current) return;
        // The decision's lock button replaces an opener that the editor's view removed, such as the list's Edit.
        const target = opener.current.isConnected ? opener.current : document.getElementById(LOCK_BUTTON_ID);
        opener.current = null;
        if (target instanceof HTMLElement) target.focus();
    }, [editing]);
    function openLock() {
        opener.current = document.activeElement;
        if (!nodeLock) setLocks([...locks, newLock(navigation.path, navigation.activeIndex, view.players)]);
        editor.open();
        setMatrixTab("Strategy");
    }
    async function editLock(saved: NodeLock) {
        const from = document.activeElement;
        // The editor's equity and GTO strategies load with the lock's path rather than after it.
        const nodeId = lockedNodeId(saved);
        if (caches) void requestReport(caches.equity, nodeId);
        if (reference) void requestReport(reference.node, nodeId);
        const node = await showLock(saved);
        if (!node) return;
        opener.current = from;
        editor.open();
        setMatrixTab("Strategy");
    }
    useEffect(() => {
        if (!editing) return;
        function onKey(event: KeyboardEvent) {
            if (event.key !== "Escape") return;
            if (selectionKeys) filterInEditor(NO_FILTERS);
            else closeEditor();
        }
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [editing, selectionKeys, filterInEditor, closeEditor]);
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
            <div className="study-browser" inert={editing}>
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
                        locks={lockMarks}
                        onPath={viewPost}
                        onAction={(...args) => void actPost(...args)}
                        onBoard={pickRunout}
                        onLock={lockable ? openLock : undefined}
                    />
                </nav>
            </div>
            <div className="workspace">
                <section className="strategy-panel">
                    <div className="panel-strip-title strategy-toolbar">
                        {session ? (
                            <LockToolbar
                                filters={session.filters}
                                showGto={showGto}
                                gtoAvailable={!!gtoNode}
                                onFilters={filterInEditor}
                                onToggleGto={editor.toggleGto}
                            />
                        ) : (
                            <>
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
                                        view={{
                                            selected: shownTab === "Strategy",
                                            onSelect: () => setMatrixTab("Strategy"),
                                        }}
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
                                    {reference !== undefined && (
                                        <button
                                            type="button"
                                            aria-pressed={shownTab === "Compare"}
                                            disabled={!decision}
                                            onClick={() => setMatrixTab("Compare")}
                                        >
                                            Compare
                                        </button>
                                    )}
                                </div>
                                {shownMode === "compareEv" && !!displayedActions?.length && (
                                    <CompareEvControls
                                        actions={displayedActions}
                                        choice={choice}
                                        onChange={setChoice}
                                    />
                                )}
                                {(filtering || (shownTab === "Compare" && hasFilters(filters))) && (
                                    <FilterChips filters={filters} onChange={setFilters} />
                                )}
                                <div className="strategy-title">
                                    <strong>{strategy.title}</strong>
                                    <span>{combos === undefined ? "" : `${combos.toFixed(2)} combos`}</span>
                                </div>
                            </>
                        )}
                    </div>
                    <div className="strategy-content">
                        {shownTab === "Ranges" && decision ? (
                            <RangesView
                                node={decision}
                                rows={rows}
                                report={report}
                                equityStatus={equityStatus}
                                names={view.players}
                                opponentEvCache={caches?.opponentEv}
                                selection={matrixSelection}
                            />
                        ) : shownTab === "Compare" && decision && reference ? (
                            <CompareView
                                node={decision}
                                rows={equityRows}
                                gto={gtoNode}
                                gtoError={gtoError}
                                equityStatus={equityStatus}
                                gtoEquityCache={reference.equity}
                                filters={filters}
                                names={view.players}
                                selection={matrixSelection}
                            />
                        ) : equityPending ? (
                            <div className="study-placeholder">
                                <h2>Filtering by equity</h2>
                                <p>{equityStatus}</p>
                            </div>
                        ) : editing && decision ? (
                            // Every class shows while a lock is edited; the selection stands out.
                            <PostflopRangeMatrix
                                node={decision}
                                rows={gtoRows}
                                mode="strategy"
                                filtered={false}
                                pinnedClasses={pinnedClasses}
                                selection={classSelection}
                                {...matrixSelection}
                            />
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
                <aside className={`inspector${view.panel === "solve" || editing ? " without-actions" : ""}`}>
                    <SpotOverview
                        equityCache={caches?.equity}
                        spot={{
                            ...view.spot,
                            selectedHand: selection.label,
                            onSeat: editing ? undefined : selectSeat,
                            onBoard: editing ? undefined : boardAction,
                        }}
                        detail={
                            reference !== undefined && (
                                <span className="locked-banner">
                                    <LockGlyph />
                                    <button type="button" disabled={changing} onClick={() => void restore()}>
                                        Back to GTO
                                    </button>
                                </span>
                            )
                        }
                    />
                    {session && nodeLock && decision ? (
                        <LockEditor
                            node={decision}
                            gto={gtoNode}
                            gtoError={gtoError}
                            rows={rows}
                            lock={nodeLock}
                            selection={selectionKeys}
                            filters={session.filters}
                            equityStatus={equityStatus}
                            pinnedActions={session.pinned}
                            onTogglePin={editor.togglePinned}
                            applyDisabled={applyDisabled}
                            onFilters={filterInEditor}
                            onChange={updateLock}
                            onExit={closeEditor}
                            onApply={applyAll}
                        />
                    ) : strategy.unavailable || loading ? (
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
                            lockCount={locks.length}
                            retryLocks={
                                status.state === "failed" && locks.length > 0 && reference !== undefined
                                    ? { preparing: applying, disabled: applyDisabled, onRetry: applyAll }
                                    : undefined
                            }
                            nodeLocks={
                                locks.length > 0 && (
                                    <NodeLocks
                                        locks={locks}
                                        isApplied={isApplied}
                                        solving={busy && !!appliedLocks}
                                        preparing={applying}
                                        applyDisabled={applyDisabled || !locksChanged}
                                        editDisabled={editDisabled}
                                        onEdit={(saved) => void editLock(saved)}
                                        onRemove={(saved) => setLocks(locks.filter((other) => other !== saved))}
                                        onApply={applyAll}
                                    />
                                )
                            }
                        />
                    ) : (
                        <>
                            {view.panel === "complete" ? (
                                <section className="line-complete" role="status">
                                    <strong>Betting complete</strong>
                                    <span>No more actions to choose.</span>
                                </section>
                            ) : (
                                <ActionSummary
                                    actions={actions}
                                    actor={strategy.actor}
                                    tool={
                                        lockable && (
                                            <button
                                                type="button"
                                                id={LOCK_BUTTON_ID}
                                                className="quiet-button lock-node-button"
                                                onClick={openLock}
                                            >
                                                <LockGlyph />
                                                {nodeLock ? "Edit lock" : "Lock node"}
                                            </button>
                                        )
                                    }
                                />
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
                                onPick={lockSelection}
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
