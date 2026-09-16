import { useEffect, useRef, useState } from "react";
import { formatNumber } from "../solver";
import { catalog, spotCount } from "../solver/catalog";
import { useStudyWorkspace, type StudyWorkspaceProps } from "../hooks/useStudyWorkspace";
import BoardPicker from "./BoardPicker";
import { BoardCard } from "./PlayingCards";
import HandDetails, { ActionSummary } from "./HandDetails";
import SpotOverview from "./SpotOverview";
import { PreflopRangeMatrix, PostflopRangeMatrix } from "./RangeMatrix";
import SolvePanel from "./SolvePanel";
import SolutionSettings from "./SolutionSettings";
import PreflopTimeline from "./PreflopTimeline";
import { PostflopTimeline } from "./PostflopTimeline";

export default function StudyWorkspace(props: StudyWorkspaceProps) {
    const { root, status, generation, changing } = props;
    const {
        format,
        history,
        preIndex,
        board,
        iterations,
        setIterations,
        accuracyPercent,
        setAccuracyPercent,
        bettingTree,
        changeBettingTree,
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
    const [selected, setSelected] = useState("AA");
    const scroll = useRef<HTMLElement>(null);
    const { preflop, strategy, busy, showPostflop } = view;
    useEffect(() => {
        const frame = window.requestAnimationFrame(() =>
            scroll.current
                ?.querySelector(".active")
                ?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" }),
        );
        return () => window.cancelAnimationFrame(frame);
    }, [preIndex, navigation.activeIndex, navigation.path.length, history.length, root]);
    return (
        <main className="app-shell">
            <header className="app-header">
                <div className="brand">
                    <div className="brand-mark">007</div>
                    <strong>007 Solver</strong>
                    <span className="preflop-nav-label">Study</span>
                </div>
                <div
                    className="study-status"
                    role="status"
                    title={
                        status.state === "ready"
                            ? `${status.iterations.toLocaleString()} iterations · ${Math.round(status.elapsedSeconds)} seconds · Target ${status.targetAccuracyPercent}% pot`
                            : undefined
                    }
                >
                    <span className={status.state === "ready" ? "solved-dot" : ""} />
                    {status.state === "ready"
                        ? `${status.stopReason === "accuracy" ? "Target reached" : "Iteration limit reached"}${status.accuracyPercent === null ? "" : ` · ${status.accuracyPercent.toPrecision(3)}% pot`}`
                        : busy
                          ? "Solving…"
                          : "GTO Wizard · chip EV · 100bb"}
                    {status.state === "ready" && (
                        <button
                            type="button"
                            className="quiet-button"
                            disabled={changing}
                            onClick={() => viewPre(history.length)}
                        >
                            Adjust solve
                        </button>
                    )}
                </div>
            </header>
            <div className="study-browser">
                <SolutionSettings
                    format={format}
                    disabled={changing}
                    onChange={(value) => void reset(value)}
                    onReset={() => void reset()}
                />
                <nav ref={scroll} className="spot-timeline" aria-label="Action history">
                    <PreflopTimeline
                        entries={preflop.timeline}
                        history={history}
                        activeIndex={preIndex}
                        disabled={changing}
                        onView={viewPreTimeline}
                        onAction={(entry, action) => void choose(entry.history, entry.node.actor, action)}
                    />
                    {preflop.complete && !root && (
                        <section className={`board-stage${preIndex === history.length ? " active" : ""}`}>
                            <button
                                type="button"
                                className="preflop-node-title"
                                onClick={() => viewPre(history.length)}
                            >
                                <strong>{preflop.canPlayPostflop ? "FLOP" : "Result"}</strong>
                                <span>{formatNumber(preflop.pot)}</span>
                            </button>
                            {preflop.canPlayPostflop ? (
                                <button
                                    type="button"
                                    className="stage-cards"
                                    disabled={busy || changing}
                                    aria-label="Select flop"
                                    onClick={openFlop}
                                >
                                    {Array.from({ length: 3 }, (_, i) => (
                                        <BoardCard card={board[i]} key={i} />
                                    ))}
                                </button>
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
                        onFlop={openFlop}
                    />
                </nav>
            </div>
            <div className="workspace">
                <section className="strategy-panel">
                    <div className="panel-strip-title">
                        <strong>{strategy.title}</strong>
                        <span>{strategy.combos === undefined ? "" : `${strategy.combos.toFixed(2)} combos`}</span>
                    </div>
                    <div className="strategy-content">
                        {strategy.matrix.kind === "postflop" ? (
                            <PostflopRangeMatrix
                                node={strategy.matrix.node}
                                selected={selected}
                                onSelect={setSelected}
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
                                selected={selected}
                                onSelect={setSelected}
                            />
                        )}
                    </div>
                    <footer className="study-footnote">
                        {showPostflop
                            ? "Current solution · exact cards and action history"
                            : `GTO Wizard · ${spotCount(format)} saved spots · ${catalog.capturedAt}`}
                        {(error || navigation.navigationError) && (
                            <span role="alert">{error ?? navigation.navigationError}</span>
                        )}
                    </footer>
                </section>
                <aside className="inspector">
                    <SpotOverview
                        generation={generation}
                        spot={{
                            ...view.spot,
                            selectedHand: selected,
                            onSeat: selectSeat,
                            onBoard: boardAction,
                        }}
                    />
                    {view.panel === "solve" ? (
                        <SolvePanel
                            board={board}
                            status={status}
                            busy={busy}
                            changing={changing}
                            ready={!!root}
                            iterations={iterations}
                            accuracyPercent={accuracyPercent}
                            onIterations={setIterations}
                            onAccuracyPercent={setAccuracyPercent}
                            bettingTree={bettingTree}
                            onBettingTree={(value) => void changeBettingTree(value)}
                            onBoard={openFlop}
                            onSolve={() => (root ? viewPost(navigation.activeIndex) : void solve())}
                            onResolve={() => void solve()}
                            onCancel={cancel}
                            context={view.solveContext}
                        />
                    ) : view.panel === "complete" ? (
                        <section className="line-complete" role="status">
                            <strong>Betting complete</strong>
                            <span>No more actions to choose.</span>
                        </section>
                    ) : (
                        <ActionSummary actions={actionCards} actor={strategy.actor} />
                    )}
                    <HandDetails label={selected} combos={strategy.handDetails(selected)} />
                </aside>
            </div>
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
