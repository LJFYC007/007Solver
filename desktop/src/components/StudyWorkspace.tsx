import { useEffect, useMemo, useRef, useState } from "react";
import {
    type ChanceNode,
    type DecisionNode,
    type Player,
    type SolverNode,
    type SolverStatus,
    aggregateActions,
    formatNumber,
    handClassCombos,
    isForcedRunout,
    orderedActions,
} from "../solver";
import {
    aggregatePreflopActions,
    comboCount,
    postflopScenario,
    preflopActionColor,
    probabilities,
    replayPreflop,
    type PostflopScenario,
} from "../solver/preflop";
import {
    catalog,
    choiceKey,
    nodeFor,
    solutionFor,
    spotCount,
    type PreflopChoice,
    type PreflopNode,
    type TableFormat,
} from "../solver/catalog";
import { useSolverNavigation } from "../hooks/useSolverNavigation";
import BoardPicker from "./BoardPicker";
import { BoardCard } from "./PlayingCards";
import HandDetails, { ActionSummary, type DetailCombo } from "./HandDetails";
import SpotOverview, { type SpotSeat } from "./SpotOverview";
import { PreflopRangeMatrix, PostflopRangeMatrix } from "./RangeMatrix";
import SolvePanel from "./SolvePanel";
import SolutionSettings from "./SolutionSettings";
import PreflopTimeline from "./PreflopTimeline";
import { PostflopTimeline } from "./PostflopTimeline";

const handKey = (cards: string[]) => cards.slice().sort().join("");
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

export default function StudyWorkspace({
    root,
    status,
    generation,
    scenario,
    changing,
    onSolve,
    onInvalidate,
}: {
    root?: SolverNode;
    status: SolverStatus;
    generation?: number;
    scenario?: PostflopScenario;
    changing: boolean;
    onSolve: (scenario: PostflopScenario) => Promise<void>;
    onInvalidate: () => Promise<void>;
}) {
    const [format, setFormat] = useState<TableFormat>("6max");
    const [history, setHistory] = useState<PreflopChoice[]>([]);
    const [preIndex, setPreIndex] = useState<number | null>(0);
    const [selected, setSelected] = useState("AA");
    const [board, setBoard] = useState<string[]>([]);
    const [iterations, setIterations] = useState(3000);
    const [accuracyPercent, setAccuracyPercent] = useState(0.01);
    const [rangeSeat, setRangeSeat] = useState<string>();
    const [error, setError] = useState<string>();
    const [picker, setPicker] = useState<{ kind: "flop" } | { kind: "runout"; node: ChanceNode; index: number }>();
    const scroll = useRef<HTMLElement>(null);
    const editing = useRef(false);
    const navigation = useSolverNavigation(root, generation);
    const showPostflop = preIndex === null;
    const solution = solutionFor(format);
    const fullState = useMemo(() => replayPreflop(format, history), [format, history]);
    const shownHistory = useMemo(() => history.slice(0, preIndex ?? history.length), [history, preIndex]);
    const state = useMemo(() => replayPreflop(format, shownHistory), [format, shownHistory]);
    const preNode = nodeFor(format, shownHistory);
    const alive = state.seats.filter((s) => !s.folded);
    const headsUp = fullState.complete && fullState.seats.filter((s) => !s.folded).length === 2;
    const preflopAllIn =
        fullState.complete &&
        fullState.seats.filter((s) => !s.folded).length > 1 &&
        fullState.seats.filter((s) => !s.folded && s.committed < solution.stack).length < 2;
    const canPlayPostflop = headsUp && !preflopAllIn;
    const preActor = state.seats.find((s) => s.position === preNode?.actor);
    const shownSeat = preActor ?? alive.find((s) => s.position === rangeSeat) ?? alive[0];
    const range = shownSeat.range;
    const available = useMemo(
        () => (preNode ? Object.fromEntries(Object.entries(range).filter(([h]) => !!preNode.hands[h])) : range),
        [preNode, range],
    );
    const preCombos = comboCount(available);
    const players = useMemo<Record<Player, string>>(
        () => ({
            hero: scenario?.heroPosition ?? "IP",
            villain: scenario?.villainPosition ?? "OOP",
        }),
        [scenario],
    );
    const current = showPostflop ? navigation.node : undefined;
    const decision =
        current?.kind === "decision"
            ? current
            : current?.kind === "chance" && !isForcedRunout(current)
              ? navigation.path
                    .slice(0, navigation.activeIndex)
                    .reverse()
                    .find((node): node is DecisionNode => node.kind === "decision")
              : undefined;
    const timeline = useMemo(() => {
        const entries: { node: PreflopNode; history: PreflopChoice[]; stack: number }[] = [];
        function add(node: PreflopNode, prefix: PreflopChoice[]) {
            const seat = replayPreflop(format, prefix).seats.find((s) => s.position === node.actor)!;
            entries.push({ node, history: prefix, stack: solution.stack - seat.committed });
        }
        for (let i = 0; i < history.length; i++) {
            const prefix = history.slice(0, i),
                node = nodeFor(format, prefix);
            if (node) add(node, prefix);
        }
        let prefix = history;
        for (let i = 0; i < solution.positions.length; i++) {
            const node = nodeFor(format, prefix);
            if (!node) break;
            add(node, prefix);
            if (!node.actions.some((a) => a.label === "Fold")) break;
            prefix = [...prefix, { actor: node.actor, action: "Fold" }];
        }
        return entries;
    }, [format, history, solution]);
    useEffect(() => {
        const frame = window.requestAnimationFrame(() =>
            scroll.current
                ?.querySelector(".active")
                ?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" }),
        );
        return () => window.cancelAnimationFrame(frame);
    }, [preIndex, navigation.activeIndex, navigation.path.length, history.length, root]);
    function viewPre(index: number) {
        if (editing.current) return;
        navigation.suspend();
        setPreIndex(index);
        setPicker(undefined);
        setError(undefined);
    }
    function openFlop() {
        if (editing.current || busy) return;
        navigation.suspend();
        setPicker({ kind: "flop" });
    }
    async function choose(prefix: PreflopChoice[], actor: string, action: string) {
        if (editing.current) return;
        const next = [...prefix, { actor, action }];
        if (choiceKey(history.slice(0, next.length)) === choiceKey(next)) {
            viewPre(next.length);
            if (next.length === history.length && canPlayPostflop && board.length === 0) openFlop();
            return;
        }
        await changeLine(next);
    }
    async function changeStudy(apply: () => void) {
        if (editing.current) return;
        editing.current = true;
        navigation.suspend();
        setPicker(undefined);
        try {
            await onInvalidate();
            apply();
            setError(undefined);
        } catch (failure) {
            setError(String(failure));
        } finally {
            editing.current = false;
        }
    }
    async function changeLine(next: PreflopChoice[]) {
        await changeStudy(() => {
            const nextState = replayPreflop(format, next);
            const nextPlayers = nextState.seats.filter((s) => !s.folded);
            setHistory(next);
            setPreIndex(next.length);
            setBoard([]);
            setRangeSeat(undefined);
            setPicker(
                nextState.complete && nextPlayers.length === 2 && nextPlayers.every((s) => s.committed < solution.stack)
                    ? { kind: "flop" }
                    : undefined,
            );
        });
    }
    async function reset(nextFormat = format) {
        await changeStudy(() => {
            setFormat(nextFormat);
            setHistory([]);
            setPreIndex(0);
            setBoard([]);
            setRangeSeat(undefined);
        });
    }
    async function solve() {
        if (editing.current || busy) return;
        try {
            const next = postflopScenario(format, history, board, iterations, accuracyPercent);
            setError(undefined);
            setPreIndex(null);
            await onSolve(next);
        } catch (failure) {
            setError(String(failure));
        }
    }
    function viewPost(index: number) {
        if (editing.current) return;
        navigation.selectPath(index);
        setPreIndex(null);
        setPicker(undefined);
        setError(undefined);
    }
    async function actPost(index: number, nodeId: number) {
        if (editing.current) return;
        setPreIndex(null);
        setPicker(undefined);
        const node = await navigation.selectChild(nodeId, index);
        if (node?.kind === "chance" && !isForcedRunout(node)) setPicker({ kind: "runout", node, index: index + 1 });
    }
    function pickRunout(index: number) {
        if (editing.current) return;
        const node = navigation.path[index];
        if (node?.kind === "chance" && !isForcedRunout(node)) {
            viewPost(index);
            setPicker({ kind: "runout", node, index });
        }
    }

    const postActions = useMemo(() => (decision ? orderedActions(aggregateActions(decision)) : []), [decision]);
    const preActions = useMemo(() => (preNode ? aggregatePreflopActions(preNode, range) : []), [preNode, range]);
    const actionCards = showPostflop
        ? postActions.map((action) => ({
              ...action,
              id: String(action.index),
              onSelect:
                  current?.kind === "decision" && !navigation.navigationPending && !changing
                      ? () => void actPost(navigation.activeIndex, current.actions[action.index].nextNodeId)
                      : undefined,
          }))
        : preActions.map((action) => ({
              ...action,
              onSelect: changing || !preNode ? undefined : () => void choose(shownHistory, preNode.actor, action.label),
          }));
    const handsByCards = useMemo(() => new Map(decision?.hands.map((hand) => [handKey(hand.cards), hand])), [decision]);
    const selectedWeights = preNode ? probabilities(preNode, selected) : undefined;
    const selectedActions =
        preNode && selectedWeights
            ? preNode.actions.map((action, index) => ({
                  id: action.code,
                  label: action.label,
                  color: preflopActionColor(action),
                  probability: selectedWeights[index],
              }))
            : [];
    const detailCombos: DetailCombo[] = handClassCombos(selected).map((cards) => {
        if (showPostflop) {
            const hand = handsByCards.get(handKey(cards));
            const blocked = current?.state.board.some((c) => cards.includes(c));
            return {
                cards,
                weight: hand?.ownReachWeight ?? 0,
                actions:
                    hand && hand.ownReachWeight > 0 && hand.strategy.length === decision?.actions.length
                        ? postActions.map((a) => ({
                              id: String(a.index),
                              label: a.label,
                              color: a.color,
                              probability: hand.strategy[a.index],
                          }))
                        : [],
                description: blocked
                    ? "Blocked by board"
                    : !hand
                      ? "Not in range"
                      : hand.ownReachWeight === 0
                        ? "Zero own reach"
                        : hand.marginalReachMass === 0
                          ? "No compatible opponent"
                          : `EV ${hand.nodeStrategyEv?.toFixed(3) ?? "—"}`,
            };
        }
        return {
            cards,
            weight: range[selected] ?? 0,
            actions: selectedActions,
            description:
                preNode && !selectedWeights ? "No captured strategy" : `Range weight ${percent(range[selected] ?? 0)}`,
        };
    });
    const streetRoot = current ? navigation.path.find((node) => node.state.street === current.state.street) : undefined;
    const seats = useMemo(() => {
        const seats: SpotSeat[] = (showPostflop ? fullState : state).seats.map((s) => ({
            position: s.position,
            stack: solution.stack - s.committed,
            folded: s.folded,
            acting: !showPostflop && preNode?.actor === s.position,
            committed: showPostflop ? undefined : s.committed,
            combos: comboCount(s.range),
        }));
        let actorEv: number | null = null;
        if (current?.kind === "decision") {
            const mass = current.hands.reduce((sum, hand) => sum + hand.marginalReachMass, 0);
            if (mass > 0)
                actorEv =
                    current.hands.reduce((sum, hand) => sum + (hand.nodeStrategyEv ?? 0) * hand.marginalReachMass, 0) /
                    mass;
        }
        if (current && scenario) {
            for (const player of ["hero", "villain"] as const) {
                const seat = seats.find((s) => s.position === players[player]);
                if (!seat) continue;
                seat.stack = current.state.stacks[player];
                seat.committed = (streetRoot?.state.stacks[player] ?? seat.stack) - seat.stack;
                seat.acting = current.kind === "decision" && current.actor === player;
                seat.combos = current.state.rangeCombos[player];
                if (current.kind === "terminal" && current.result.reason === "fold") {
                    seat.folded = current.result.foldedBy === player;
                    seat.ev = seat.folded ? 0 : current.state.pot;
                }
                if (current.kind === "decision" && actorEv !== null)
                    seat.ev = current.actor === player ? actorEv : current.state.pot - actorEv;
            }
        }
        return seats;
    }, [showPostflop, fullState, state, solution, preNode, current, scenario, players, streetRoot]);
    const busy = status.state !== "idle" && status.state !== "ready" && status.state !== "failed";
    const toCall =
        current?.kind === "decision"
            ? current.actions.find((a) => a.kind === "call")?.chipsCommitted
            : preActor && preNode?.actions.some((action) => action.label === "Call")
              ? Math.max(...state.seats.map((s) => s.committed)) - preActor.committed
              : undefined;
    const boardAction =
        changing || busy
            ? undefined
            : current?.kind === "chance" && !isForcedRunout(current)
              ? () => pickRunout(navigation.activeIndex)
              : !showPostflop && canPlayPostflop
                ? openFlop
                : undefined;
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
                        entries={timeline}
                        history={history}
                        activeIndex={preIndex}
                        disabled={changing}
                        onView={(index) => {
                            const prefix = timeline[index].history;
                            if (prefix.length > history.length) void changeLine(prefix);
                            else viewPre(index);
                        }}
                        onAction={(entry, action) => void choose(entry.history, entry.node.actor, action)}
                    />
                    {fullState.complete && !root && (
                        <section className={`board-stage${preIndex === history.length ? " active" : ""}`}>
                            <button
                                type="button"
                                className="preflop-node-title"
                                onClick={() => viewPre(history.length)}
                            >
                                <strong>{canPlayPostflop ? "FLOP" : "Result"}</strong>
                                <span>{formatNumber(fullState.pot)}</span>
                            </button>
                            {canPlayPostflop ? (
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
                                <span>
                                    {fullState.seats.filter((s) => !s.folded).length === 1
                                        ? "Hand complete"
                                        : preflopAllIn
                                          ? "All-in · Betting complete"
                                          : "Multiway pot"}
                                </span>
                            )}
                        </section>
                    )}
                    <PostflopTimeline
                        path={navigation.path}
                        activeIndex={navigation.activeIndex}
                        visible={showPostflop}
                        disabled={navigation.navigationPending || changing}
                        players={players}
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
                        <strong>
                            {showPostflop
                                ? decision
                                    ? `${players[decision.actor]} strategy${current?.kind === "chance" ? " · last decision" : ""}`
                                    : "Postflop"
                                : preNode
                                  ? `${preNode.actor} strategy`
                                  : "Preflop ranges"}
                        </strong>
                        <span>
                            {showPostflop
                                ? decision
                                    ? `${decision.hands.reduce((s, h) => s + h.ownReachWeight, 0).toFixed(2)} combos`
                                    : ""
                                : `${preCombos.toFixed(2)} combos`}
                        </span>
                    </div>
                    <div className="strategy-content">
                        {showPostflop ? (
                            decision ? (
                                <PostflopRangeMatrix node={decision} selected={selected} onSelect={setSelected} />
                            ) : (
                                <div className="study-placeholder">
                                    <h2>
                                        {isForcedRunout(current)
                                            ? "All-in · Betting complete"
                                            : current?.kind === "terminal"
                                              ? "Line complete"
                                              : status.state === "failed"
                                                ? "Unable to solve"
                                                : "Solving strategy"}
                                    </h2>
                                    <p>
                                        {isForcedRunout(current)
                                            ? "No further betting decisions. Equity includes every legal runout."
                                            : current?.kind === "terminal"
                                              ? current.result.reason === "fold"
                                                  ? `${players[current.result.foldedBy]} folded.`
                                                  : "Showdown reached."
                                              : status.state === "failed"
                                                ? status.message
                                                : "Preparing your strategy. Follow progress in the solve panel."}
                                    </p>
                                </div>
                            )
                        ) : !preNode && !state.complete ? (
                            <div className="study-placeholder">
                                <h2>This branch is unavailable</h2>
                                <p>The saved GTO Wizard catalog has no strategy for this action history.</p>
                            </div>
                        ) : (
                            <PreflopRangeMatrix
                                node={preNode}
                                range={range}
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
                            seats,
                            pot: current?.state.pot ?? state.pot,
                            board: current?.state.board ?? (state.complete ? board : []),
                            toCall,
                            basePot: streetRoot?.state.pot,
                            players: current ? players : undefined,
                            showdown:
                                isForcedRunout(current) ||
                                (current?.kind === "terminal" && current.result.reason === "showdown"),
                            nodeId: current?.nodeId,
                            selectedHand: selected,
                            onSeat: (position) => {
                                if (editing.current) return;
                                if (!showPostflop && !preNode) setRangeSeat(position);
                                else if (showPostflop) {
                                    const index = navigation.path
                                        .map((entry, index) => ({ entry, index }))
                                        .slice(0, navigation.activeIndex + 1)
                                        .reverse()
                                        .find(
                                            ({ entry }) =>
                                                entry.kind === "decision" && players[entry.actor] === position,
                                        )?.index;
                                    if (index !== undefined) viewPost(index);
                                } else {
                                    const index = timeline
                                        .map((entry, index) => ({ entry, index }))
                                        .filter(({ entry }) => entry.history.length <= history.length)
                                        .reverse()
                                        .find(({ entry }) => entry.node.actor === position)?.index;
                                    if (index !== undefined) viewPre(index);
                                }
                            },
                            onBoard: boardAction,
                        }}
                    />
                    {state.complete && canPlayPostflop && (!showPostflop || busy || status.state === "failed") ? (
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
                            onBoard={openFlop}
                            onSolve={() => (root ? viewPost(navigation.activeIndex) : void solve())}
                            onResolve={() => void solve()}
                            onCancel={() => void changeStudy(() => setPreIndex(history.length))}
                            context={`Pot ${formatNumber(fullState.pot)} · ${fullState.seats
                                .filter((seat) => !seat.folded)
                                .map((seat) => `${seat.position} ${formatNumber(solution.stack - seat.committed)}`)
                                .join(" / ")}`}
                        />
                    ) : isForcedRunout(current) || (preflopAllIn && state.complete) || current?.kind === "terminal" ? (
                        <section className="line-complete" role="status">
                            <strong>Betting complete</strong>
                            <span>No more actions to choose.</span>
                        </section>
                    ) : (
                        <ActionSummary
                            actions={actionCards}
                            actor={showPostflop && decision ? players[decision.actor] : preNode?.actor}
                        />
                    )}
                    <HandDetails label={selected} combos={detailCombos} />
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
                    onClose={() => setPicker(undefined)}
                    onConfirm={(cards) => {
                        if (picker.kind === "runout") {
                            const outcome = picker.node.outcomes.find((o) => o.card === cards[cards.length - 1]);
                            if (outcome) void actPost(picker.index, outcome.nextNodeId);
                        } else {
                            if (cards.join() !== board.join())
                                void changeStudy(() => {
                                    setBoard(cards);
                                    setPreIndex(history.length);
                                });
                        }
                        setPicker(undefined);
                    }}
                />
            )}
        </main>
    );
}
