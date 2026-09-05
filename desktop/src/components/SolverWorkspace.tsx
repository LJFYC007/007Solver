import { Fragment, useMemo, useState } from "react";
import {
    type ChanceNode,
    type DecisionNode,
    type HandGroup,
    type SolverNode,
    RANKS,
    SUITS,
    SUIT_SYMBOLS,
    actionLabel,
    aggregateActions,
    formatNumber,
    groupHands,
    playerLabel,
} from "../solver";
import { type PathEntry, useSolverNavigation } from "../hooks/useSolverNavigation";
import Inspector from "./Inspector";
import { Board } from "./PlayingCards";
import RangeMatrix from "./RangeMatrix";

function AppHeader({ iterations, nodeCount }: { iterations: number; nodeCount: number }) {
    return (
        <header className="app-header">
            <div className="brand">
                <div className="brand-mark">007</div>
                <strong>007 Solver</strong>
            </div>

            <div className="header-solution">
                <span className="solved-dot" />
                <strong>Solution ready</strong>
                <small>
                    {iterations.toLocaleString()} iterations · {nodeCount.toLocaleString()} tree nodes
                </small>
            </div>
        </header>
    );
}

function DecisionActions({
    disabled,
    node,
    onSelect,
    selectedNodeId,
}: {
    disabled: boolean;
    node: DecisionNode;
    onSelect?: (nodeId: number, label: string) => void;
    selectedNodeId?: number;
}) {
    return (
        <div className="node-action-list">
            {node.actions.map((action) => {
                const label = actionLabel(action);
                if (!onSelect) {
                    return (
                        <span
                            className={selectedNodeId === action.nextNodeId ? "selected" : undefined}
                            key={action.nextNodeId}
                        >
                            {label}
                        </span>
                    );
                }

                return (
                    <button
                        disabled={disabled}
                        key={action.nextNodeId}
                        onClick={() => onSelect(action.nextNodeId, label)}
                        type="button"
                    >
                        {label}
                    </button>
                );
            })}
        </div>
    );
}

function BoardStage({ board, node }: { board: string[]; node: SolverNode }) {
    return (
        <div className={`board-stage${board.length === 1 ? " single-card" : ""}`}>
            <span className="path-node-heading">
                <strong>{node.state.street.toUpperCase()}</strong>
                <b>{formatNumber(node.state.pot)}</b>
            </span>
            <Board board={board} />
        </div>
    );
}

function Timeline({
    activeIndex,
    disabled,
    path,
    onPathSelect,
    onSelectChild,
}: {
    activeIndex: number;
    disabled: boolean;
    path: PathEntry[];
    onPathSelect: (index: number) => void;
    onSelectChild: (nodeId: number, label: string) => void;
}) {
    return (
        <div className="spot-timeline">
            {path.map((entry, index) => {
                const previousNode = path[index - 1]?.node;
                const board = previousNode
                    ? entry.node.state.board.slice(previousNode.state.board.length)
                    : entry.node.state.board;
                const isCurrent = index === activeIndex;
                const isFuture = index > activeIndex;
                const displayedStack =
                    entry.node.kind === "decision"
                        ? entry.node.state.stacks[entry.node.actor]
                        : Math.min(entry.node.state.stacks.hero, entry.node.state.stacks.villain);
                const nodeLabel =
                    entry.node.kind === "decision"
                        ? playerLabel(entry.node.actor)
                        : entry.node.kind === "chance"
                          ? "Runout"
                          : "Terminal";

                return (
                    <Fragment key={`${entry.node.nodeId}-${index}`}>
                        {board.length > 0 && <BoardStage board={board} node={entry.node} />}
                        {isCurrent ? (
                            <div className="current-node-card active">
                                <span className="path-node-heading">
                                    <strong>{nodeLabel}</strong>
                                    <b>{formatNumber(displayedStack)}</b>
                                </span>
                                {entry.node.kind === "decision" && (
                                    <DecisionActions disabled={disabled} node={entry.node} onSelect={onSelectChild} />
                                )}
                                {entry.node.kind === "chance" && (
                                    <strong className="current-node-message">Choose next card</strong>
                                )}
                                {entry.node.kind === "terminal" && (
                                    <strong className="current-node-message">Line complete</strong>
                                )}
                            </div>
                        ) : (
                            <button
                                className={`history-node-card${isFuture ? " future" : ""}`}
                                onClick={() => onPathSelect(index)}
                                type="button"
                            >
                                <span className="path-node-heading">
                                    <strong>{nodeLabel}</strong>
                                    <b>{formatNumber(displayedStack)}</b>
                                </span>
                                {entry.node.kind === "decision" && (
                                    <DecisionActions
                                        disabled
                                        node={entry.node}
                                        selectedNodeId={path[index + 1]?.node.nodeId}
                                    />
                                )}
                                {entry.node.kind !== "decision" && (
                                    <span className="history-node-choice">
                                        {entry.node.kind === "chance"
                                            ? (path[index + 1]?.label ?? "Choose card")
                                            : "Line complete"}
                                    </span>
                                )}
                            </button>
                        )}
                    </Fragment>
                );
            })}
        </div>
    );
}

function SpotBrowser({
    activeIndex,
    path,
    navigationPending,
    navigationError,
    onPathSelect,
    onSelectChild,
}: {
    activeIndex: number;
    navigationError?: string;
    navigationPending: boolean;
    path: PathEntry[];
    onPathSelect: (index: number) => void;
    onSelectChild: (nodeId: number, label: string) => void;
}) {
    return (
        <section className="spot-browser">
            <Timeline
                activeIndex={activeIndex}
                disabled={navigationPending}
                onPathSelect={onPathSelect}
                onSelectChild={onSelectChild}
                path={path}
            />
            {navigationError && <span className="timeline-error">{navigationError}</span>}
        </section>
    );
}

function ChancePicker({
    disabled,
    node,
    onSelect,
}: {
    disabled: boolean;
    node: ChanceNode;
    onSelect: (nodeId: number, label: string) => void;
}) {
    const outcomeByCard = new Map(node.outcomes.map((outcome) => [outcome.card, outcome]));

    return (
        <div className="chance-picker">
            <div className="chance-hero">
                <span className="chance-icon">✦</span>
                <h3>Choose the next card</h3>
                <p>Select one of the {node.outcomes.length} legal runouts to continue through the strategy.</p>
            </div>
            <div className="card-grid" aria-label="Available runout cards">
                {SUITS.flatMap((suit) =>
                    RANKS.map((rank) => {
                        const card = `${rank}${suit}`;
                        const outcome = outcomeByCard.get(card);
                        return (
                            <button
                                className={`runout-card suit-${suit}`}
                                disabled={!outcome || disabled}
                                key={card}
                                onClick={() => outcome && onSelect(outcome.nextNodeId, outcome.card)}
                                title={
                                    outcome
                                        ? `Deal ${rank}${SUIT_SYMBOLS[suit]}`
                                        : `${rank}${SUIT_SYMBOLS[suit]} is unavailable`
                                }
                                type="button"
                            >
                                <span>{rank}</span>
                                <span>{SUIT_SYMBOLS[suit]}</span>
                            </button>
                        );
                    }),
                )}
            </div>
        </div>
    );
}

function StrategyPanel({
    navigationPending,
    groups,
    node,
    selected,
    onSelectChild,
    onHoverHand,
}: {
    groups: Map<string, HandGroup>;
    navigationPending: boolean;
    node: SolverNode;
    selected?: string;
    onSelectChild: (nodeId: number, label: string) => void;
    onHoverHand: (label: string) => void;
}) {
    const weightedCombos =
        node.kind === "decision" ? node.hands.reduce((sum, hand) => sum + Math.max(0, hand.ownReachWeight), 0) : 0;

    return (
        <main className="strategy-panel">
            <div className="panel-strip-title">
                <strong>Range strategy</strong>
                <div className="strategy-toolbar">
                    {node.kind === "decision" && <span>{weightedCombos.toFixed(1)} own-reach combos</span>}
                </div>
            </div>

            <div className="strategy-content">
                {node.kind === "decision" && (
                    <RangeMatrix board={node.state.board} groups={groups} onHover={onHoverHand} selected={selected} />
                )}
                {node.kind === "chance" && (
                    <ChancePicker disabled={navigationPending} node={node} onSelect={onSelectChild} />
                )}
                {node.kind === "terminal" && (
                    <div className="terminal-state">
                        <div className="terminal-mark">✓</div>
                        <h3>Terminal node reached</h3>
                        <p>
                            {node.result.reason === "fold"
                                ? `${playerLabel(node.result.foldedBy)} folded.`
                                : "The action reached showdown; payoff information is not included."}
                        </p>
                    </div>
                )}
            </div>
        </main>
    );
}

export default function SolverWorkspace({
    iterations,
    nodeCount,
    root,
}: {
    iterations: number;
    nodeCount: number;
    root: SolverNode;
}) {
    const [hoveredHand, setHoveredHand] = useState<{ nodeId: number; label: string }>();
    const { activeIndex, navigationError, navigationPending, node, path, selectChild, selectPath } =
        useSolverNavigation(root);
    const actions = useMemo(() => (node.kind === "decision" ? aggregateActions(node) : []), [node]);
    const groups = useMemo(() => (node.kind === "decision" ? groupHands(node) : new Map<string, HandGroup>()), [node]);
    const firstReachable = [...groups.values()].find((group) => group.ownReachWeight > 0);
    const selectedHand =
        hoveredHand?.nodeId === node.nodeId ? hoveredHand.label : (firstReachable?.label ?? [...groups.keys()][0]);
    const selectedGroup = selectedHand ? groups.get(selectedHand) : undefined;

    return (
        <div className="app-shell">
            <AppHeader iterations={iterations} nodeCount={nodeCount} />
            <SpotBrowser
                activeIndex={activeIndex}
                navigationError={navigationError}
                navigationPending={navigationPending}
                onPathSelect={selectPath}
                onSelectChild={selectChild}
                path={path}
            />
            <div className="workspace">
                <StrategyPanel
                    groups={groups}
                    navigationPending={navigationPending}
                    node={node}
                    onSelectChild={selectChild}
                    onHoverHand={(label) => setHoveredHand({ nodeId: node.nodeId, label })}
                    selected={selectedHand}
                />
                <Inspector actions={actions} node={node} selectedGroup={selectedGroup} />
            </div>
        </div>
    );
}
