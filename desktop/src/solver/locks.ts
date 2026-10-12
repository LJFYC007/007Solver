// Node locks: a locked decision keeps every hand's strategy while the solver re-solves everything else.

import type { HandRow } from "./display";
import type { PostflopScenario } from "./preflop";
import { SUIT_SYMBOLS, actionLabel, cardsKey, reachWeight } from "./strategy";
import type { DecisionNode, HandStrategy, Player, SolverNode } from "./types";

/** A step from the postflop root: the action index at a decision, the dealt card at a chance node. */
type LockStep = number | string;

/** A lock as the scenario states it: one weight per action for listed hands at the decision its path reaches. */
export interface ScenarioLock {
    path: LockStep[];
    hands: Record<string, number[]>;
}

/** The locked hands' strategies by cards key. */
export type LockedHands = Readonly<Record<string, number[]>>;

export interface NodeLock {
    /** The steps to the locked decision. */
    path: LockStep[];
    /** The nodes from the root to the locked decision; solves of one scenario share NodeIds. */
    nodeIds: number[];
    /** The decision's actor and the steps to it, as action labels and dealt cards. */
    label: string;
    /** Whether applying also locks every earlier decision without a lock of its own to its GTO strategy. */
    keepEarlier: boolean;
    /** Every editable hand's strategy (see editableHands). */
    hands: LockedHands;
    /** The hands set through a selection, which whole-range changes leave as they are. */
    pinned: ReadonlySet<string>;
}

export const lockedNodeId = (lock: NodeLock) => lock.nodeIds[lock.nodeIds.length - 1];
export const hasEarlierDecisions = (lock: NodeLock) => lock.path.some((step) => typeof step === "number");

/**
 * Hands with joint reach and action EVs, the only ones the editor locks; the service omits EVs where the compatible
 * opponent mass is too small to scale (see AnalysisSession).
 */
const isEditable = (node: DecisionNode, hand: HandStrategy) =>
    hand.strategy.length === node.actions.length && hand.actionEvs.length === node.actions.length;
export const editableHands = (node: DecisionNode) => node.hands.filter((hand) => isEditable(node, hand));
/** The editable hands with a cards key in `keys`, such as the selected hands, or every editable hand without `keys`. */
export const selectedHands = (node: DecisionNode, keys: ReadonlySet<string> | undefined) =>
    keys ? editableHands(node).filter((hand) => keys.has(cardsKey(hand.cards))) : editableHands(node);
export const editableKeys = (node: DecisionNode, rows: readonly HandRow[]) =>
    rows.filter((row) => row.hand && isEditable(node, row.hand)).map((row) => row.key);

/** Each hand's strategy by cards key, for the hands that have one: those with joint reach. */
export const strategies = (node: DecisionNode) =>
    new Map(
        node.hands
            .filter((hand) => hand.strategy.length === node.actions.length)
            .map((hand) => [cardsKey(hand.cards), hand.strategy]),
    );

/** The selection's editable hands, or the whole range's, at their locked strategies, or their own without one. */
export const scopeHands = (node: DecisionNode, hands: LockedHands, selection: ReadonlySet<string> | undefined) =>
    selectedHands(node, selection).map((hand) => ({ ...hand, strategy: hands[cardsKey(hand.cards)] ?? hand.strategy }));

/** Whether two locks lock the same hands of one decision to the same strategies, with the same options. */
export const sameLock = (a: NodeLock, b: NodeLock) =>
    a === b ||
    (lockedNodeId(a) === lockedNodeId(b) &&
        a.keepEarlier === b.keepEarlier &&
        Object.keys(a.hands).length === Object.keys(b.hands).length &&
        Object.entries(a.hands).every(([key, strategy]) =>
            strategy.every((weight, i) => weight === b.hands[key]?.[i]),
        ));

/** The rows with the strategies `strategy` gives by cards key; other rows keep theirs. */
export const withStrategies = (rows: readonly HandRow[], strategy: (key: string) => number[] | undefined) =>
    rows.map((row) => {
        const replaced = row.hand && strategy(row.key);
        return row.hand && replaced ? { ...row, hand: { ...row.hand, strategy: replaced } } : row;
    });

/** A new lock of the decision at `index` of a navigation path: its whole range at the strategies it shows. */
export function newLock(path: readonly SolverNode[], index: number, players: Record<Player, string>): NodeLock {
    const node = path[index] as DecisionNode;
    const steps: LockStep[] = [];
    const names: string[] = [];
    for (let i = 0; i < index; i++) {
        const step = path[i];
        const next = path[i + 1].nodeId;
        if (step.kind === "decision") {
            const action = step.actions.findIndex((candidate) => candidate.nextNodeId === next);
            steps.push(action);
            names.push(`${players[step.actor]} ${actionLabel(step.actions[action])}`);
        } else if (step.kind === "chance") {
            const card = step.outcomes.find((outcome) => outcome.nextNodeId === next)!.card;
            steps.push(card);
            names.push(card[0] + SUIT_SYMBOLS[card[1]]);
        }
    }
    return {
        path: steps,
        nodeIds: path.slice(0, index + 1).map((step) => step.nodeId),
        label: `${players[node.actor]} at ${names.join(" · ") || "Flop"}`,
        keepEarlier: true,
        hands: Object.fromEntries(editableHands(node).map((hand) => [cardsKey(hand.cards), hand.strategy])),
        pinned: new Set(),
    };
}

interface FrequencyHand {
    key: string;
    /** The hand's reachWeight, as in the aggregate frequencies. */
    weight: number;
    strategy: number[];
    evs: number[];
    /** The hand counts in the frequency but keeps its strategy. */
    held: boolean;
}

/**
 * New strategies of the hands that move an action's aggregate frequency toward `target`, leaving pinned actions and
 * held hands as they are. As in PioSolver, raising the action switches first the hands that lose the least EV by
 * playing it instead of their other free actions, which give up weight in proportion to their shares; lowering it is
 * symmetric, toward the other free actions by their shares, or the best of them for a hand that plays none. Hands
 * whose losses are within 1% of the pot of their tier's first hand change by the same fraction.
 */
function setFrequency(
    hands: readonly FrequencyHand[],
    action: number,
    target: number,
    pinned: ReadonlySet<number>,
    pot: number,
) {
    const changed = new Map<string, number[]>();
    const free = hands[0].strategy.map((_, i) => i).filter((i) => i !== action && !pinned.has(i));
    const total = hands.reduce((sum, hand) => sum + hand.weight, 0);
    const current = hands.reduce((sum, hand) => sum + hand.weight * hand.strategy[action], 0);
    const raising = target * total > current;
    // Mass of the action still to move. Reports hold single-precision strategies, so a sliver of it is rounding: no
    // reason to change another hand, nor to leave a hand short of a full move.
    let remaining = Math.abs(target * total - current);
    const sliver = 1e-6 * total;
    const done = () => remaining <= sliver;
    const moves = hands
        .flatMap((hand) => {
            const freeShare = free.reduce((sum, i) => sum + hand.strategy[i], 0);
            const share = raising ? freeShare : hand.strategy[action];
            if (hand.held || hand.weight <= 0 || share <= 0) return [];
            const best = free.reduce((a, b) => (hand.evs[b] > hand.evs[a] ? b : a));
            const mix = free.map((i) => (freeShare > 0 ? hand.strategy[i] / freeShare : i === best ? 1 : 0));
            const otherEv = free.reduce((sum, i, j) => sum + mix[j] * hand.evs[i], 0);
            return [{ hand, share, mix, loss: raising ? otherEv - hand.evs[action] : hand.evs[action] - otherEv }];
        })
        .sort((a, b) => a.loss - b.loss);
    for (let start = 0; start < moves.length && !done();) {
        let end = start + 1;
        while (end < moves.length && moves[end].loss <= moves[start].loss + pot / 100) end++;
        const tier = moves.slice(start, end);
        const capacity = tier.reduce((sum, move) => sum + move.hand.weight * move.share, 0);
        // A tier that the remaining mass covers but for a sliver moves fully, so 100% leaves none of the other actions.
        const fraction = remaining + sliver >= capacity ? 1 : remaining / capacity;
        for (const { hand, share, mix } of tier) {
            // Scaling instead of subtracting keeps every weight non-negative, which the solver requires.
            const strategy = hand.strategy.map((weight, i) =>
                free.includes(i) && raising ? weight * (1 - fraction) : weight,
            );
            if (raising) strategy[action] += share * fraction;
            else {
                strategy[action] *= 1 - fraction;
                free.forEach((i, j) => (strategy[i] += share * fraction * mix[j]));
            }
            changed.set(hand.key, strategy);
        }
        // A tier that reaches the target changes only partly, and later tiers keep their strategies.
        remaining = fraction < 1 ? 0 : remaining - capacity;
        start = end;
    }
    return changed;
}

/** The scopeHands that a move changes; a whole-range move holds the pinned hands, which still count in its frequency. */
function frequencyHands(
    node: DecisionNode,
    { hands, pinned }: Pick<NodeLock, "hands" | "pinned">,
    selection: ReadonlySet<string> | undefined,
): FrequencyHand[] {
    return scopeHands(node, hands, selection).map((hand) => {
        const key = cardsKey(hand.cards);
        return {
            key,
            weight: reachWeight(hand),
            strategy: hand.strategy,
            evs: hand.actionEvs,
            held: !selection && pinned.has(key),
        };
    });
}

/**
 * The lock's hands with strategies that move an action's aggregate frequency over the selection, or the whole range,
 * toward `target` (see setFrequency); the hands the move leaves alone lock to the strategies they have.
 */
export function lockFrequency(
    node: DecisionNode,
    lock: Pick<NodeLock, "hands" | "pinned">,
    selection: ReadonlySet<string> | undefined,
    action: number,
    target: number,
    pinnedActions: ReadonlySet<number>,
) {
    const hands = frequencyHands(node, lock, selection);
    const moved = setFrequency(hands, action, target, pinnedActions, node.state.pot);
    const locked = { ...lock.hands };
    for (const hand of hands) locked[hand.key] = moved.get(hand.key) ?? hand.strategy;
    return locked;
}

/**
 * For each action by node index, the lowest and highest aggregate frequency that lockFrequency can reach over the same
 * hands: held hands keep their shares, and the others can give the action up to the other free actions or take all
 * of theirs.
 */
export function frequencyLimits(
    node: DecisionNode,
    lock: Pick<NodeLock, "hands" | "pinned">,
    selection: ReadonlySet<string> | undefined,
    pinnedActions: ReadonlySet<number>,
): [number, number][] {
    const hands = frequencyHands(node, lock, selection);
    const total = hands.reduce((sum, hand) => sum + hand.weight, 0);
    const indices = node.actions.map((_, i) => i);
    return indices.map((action) => {
        const free = indices.filter((i) => i !== action && !pinnedActions.has(i));
        let low = 0;
        let high = 0;
        for (const { weight, strategy, held } of hands) {
            const own = weight * strategy[action];
            if (held || !free.length) {
                low += own;
                high += own;
            } else high += own + weight * free.reduce((sum, i) => sum + strategy[i], 0);
        }
        return [low / total, high / total];
    });
}

/**
 * The baseline scenario with the study's locks. Locks that lock earlier decisions to GTO also lock every decision on
 * their path that the study does not lock itself, both players', to the strategies of every hand that has one in the
 * node `load` returns for it.
 */
export async function lockedScenario(
    baseline: PostflopScenario,
    locks: readonly NodeLock[],
    load: (nodeId: number) => Promise<SolverNode>,
): Promise<PostflopScenario> {
    // By the locked decision's node id: a path reaches one node of the tree.
    const scenarioLocks = new Map(locks.map((lock) => [lockedNodeId(lock), { path: lock.path, hands: lock.hands }]));
    const earlier = new Map<number, LockStep[]>();
    for (const lock of locks)
        if (lock.keepEarlier)
            lock.path.forEach((step, i) => {
                const nodeId = lock.nodeIds[i];
                if (typeof step === "number" && !scenarioLocks.has(nodeId)) earlier.set(nodeId, lock.path.slice(0, i));
            });
    const decisions = [...earlier];
    const nodes = await Promise.all(decisions.map(([nodeId]) => load(nodeId)));
    decisions.forEach(([nodeId, path], i) =>
        scenarioLocks.set(nodeId, { path, hands: Object.fromEntries(strategies(nodes[i] as DecisionNode)) }),
    );
    return { ...baseline, locks: [...scenarioLocks.values()] };
}
