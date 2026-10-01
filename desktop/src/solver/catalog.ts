import plan from "../../../resources/gtowizard-preflop/capture-plan.json";
import type library from "../../../resources/gtowizard-preflop/library.json";

export type TableFormat = string;
export interface PreflopChoice {
    actor: string;
    action: string;
}
type ActionGroup = "FOLD" | "CALL" | "CHECK" | "BET_SMALL" | "BET_OVERBET";
/** GTO Wizard's action groups in the postflop action palette. */
const GROUP_COLORS: Record<ActionGroup, string> = {
    FOLD: "var(--action-fold)",
    CALL: "var(--action-call)",
    CHECK: "var(--action-check)",
    BET_SMALL: "var(--action-small)",
    BET_OVERBET: "var(--action-overbet)",
};
interface StoredAction {
    code: string;
    label: string;
    group: ActionGroup;
    next: { kind: "node" | "flop" | "showdown" | "hand-end" };
}
export interface PreflopAction extends Omit<StoredAction, "group"> {
    color: string;
    /** The action leads to a preflop node that has not been downloaded. */
    locked: boolean;
}
export interface PreflopNode {
    sourceUrl: string;
    actor: string;
    history: PreflopChoice[];
    actions: PreflopAction[];
    hands: Record<string, number[]>;
    incomingRanges?: Record<string, Record<string, number>>;
    nodeEvs?: Record<string, number | null>;
    actionEvs?: Record<string, (number | null)[]>;
    sourceWarning?: "ZERO_RANGE" | null;
}
/** A block file's node: EVs and incoming ranges are 169-class arrays in the manifest's handOrder. */
interface StoredNode extends Omit<PreflopNode, "actions" | "incomingRanges" | "nodeEvs" | "actionEvs"> {
    actions: StoredAction[];
    incomingRanges?: Record<string, number[]>;
    nodeEvs?: (number | null)[];
    /** One array per action, in the actions' order. */
    actionEvs?: (number | null)[][];
}
export type LibrarySolution = (typeof library.solutions)[number];
export interface PreflopSolution {
    schemaVersion: number;
    id: TableFormat;
    gameType: string;
    positions: string[];
    stack: number;
    smallBlind: number;
    bigBlind: number;
    ante: number;
    rake: string;
    rakePercent: number;
    rakeCap: number;
    openingSize: number;
    complete: boolean;
    handOrder: string[];
    /** The case's unlocked library listing. */
    listing: LibrarySolution;
}

const manifests = import.meta.glob<PreflopSolution>("../../../resources/gtowizard-preflop/cases/*/manifest.json", {
    eager: true,
    import: "default",
});
export const capturePlan = plan;
/** The whole captured listing, which only fills the library's filters; it loads on first use. */
export const loadLibrarySolutions = () =>
    import("../../../resources/gtowizard-preflop/library.json").then((module) => module.default.solutions);
/** Target cases with a saved root (partial or complete), joined with their listings. */
export const capturedSolutions = plan.targetCases.flatMap((id) => {
    const saved = manifests[`../../../resources/gtowizard-preflop/cases/${id}/manifest.json`];
    return saved?.schemaVersion === 3 ? [{ ...saved.listing, saved }] : [];
});
export type CapturedSolution = (typeof capturedSolutions)[number];
export const choiceKey = (history: PreflopChoice[]) =>
    JSON.stringify(history.map(({ actor, action }) => [actor, action]));
export const solutionFor = (format: TableFormat): PreflopSolution => {
    const solution = capturedSolutions.find((entry) => entry.id === format)?.saved;
    if (!solution) throw new Error("This solution has no downloaded root.");
    return solution;
};

// Only asset URLs enter the application bundle; strategy JSON is fetched on demand.
const files = import.meta.glob<string>(
    [
        "../../../resources/gtowizard-preflop/cases/*/index.json",
        "../../../resources/gtowizard-preflop/cases/*/chunks/*.json",
    ],
    { eager: true, query: "?url&no-inline", import: "default" },
);
type NodeIndex = Record<string, string>;
const indexes = new Map<TableFormat, NodeIndex>();
/** A case's index and the blocks one line reads: its history ancestors and the saved Fold preview. */
export interface PreflopLine {
    format: TableFormat;
    index: NodeIndex;
    /** Nodes by choiceKey in each loaded block file; the index assigns every history to one block. */
    blocks: Map<string, Map<string, PreflopNode>>;
}
/** A case's line before its first load, without nodes. */
export const unloadedLine = (format: TableFormat): PreflopLine => ({ format, index: {}, blocks: new Map() });

async function readFile<T>(format: TableFormat, file: string): Promise<T> {
    const url = files[`../../../resources/gtowizard-preflop/cases/${format}/${file}`];
    if (!url) throw new Error("Downloaded strategy file is missing. Rebuild after exporting the capture.");
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Unable to load strategy: HTTP ${response.status}`);
    return response.json() as Promise<T>;
}

async function readIndex(format: TableFormat) {
    const index = await readFile<NodeIndex>(format, "index.json");
    if (!index[choiceKey([])]) throw new Error("Downloaded root is missing from the index.");
    indexes.set(format, index);
    return index;
}

export function nodeFor(line: PreflopLine, history: PreflopChoice[]) {
    const key = choiceKey(history);
    return line.blocks.get(line.index[key])?.get(key);
}
/** The history after the node's Fold when that leads to a saved decision; ZERO_RANGE markers have no actions. */
export function foldPreview(node: PreflopNode) {
    const fold = node.actions.find((action) => action.label === "Fold");
    return fold?.next.kind === "node" && !fold.locked
        ? [...node.history, { actor: node.actor, action: fold.label }]
        : undefined;
}

/** Load history ancestors and the saved Fold preview, reusing the previous line's blocks of the same case. */
export async function loadPreflopLine(
    format: TableFormat,
    history: PreflopChoice[],
    previous: PreflopLine,
): Promise<PreflopLine> {
    const solution = solutionFor(format);
    const index = indexes.get(format) ?? (await readIndex(format));
    const line: PreflopLine = { format, index, blocks: new Map() };
    /** In-flight block reads, so prefixes loading in parallel fetch a shared block once. */
    const reads = new Map<string, Promise<Map<string, PreflopNode>>>();
    const byHand = <T>(values: (hand: number) => T) =>
        Object.fromEntries(solution.handOrder.map((hand, i) => [hand, values(i)]));
    async function readBlock(file: string) {
        const captured = await readFile<StoredNode[]>(format, file);
        const block = new Map<string, PreflopNode>();
        for (const { actions, incomingRanges, nodeEvs, actionEvs, ...node } of captured) {
            const key = choiceKey(node.history);
            const source = new URL(node.sourceUrl);
            if (
                block.has(key) ||
                index[key] !== file ||
                !solution.positions.includes(node.actor) ||
                source.searchParams.get("gametype") !== solution.gameType ||
                source.searchParams.get("depth") !== String(solution.stack)
            )
                throw new Error("Strategy block does not match its case index.");
            block.set(key, {
                ...node,
                actions: actions.map(({ group, ...action }) => ({
                    ...action,
                    color: GROUP_COLORS[group],
                    locked:
                        action.next.kind === "node" &&
                        !index[choiceKey([...node.history, { actor: node.actor, action: action.label }])],
                })),
                incomingRanges:
                    incomingRanges &&
                    Object.fromEntries(
                        Object.entries(incomingRanges).map(([position, weights]) => [
                            position,
                            byHand((i) => weights[i]),
                        ]),
                    ),
                nodeEvs: nodeEvs && byHand((i) => nodeEvs[i]),
                actionEvs: actionEvs && byHand((i) => actionEvs.map((evs) => evs[i])),
            });
        }
        return block;
    }
    async function load(prefix: PreflopChoice[]) {
        const file = index[choiceKey(prefix)];
        if (!file) return undefined;
        let block = line.blocks.get(file) ?? (previous.format === format ? previous.blocks.get(file) : undefined);
        if (!block) {
            if (!reads.has(file)) reads.set(file, readBlock(file));
            block = await reads.get(file)!;
        }
        line.blocks.set(file, block);
        const node = block.get(choiceKey(prefix));
        if (!node) throw new Error("Indexed strategy node is missing from its block.");
        return node;
    }
    // The index names every ancestor's block, so they load in parallel.
    const ancestors = await Promise.all(history.map((_, i) => load(history.slice(0, i))));
    if (!ancestors.every(Boolean)) throw new Error("This history contains an undownloaded branch.");
    let prefix: PreflopChoice[] | undefined = history;
    for (let i = 0; prefix && i < solution.positions.length; i++) {
        const node = await load(prefix);
        prefix = node && foldPreview(node);
    }
    return line;
}
