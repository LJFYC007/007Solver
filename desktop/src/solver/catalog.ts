import catalog from "../../../resources/gtowizard-preflop/catalog.json";

export type TableFormat = "6max" | "8max";
export interface PreflopChoice {
    actor: string;
    action: string;
}
export interface PreflopAction {
    code: string;
    label: string;
    color: string;
    displayedFrequency: string;
    displayedCombos: string;
}
export interface PreflopNode {
    sourceUrl: string;
    actor: string;
    history: PreflopChoice[];
    actions: PreflopAction[];
    hands: Record<string, number[]>;
}
export interface PreflopSolution {
    id: TableFormat;
    gameType: string;
    positions: string[];
    stack: number;
    smallBlind: number;
    bigBlind: number;
    ante: number;
    rake: string;
    openingSize: number;
}

export { catalog };
export const choiceKey = (history: PreflopChoice[]) =>
    JSON.stringify(history.map(({ actor, action }) => [actor, action]));
export const solutionFor = (format: TableFormat): PreflopSolution =>
    catalog.solutions.find((solution) => solution.id === format)! as PreflopSolution;

const files = import.meta.glob<PreflopNode[]>("../../../resources/gtowizard-preflop/*/*.json", {
    eager: true,
    import: "default",
});
const nodeIndex = new Map<string, PreflopNode>();
const counts = new Map<TableFormat, number>();
for (const [path, nodes] of Object.entries(files).sort(([a], [b]) => a.localeCompare(b))) {
    const [format, filename] = path.split("/").slice(-2);
    const solution = catalog.solutions.find((candidate) => candidate.id === format);
    for (const node of nodes) {
        if (
            !solution ||
            new URL(node.sourceUrl).searchParams.get("gametype") !== solution.gameType ||
            filename !== `${node.actor.toLowerCase()}.json` ||
            !solution.positions.includes(node.actor)
        )
            throw new Error(`Source node does not match its catalog directory: ${path}`);
        const key = `${format}:${choiceKey(node.history)}`;
        if (nodeIndex.has(key)) throw new Error(`Duplicate source history: ${path}`);
        nodeIndex.set(key, node);
        counts.set(format as TableFormat, (counts.get(format as TableFormat) ?? 0) + 1);
    }
}

export const nodeFor = (format: TableFormat, history: PreflopChoice[]) =>
    nodeIndex.get(`${format}:${choiceKey(history)}`);
export const spotCount = (format: TableFormat) => counts.get(format) ?? 0;
