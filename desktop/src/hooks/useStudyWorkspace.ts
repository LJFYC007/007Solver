import { useMemo, useRef, useState } from "react";
import { isForcedRunout, type ChanceNode, type SolverNode, type SolverStatus } from "../solver";
import { choiceKey, solutionFor, type PreflopChoice, type TableFormat } from "../solver/catalog";
import { postflopScenario, preflopOutcome, replayPreflop, type PostflopScenario } from "../solver/preflop";
import { buildStudyView } from "../solver/study";
import { useSolverNavigation } from "./useSolverNavigation";

export interface StudyWorkspaceProps {
    root?: SolverNode;
    status: SolverStatus;
    generation?: number;
    scenario?: PostflopScenario;
    changing: boolean;
    onSolve: (scenario: PostflopScenario) => Promise<void>;
    onInvalidate: () => Promise<void>;
}

export function useStudyWorkspace({
    root,
    status,
    generation,
    scenario,
    changing,
    onSolve,
    onInvalidate,
}: StudyWorkspaceProps) {
    const [format, setFormat] = useState<TableFormat>("6max");
    const [history, setHistory] = useState<PreflopChoice[]>([]);
    const [preIndex, setPreIndex] = useState<number | null>(0);
    const [board, setBoard] = useState<string[]>([]);
    const [iterations, setIterations] = useState(3000);
    const [accuracyPercent, setAccuracyPercent] = useState(0.01);
    const [rangeSeat, setRangeSeat] = useState<string>();
    const [error, setError] = useState<string>();
    const [picker, setPicker] = useState<{ kind: "flop" } | { kind: "runout"; node: ChanceNode; index: number }>();
    const editing = useRef(false);
    const navigation = useSolverNavigation(root, generation);
    const solveState = status.state;
    const solveError = status.state === "failed" ? status.message : "";
    const view = useMemo(
        () =>
            buildStudyView({
                format,
                history,
                preIndex,
                board,
                rangeSeat,
                path: navigation.path,
                activeIndex: navigation.activeIndex,
                scenario,
                solveState,
                solveError,
            }),
        [
            format,
            history,
            preIndex,
            board,
            rangeSeat,
            navigation.path,
            navigation.activeIndex,
            scenario,
            solveState,
            solveError,
        ],
    );
    function viewPre(index: number) {
        if (editing.current) return;
        navigation.suspend();
        setPreIndex(index);
        setPicker(undefined);
        setError(undefined);
    }
    function openFlop() {
        if (editing.current || view.busy) return;
        navigation.suspend();
        setPicker({ kind: "flop" });
    }
    async function choose(prefix: PreflopChoice[], actor: string, action: string) {
        if (editing.current) return;
        const next = [...prefix, { actor, action }];
        if (choiceKey(history.slice(0, next.length)) === choiceKey(next)) {
            viewPre(next.length);
            if (next.length === history.length && view.preflop.canPlayPostflop && board.length === 0) openFlop();
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
            setHistory(next);
            setPreIndex(next.length);
            setBoard([]);
            setRangeSeat(undefined);
            setPicker(
                preflopOutcome(nextState, solutionFor(format).stack) === "headsUp" ? { kind: "flop" } : undefined,
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
        if (editing.current || view.busy) return;
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

    function viewPreTimeline(index: number) {
        const prefix = view.preflop.timeline[index].history;
        if (prefix.length > history.length) void changeLine(prefix);
        else viewPre(index);
    }
    function selectSeat(position: string) {
        if (editing.current) return;
        if (!view.showPostflop && !view.preflop.node) setRangeSeat(position);
        else if (view.showPostflop) {
            const index = navigation.path
                .map((entry, index) => ({ entry, index }))
                .slice(0, navigation.activeIndex + 1)
                .reverse()
                .find(({ entry }) => entry.kind === "decision" && view.players[entry.actor] === position)?.index;
            if (index !== undefined) viewPost(index);
        } else {
            const index = view.preflop.timeline
                .map((entry, index) => ({ entry, index }))
                .filter(({ entry }) => entry.history.length <= history.length)
                .reverse()
                .find(({ entry }) => entry.node.actor === position)?.index;
            if (index !== undefined) viewPre(index);
        }
    }
    function confirmBoard(cards: string[]) {
        if (!picker || editing.current) return;
        if (picker.kind === "runout") {
            const outcome = picker.node.outcomes.find((o) => o.card === cards[cards.length - 1]);
            if (outcome) void actPost(picker.index, outcome.nextNodeId);
        } else if (cards.join() !== board.join()) {
            void changeStudy(() => {
                setBoard(cards);
                setPreIndex(history.length);
            });
        }
        setPicker(undefined);
    }
    function cancel() {
        void changeStudy(() => setPreIndex(history.length));
    }
    function closePicker() {
        setPicker(undefined);
    }
    const actionCards = view.strategy.actions.map((action) => {
        const nodeId = action.nextNodeId;
        const preNode = view.preflop.node;
        return {
            ...action,
            onSelect: changing
                ? undefined
                : view.showPostflop
                  ? nodeId !== undefined && !navigation.navigationPending
                      ? () => void actPost(navigation.activeIndex, nodeId)
                      : undefined
                  : preNode
                    ? () => void choose(view.preflop.shownHistory, preNode.actor, action.label)
                    : undefined,
        };
    });
    const boardAction =
        changing || view.busy
            ? undefined
            : view.boardTarget === "runout"
              ? () => pickRunout(navigation.activeIndex)
              : view.boardTarget === "flop"
                ? openFlop
                : undefined;
    return {
        format,
        history,
        preIndex,
        board,
        iterations,
        setIterations,
        accuracyPercent,
        setAccuracyPercent,
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
    };
}
