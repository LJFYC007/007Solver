import { invoke } from "@tauri-apps/api/core";
import type { EquityReport, SolverNode, SolverStatus } from "./types";
import type { PostflopScenario } from "./preflop";

export function solveScenario(scenario: PostflopScenario): Promise<number> {
    return invoke<number>("solve_scenario", { scenario });
}

export function cancelSolver(): Promise<void> {
    return invoke<void>("cancel_solver");
}

export function getSolverStatus(): Promise<SolverStatus> {
    return invoke<SolverStatus>("solver_status");
}

export function querySolverNode(nodeId: number, generation: number): Promise<SolverNode> {
    return invoke<SolverNode>("query_solver_node", { nodeId, generation });
}

export function querySolverEquity(nodeId: number, generation: number): Promise<EquityReport> {
    return invoke<EquityReport>("query_solver_equity", { nodeId, generation });
}

export function querySolverNodeEvs(nodeId: number, generation: number): Promise<SolverNode> {
    return invoke<SolverNode>("query_solver_node_evs", { nodeId, generation });
}
