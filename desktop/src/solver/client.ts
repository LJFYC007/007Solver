import { invoke } from "@tauri-apps/api/core";
import type { SolverNode, SolverStatus } from "./types";

export function getSolverStatus(): Promise<SolverStatus> {
    return invoke<SolverStatus>("solver_status");
}

export function querySolverNode(nodeId: number): Promise<SolverNode> {
    return invoke<SolverNode>("query_solver_node", { nodeId });
}
