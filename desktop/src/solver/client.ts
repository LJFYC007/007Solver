import { invoke, isTauri } from "@tauri-apps/api/core";
import type { EquityReport, SolverNode, SolverStatus } from "./types";
import type { PostflopScenario } from "./preflop";

// The desktop app calls its bridge in process; browsers reach `007solver --serve`, which keeps
// one solve per page session.
export const isDesktopApp = isTauri();

const session = Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
    byte.toString(16).padStart(2, "0"),
).join("");
const sessionUrl = `/api/sessions/${session}`;
// Closing or reloading the page releases its solve at once. A page kept in the back/forward cache
// keeps its session, so going back resumes it until the server releases it.
if (!isDesktopApp)
    window.addEventListener("pagehide", (event) => {
        if (!event.persisted) navigator.sendBeacon(`${sessionUrl}/close`);
    });

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(sessionUrl + path, init);
    // Like invoke, reject with the server's message; proxies answer errors with HTML pages.
    if (!response.ok)
        throw response.headers.get("Content-Type")?.startsWith("text/plain")
            ? await response.text()
            : `The server answered ${response.status}`;
    return response.status === 204 ? (undefined as T) : response.json();
}

// HTTP requests can reach the server out of order, so a solve or cancel waits for the previous one.
let lastControl: Promise<unknown> = Promise.resolve();
function sendControl<T>(path: string, init: RequestInit): Promise<T> {
    const result = lastControl.then(() => request<T>(path, init));
    lastControl = result.catch(() => undefined);
    return result;
}

export function solveScenario(scenario: PostflopScenario): Promise<number> {
    return isDesktopApp
        ? invoke<number>("solve_scenario", { scenario })
        : sendControl<number>("/solve", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(scenario),
          });
}

export function cancelSolver(): Promise<void> {
    return isDesktopApp ? invoke<void>("cancel_solver") : sendControl<void>("/cancel", { method: "POST" });
}

export function getSolverStatus(): Promise<SolverStatus> {
    return isDesktopApp ? invoke<SolverStatus>("solver_status") : request<SolverStatus>("/status");
}

export function querySolverNode(nodeId: number, generation: number): Promise<SolverNode> {
    return isDesktopApp
        ? invoke<SolverNode>("query_solver_node", { nodeId, generation })
        : request<SolverNode>(`/nodes/${nodeId}?generation=${generation}`);
}

export function querySolverEquity(nodeId: number, generation: number): Promise<EquityReport> {
    return isDesktopApp
        ? invoke<EquityReport>("query_solver_equity", { nodeId, generation })
        : request<EquityReport>(`/equity/${nodeId}?generation=${generation}`);
}
