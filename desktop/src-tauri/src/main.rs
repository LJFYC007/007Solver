#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod solver_bridge;
mod solver_protocol;

use serde_json::Value;
use solver_bridge::{start_solver, stop_solver, SolverBridge};
use solver_protocol::SolverStatus;
use tauri::{AppHandle, State};

#[tauri::command]
fn solver_status(state: State<'_, SolverBridge>) -> SolverStatus {
    state.status()
}

#[tauri::command]
async fn query_solver_node(
    node_id: i32,
    generation: u64,
    state: State<'_, SolverBridge>,
) -> Result<Value, String> {
    state.query(node_id, generation, "query_node").await
}

#[tauri::command]
async fn query_solver_node_evs(
    node_id: i32,
    generation: u64,
    state: State<'_, SolverBridge>,
) -> Result<Value, String> {
    state.query(node_id, generation, "query_node_evs").await
}

#[tauri::command]
async fn query_solver_equity(
    node_id: i32,
    generation: u64,
    state: State<'_, SolverBridge>,
) -> Result<Value, String> {
    state.query(node_id, generation, "query_equity").await
}

#[tauri::command]
fn solve_scenario(app: AppHandle, scenario: Value) -> Result<u64, String> {
    start_solver(&app, scenario)
}

#[tauri::command]
fn cancel_solver(app: AppHandle) {
    stop_solver(&app);
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SolverBridge::default())
        .invoke_handler(tauri::generate_handler![
            solver_status,
            query_solver_node,
            query_solver_node_evs,
            query_solver_equity,
            solve_scenario,
            cancel_solver
        ])
        .build(tauri::generate_context!())
        .expect("error while running 007 Solver");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            stop_solver(app_handle);
        }
    });
}
