#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod server;
mod solver_bridge;
mod solver_protocol;

use serde_json::Value;
use solver_bridge::SolverBridge;
use solver_protocol::SolverStatus;
use tauri::{Manager, State};

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
async fn query_solver_equity(
    node_id: i32,
    generation: u64,
    state: State<'_, SolverBridge>,
) -> Result<Value, String> {
    state.query(node_id, generation, "query_equity").await
}

#[tauri::command]
fn solve_scenario(scenario: Value, state: State<'_, SolverBridge>) -> Result<u64, String> {
    state.start(&scenario, None)
}

#[tauri::command]
fn cancel_solver(state: State<'_, SolverBridge>) {
    state.stop();
}

fn main() {
    let context = tauri::generate_context!();
    // `--serve[=ADDRESS]` serves the interface to browsers instead of opening the window.
    let argument = std::env::args().nth(1);
    let address = match argument.as_deref() {
        Some("--serve") => Some(server::DEFAULT_ADDRESS),
        Some(argument) => argument.strip_prefix("--serve="),
        None => None,
    };
    if let Some(address) = address {
        if let Err(error) = tauri::async_runtime::block_on(server::serve(context, address)) {
            eprintln!("Cannot serve on {address}: {error}");
            std::process::exit(1);
        }
        return;
    }

    let app = tauri::Builder::default()
        .manage(SolverBridge::default())
        .invoke_handler(tauri::generate_handler![
            solver_status,
            query_solver_node,
            query_solver_equity,
            solve_scenario,
            cancel_solver
        ])
        .build(context)
        .expect("error while running 007 Solver");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            app_handle.state::<SolverBridge>().stop();
        }
    });
}
