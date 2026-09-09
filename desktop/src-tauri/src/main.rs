#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod solver_bridge;
mod solver_protocol;

use serde_json::Value;
use solver_bridge::{start_solver, stop_solver, SolverBridge};
use solver_protocol::SolverStatus;
use tauri::State;

#[tauri::command]
fn solver_status(state: State<'_, SolverBridge>) -> SolverStatus {
    state.status()
}

#[tauri::command]
async fn query_solver_node(node_id: i32, state: State<'_, SolverBridge>) -> Result<Value, String> {
    state.query_node(node_id).await
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SolverBridge::default())
        .invoke_handler(tauri::generate_handler![solver_status, query_solver_node])
        .setup(|app| {
            if let Err(error) = start_solver(app.handle()) {
                solver_bridge::fail_solver(app.handle(), error);
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while running 007 Solver");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            stop_solver(app_handle);
        }
    });
}
