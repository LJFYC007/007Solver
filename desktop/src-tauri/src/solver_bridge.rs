use crate::solver_protocol::{
    encode_query_node, parse_service_message, ServiceEvent, ServiceMessage, SolverStatus,
};
use serde_json::Value;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, RwLock,
    },
};
use tauri::{AppHandle, Manager};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use tokio::sync::oneshot;

type PendingResponse = oneshot::Sender<Result<Value, String>>;

pub(crate) struct SolverBridge {
    child: Mutex<Option<CommandChild>>,
    next_request_id: AtomicU64,
    pending: Mutex<HashMap<u64, PendingResponse>>,
    status: RwLock<SolverStatus>,
}

impl Default for SolverBridge {
    fn default() -> Self {
        Self {
            child: Mutex::new(None),
            next_request_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            status: RwLock::new(SolverStatus::Starting),
        }
    }
}

impl SolverBridge {
    pub(crate) fn status(&self) -> SolverStatus {
        self.status.read().unwrap().clone()
    }

    pub(crate) async fn query_node(&self, node_id: i32) -> Result<Value, String> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let mut request = encode_query_node(request_id, node_id)?;
        request.push(b'\n');
        let (sender, receiver) = oneshot::channel();
        {
            let mut pending = self.pending.lock().unwrap();
            if !matches!(*self.status.read().unwrap(), SolverStatus::Ready { .. }) {
                return Err("Solver is not ready".to_owned());
            }
            pending.insert(request_id, sender);
        }

        let write_result = self
            .child
            .lock()
            .unwrap()
            .as_mut()
            .ok_or_else(|| "Solver service is not running".to_owned())
            .and_then(|child| child.write(&request).map_err(|error| error.to_string()));

        if let Err(error) = write_result {
            self.pending.lock().unwrap().remove(&request_id);
            return Err(error);
        }

        receiver
            .await
            .map_err(|_| "Solver service stopped before responding".to_owned())?
    }
}

fn set_status(app: &AppHandle, status: SolverStatus) {
    *app.state::<SolverBridge>().status.write().unwrap() = status;
}

pub(crate) fn fail_solver(app: &AppHandle, message: String) {
    set_status(
        app,
        SolverStatus::Failed {
            message: message.clone(),
        },
    );

    for (_, sender) in app.state::<SolverBridge>().pending.lock().unwrap().drain() {
        let _ = sender.send(Err(message.clone()));
    }
}

fn handle_protocol_message(app: &AppHandle, line: &[u8]) {
    let message = match parse_service_message(line) {
        Ok(message) => message,
        Err(error) => {
            fail_solver(app, error);
            return;
        }
    };

    match message {
        ServiceMessage::Event(event) => match event {
            ServiceEvent::BuildingTree { total_iterations } => {
                set_status(app, SolverStatus::BuildingTree { total_iterations });
            }
            ServiceEvent::Solving {
                completed_iterations,
                total_iterations,
            } => {
                set_status(
                    app,
                    SolverStatus::Solving {
                        completed_iterations,
                        total_iterations,
                    },
                );
            }
            ServiceEvent::Ready {
                iterations,
                node_count,
                root_node_id,
            } => {
                set_status(
                    app,
                    SolverStatus::Ready {
                        iterations,
                        node_count,
                        root_node_id,
                    },
                );
            }
            ServiceEvent::Failed { message } => fail_solver(app, message),
        },
        ServiceMessage::Response(response) => {
            let Some(sender) = app
                .state::<SolverBridge>()
                .pending
                .lock()
                .unwrap()
                .remove(&response.request_id)
            else {
                return;
            };

            let result = if response.ok {
                response.node.ok_or_else(|| {
                    "Invalid response from solver service: successful query has no node".to_owned()
                })
            } else {
                Err(response
                    .error
                    .unwrap_or_else(|| "Solver request failed".to_owned()))
            };
            let _ = sender.send(result);
        }
    }
}

async fn read_solver_events(
    app: AppHandle,
    mut receiver: tauri::async_runtime::Receiver<CommandEvent>,
) {
    let mut stdout_buffer = Vec::new();

    while let Some(event) = receiver.recv().await {
        match event {
            CommandEvent::Stdout(bytes) => {
                stdout_buffer.extend(bytes);
                while let Some(newline) = stdout_buffer.iter().position(|byte| *byte == b'\n') {
                    let mut line: Vec<u8> = stdout_buffer.drain(..=newline).collect();
                    while matches!(line.last(), Some(b'\n' | b'\r')) {
                        line.pop();
                    }
                    if !line.is_empty() {
                        handle_protocol_message(&app, &line);
                    }
                }
            }
            CommandEvent::Stderr(bytes) => {
                eprint!("{}", String::from_utf8_lossy(&bytes));
            }
            CommandEvent::Error(error) => {
                fail_solver(&app, error);
            }
            CommandEvent::Terminated(payload) => {
                if !matches!(
                    *app.state::<SolverBridge>().status.read().unwrap(),
                    SolverStatus::Failed { .. }
                ) {
                    fail_solver(
                        &app,
                        format!("Solver service exited unexpectedly: {payload:?}"),
                    );
                }
                break;
            }
            _ => {}
        }
    }
}

fn scenario_path(app: &AppHandle) -> Result<PathBuf, String> {
    #[cfg(debug_assertions)]
    {
        let source_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../resources/default.json");
        if source_path.is_file() {
            return Ok(source_path);
        }
    }

    app.path()
        .resource_dir()
        .map(|directory| directory.join("default.json"))
        .map_err(|error| error.to_string())
}

pub(crate) fn start_solver(app: &AppHandle) -> Result<(), String> {
    let scenario_path = scenario_path(app)?;
    let sidecar = app
        .shell()
        .sidecar("solver-service")
        .map_err(|error| error.to_string())?
        .args([scenario_path.to_string_lossy().into_owned()]);
    let (receiver, child) = sidecar.spawn().map_err(|error| error.to_string())?;

    *app.state::<SolverBridge>().child.lock().unwrap() = Some(child);
    tauri::async_runtime::spawn(read_solver_events(app.clone(), receiver));
    Ok(())
}

pub(crate) fn stop_solver(app: &AppHandle) {
    if let Some(child) = app.state::<SolverBridge>().child.lock().unwrap().take() {
        let _ = child.kill();
    }
}
