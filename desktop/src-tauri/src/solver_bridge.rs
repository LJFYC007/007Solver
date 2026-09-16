use crate::solver_protocol::{
    encode_query, parse_service_message, ServiceEvent, ServiceMessage, SolverStatus,
};
use serde_json::Value;
use std::{collections::HashMap, sync::Mutex};
use tauri::{AppHandle, Manager};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};
use tokio::sync::oneshot;

type PendingResponse = oneshot::Sender<Result<Value, String>>;

struct BridgeState {
    child: Option<CommandChild>,
    generation: u64,
    next_request_id: u64,
    pending: HashMap<u64, PendingResponse>,
    status: SolverStatus,
}

pub(crate) struct SolverBridge(Mutex<BridgeState>);

impl Default for SolverBridge {
    fn default() -> Self {
        Self(Mutex::new(BridgeState {
            child: None,
            generation: 0,
            next_request_id: 1,
            pending: HashMap::new(),
            status: SolverStatus::Idle,
        }))
    }
}

impl BridgeState {
    fn fail(&mut self, message: String) {
        self.status = SolverStatus::Failed {
            message: message.clone(),
        };
        for (_, sender) in self.pending.drain() {
            let _ = sender.send(Err(message.clone()));
        }
    }

    fn stop(&mut self) {
        self.generation += 1;
        if let Some(child) = self.child.take() {
            let _ = child.kill();
        }
        for (_, sender) in self.pending.drain() {
            let _ = sender.send(Err("The selected solution changed".to_owned()));
        }
        self.status = SolverStatus::Idle;
    }
}

impl SolverBridge {
    pub(crate) fn status(&self) -> SolverStatus {
        self.0.lock().unwrap().status.clone()
    }

    pub(crate) async fn query(
        &self,
        node_id: i32,
        generation: u64,
        command: &'static str,
    ) -> Result<Value, String> {
        let receiver = {
            let mut state = self.0.lock().unwrap();
            if state.generation != generation || !matches!(state.status, SolverStatus::Ready(_)) {
                return Err("The selected solution is not ready".to_owned());
            }
            let request_id = state.next_request_id;
            state.next_request_id += 1;
            let mut request = encode_query(request_id, node_id, command)?;
            request.push(b'\n');
            let (sender, receiver) = oneshot::channel();
            state
                .child
                .as_mut()
                .ok_or("Solver service is not running")?
                .write(&request)
                .map_err(|error| error.to_string())?;
            state.pending.insert(request_id, sender);
            receiver
        };
        receiver
            .await
            .map_err(|_| "Solver service stopped before responding".to_owned())?
    }
}

fn handle_protocol_message(state: &mut BridgeState, line: &[u8]) {
    let message = match parse_service_message(line) {
        Ok(message) => message,
        Err(error) => {
            state.fail(error);
            return;
        }
    };
    match message {
        ServiceMessage::Event(event) => match event {
            ServiceEvent::BuildingTree(progress) => {
                state.status = SolverStatus::BuildingTree(progress)
            }
            ServiceEvent::Solving(progress) => state.status = SolverStatus::Solving(progress),
            ServiceEvent::Ready(solution) => state.status = SolverStatus::Ready(solution),
            ServiceEvent::Failed { message } => state.fail(message),
        },
        ServiceMessage::Response(response) => {
            if let Some(sender) = state.pending.remove(&response.request_id) {
                let result = if response.ok {
                    response
                        .node
                        .or(response.equity)
                        .ok_or_else(|| "Successful solver response has no data".to_owned())
                } else {
                    Err(response
                        .error
                        .unwrap_or_else(|| "Solver request failed".to_owned()))
                };
                let _ = sender.send(result);
            }
        }
    }
}

async fn read_solver_events(
    app: AppHandle,
    generation: u64,
    mut receiver: tauri::async_runtime::Receiver<CommandEvent>,
) {
    let mut stdout_buffer = Vec::new();
    while let Some(event) = receiver.recv().await {
        let bridge = app.state::<SolverBridge>();
        let mut state = bridge.0.lock().unwrap();
        if state.generation != generation {
            break;
        }
        match event {
            CommandEvent::Stdout(bytes) => {
                stdout_buffer.extend(bytes);
                while let Some(newline) = stdout_buffer.iter().position(|byte| *byte == b'\n') {
                    let mut line: Vec<u8> = stdout_buffer.drain(..=newline).collect();
                    while matches!(line.last(), Some(b'\n' | b'\r')) {
                        line.pop();
                    }
                    if !line.is_empty() {
                        handle_protocol_message(&mut state, &line);
                    }
                }
            }
            CommandEvent::Stderr(bytes) => eprint!("{}", String::from_utf8_lossy(&bytes)),
            CommandEvent::Error(error) => state.fail(error),
            CommandEvent::Terminated(payload) => {
                if !matches!(state.status, SolverStatus::Failed { .. }) {
                    state.fail(format!("Solver service exited unexpectedly: {payload:?}"));
                }
                state.child = None;
                break;
            }
            _ => {}
        }
    }
}

pub(crate) fn start_solver(app: &AppHandle, scenario: Value) -> Result<u64, String> {
    let mut request = serde_json::to_vec(&scenario).map_err(|error| error.to_string())?;
    request.push(b'\n');
    let bridge = app.state::<SolverBridge>();
    let mut state = bridge.0.lock().unwrap();
    state.stop();
    state.status = SolverStatus::Starting;
    let result = (|| {
        let sidecar = app
            .shell()
            .sidecar("solver-service")
            .map_err(|error| error.to_string())?
            .args(["--stdin"]);
        let (receiver, mut child) = sidecar.spawn().map_err(|error| error.to_string())?;
        if let Err(error) = child.write(&request) {
            let _ = child.kill();
            return Err(error.to_string());
        }
        state.child = Some(child);
        tauri::async_runtime::spawn(read_solver_events(app.clone(), state.generation, receiver));
        Ok(state.generation)
    })();
    if let Err(error) = &result {
        state.fail(error.clone());
    }
    result
}

pub(crate) fn stop_solver(app: &AppHandle) {
    app.state::<SolverBridge>().0.lock().unwrap().stop();
}
