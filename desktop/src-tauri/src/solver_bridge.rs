use crate::solver_protocol::{
    encode_query, parse_service_message, ServiceEvent, ServiceMessage, SolverStatus,
};
use serde_json::Value;
use std::{
    collections::HashMap,
    io::{self, BufRead, BufReader, Write},
    os::windows::process::CommandExt,
    process::{Child, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};
use tokio::sync::{oneshot, Semaphore};

type PendingResponse = oneshot::Sender<Result<Value, String>>;

// Keeps the console service from opening a window.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
// Keeps unanswered requests well inside the service's stdin pipe buffer, so a query never
// blocks writing while it holds the lock the output reader needs.
const MAX_PENDING_QUERIES: usize = 32;

struct BridgeState {
    child: Option<Child>,
    generation: u64,
    next_request_id: u64,
    pending: HashMap<u64, PendingResponse>,
    status: SolverStatus,
}

// Owns one solve at a time: the desktop app has one bridge, the web server one per page session.
#[derive(Clone)]
pub(crate) struct SolverBridge(Arc<Mutex<BridgeState>>);

impl Default for SolverBridge {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(BridgeState {
            child: None,
            generation: 0,
            next_request_id: 1,
            pending: HashMap::new(),
            status: SolverStatus::Idle,
        })))
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
        if let Some(mut child) = self.child.take() {
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

    // Replaces the current solve. With a GPU queue, the service starts once it holds the
    // queue's permit and keeps it until its solve is ready or the service exits.
    pub(crate) fn start(
        &self,
        scenario: &Value,
        gpu: Option<Arc<Semaphore>>,
    ) -> Result<u64, String> {
        let mut request = serde_json::to_vec(scenario).map_err(|error| error.to_string())?;
        request.push(b'\n');
        let mut state = self.0.lock().unwrap();
        state.stop();
        state.status = SolverStatus::Starting;
        let generation = state.generation;
        let bridge = self.0.clone();
        thread::spawn(move || run_service(&bridge, generation, &request, gpu));
        Ok(generation)
    }

    pub(crate) fn stop(&self) {
        self.0.lock().unwrap().stop();
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
            if state.pending.len() >= MAX_PENDING_QUERIES {
                return Err("Too many solver requests are in progress".to_owned());
            }
            let request_id = state.next_request_id;
            state.next_request_id += 1;
            let mut request = encode_query(request_id, node_id, command)?;
            request.push(b'\n');
            let (sender, receiver) = oneshot::channel();
            state
                .child
                .as_mut()
                .and_then(|child| child.stdin.as_mut())
                .ok_or("Solver service is not running")?
                .write_all(&request)
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

// Runs one solve's service and routes its messages until it exits or the solve is replaced.
fn run_service(
    bridge: &Mutex<BridgeState>,
    generation: u64,
    request: &[u8],
    gpu: Option<Arc<Semaphore>>,
) {
    let mut permit = None;
    if let Some(gpu) = gpu {
        permit = gpu.clone().try_acquire_owned().ok();
        if permit.is_none() {
            let mut state = bridge.lock().unwrap();
            if state.generation != generation {
                return;
            }
            state.status = SolverStatus::Queued;
            drop(state);
            permit = tauri::async_runtime::block_on(gpu.acquire_owned()).ok();
        }
    }
    let stdout = {
        let mut state = bridge.lock().unwrap();
        if state.generation != generation {
            return;
        }
        state.status = SolverStatus::Starting;
        match spawn_service(request) {
            Ok((child, stdout)) => {
                state.child = Some(child);
                stdout
            }
            Err(error) => {
                state.fail(error.to_string());
                return;
            }
        }
    };
    let mut reader = BufReader::new(stdout);
    let mut line = Vec::new();
    let child = loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line);
        let mut state = bridge.lock().unwrap();
        if state.generation != generation {
            return;
        }
        if !matches!(read, Ok(1..)) {
            break state.child.take();
        }
        let message = line.trim_ascii_end();
        if !message.is_empty() {
            handle_protocol_message(&mut state, message);
        }
        // An exported solution no longer uses the GPU.
        if matches!(state.status, SolverStatus::Ready(_)) {
            permit = None;
        }
    };
    // The service closed its output; hold any permit until it has exited.
    let Some(mut child) = child else { return };
    let exit = child
        .wait()
        .map_or_else(|error| error.to_string(), |status| status.to_string());
    let mut state = bridge.lock().unwrap();
    if state.generation == generation && !matches!(state.status, SolverStatus::Failed { .. }) {
        state.fail(format!("Solver service exited unexpectedly ({exit})"));
    }
    drop(permit);
}

fn spawn_service(request: &[u8]) -> io::Result<(Child, ChildStdout)> {
    let program = std::env::current_exe()?.with_file_name(service_sidecar());
    let mut child = Command::new(program)
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()?;
    let stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    thread::spawn(move || io::copy(&mut stderr, &mut io::stderr()));
    if let Err(error) = child.stdin.as_mut().unwrap().write_all(request) {
        let _ = child.kill();
        return Err(error);
    }
    Ok((child, stdout))
}

// The package adds an SSE2 service for CPUs that cannot run the /arch:AVX2 build,
// which may also use FMA and BMI instructions.
fn service_sidecar() -> &'static str {
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
    {
        "solver-service.exe"
    } else {
        "solver-service-sse2.exe"
    }
}
