use crate::solver_protocol::{
    encode_query, parse_service_message, QueryKind, ServiceEvent, ServiceMessage, SolverStatus,
};
use serde_json::Value;
use std::{
    collections::HashMap,
    io::{self, BufRead, BufReader, Write},
    iter, mem,
    os::windows::process::CommandExt,
    process::{Child, ChildStdout, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
};
use tokio::sync::{oneshot, Semaphore};

type PendingResponse = oneshot::Sender<Result<Value, String>>;

// Keeps the console service from opening a window.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

// One solve's service and requests.
struct Solution {
    service: Option<Service>,
    generation: u64,
    pending: HashMap<u64, PendingResponse>,
    status: SolverStatus,
}

// A running service process and the request lines for its stdin writer; see `spawn_service`.
struct Service {
    child: Child,
    requests: mpsc::Sender<Vec<u8>>,
}

struct BridgeState {
    current: Solution,
    // A ready solution that keeps answering queries while the current one solves; see `start`.
    reference: Option<Solution>,
    last_generation: u64,
    next_request_id: u64,
}

// Owns one solve at a time and optionally a reference solution: the desktop app has one bridge,
// the web server one per page session.
#[derive(Clone)]
pub(crate) struct SolverBridge(Arc<Mutex<BridgeState>>);

impl Default for SolverBridge {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(BridgeState {
            current: Solution::new(0),
            reference: None,
            last_generation: 0,
            next_request_id: 1,
        })))
    }
}

impl Solution {
    fn new(generation: u64) -> Self {
        Self {
            service: None,
            generation,
            pending: HashMap::new(),
            status: SolverStatus::Idle,
        }
    }

    fn fail(&mut self, message: String) {
        self.status = SolverStatus::Failed {
            message: message.clone(),
        };
        for (_, sender) in self.pending.drain() {
            let _ = sender.send(Err(message.clone()));
        }
    }
}

// Dropping a solution stops its service, whose output reader then finds no solution of its
// generation.
impl Drop for Solution {
    fn drop(&mut self) {
        for (_, sender) in self.pending.drain() {
            let _ = sender.send(Err("The selected solution changed".to_owned()));
        }
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

impl BridgeState {
    // Takes the current solution, leaving an idle one of a new generation.
    fn take_current(&mut self) -> Solution {
        self.last_generation += 1;
        mem::replace(&mut self.current, Solution::new(self.last_generation))
    }

    fn solution(&mut self, generation: u64) -> Option<&mut Solution> {
        if self.current.generation == generation {
            Some(&mut self.current)
        } else {
            self.reference
                .as_mut()
                .filter(|reference| reference.generation == generation)
        }
    }

    fn ready(&mut self, generation: u64) -> Option<&mut Solution> {
        self.solution(generation)
            .filter(|solution| matches!(solution.status, SolverStatus::Ready(_)))
    }
}

impl SolverBridge {
    pub(crate) fn status(&self) -> SolverStatus {
        self.0.lock().unwrap().current.status.clone()
    }

    // Replaces the current solve, keeping the ready solution of a `reference` generation
    // queryable (solver/ARCHITECTURE.md "Service and desktop lifecycle"). With a GPU queue, the
    // service starts once it holds the queue's permit and keeps it until its solve is ready or the
    // service exits.
    pub(crate) fn start(
        &self,
        scenario: &Value,
        gpu: Option<Arc<Semaphore>>,
        reference: Option<u64>,
    ) -> Result<u64, String> {
        let mut request = serde_json::to_vec(scenario).map_err(|error| error.to_string())?;
        request.push(b'\n');
        let mut state = self.0.lock().unwrap();
        if reference.is_some_and(|generation| state.ready(generation).is_none()) {
            return Err("The GTO solution is no longer available; solve again".to_owned());
        }
        let previous = state.take_current();
        if reference.is_none() {
            state.reference = None;
        } else if reference == Some(previous.generation) {
            state.reference = Some(previous);
        }
        state.current.status = SolverStatus::Starting;
        let generation = state.current.generation;
        let bridge = self.0.clone();
        thread::spawn(move || run_service(&bridge, generation, request, gpu));
        Ok(generation)
    }

    // Stops the current solve and makes the reference current again under its generation, returning
    // its status.
    pub(crate) fn restore_reference(&self) -> Result<SolverStatus, String> {
        let mut state = self.0.lock().unwrap();
        state.current = state
            .reference
            .take()
            .ok_or("There is no reference solution to restore")?;
        Ok(state.current.status.clone())
    }

    // Stops the current solve and drops the reference.
    pub(crate) fn stop(&self) {
        let mut state = self.0.lock().unwrap();
        state.reference = None;
        drop(state.take_current());
    }

    pub(crate) async fn query(
        &self,
        node_id: i32,
        generation: u64,
        kind: QueryKind,
    ) -> Result<Value, String> {
        let receiver = {
            let mut state = self.0.lock().unwrap();
            let request_id = state.next_request_id;
            state.next_request_id += 1;
            let solution = state
                .ready(generation)
                .ok_or("The selected solution is not ready")?;
            let mut request = encode_query(request_id, node_id, kind)?;
            request.push(b'\n');
            let (sender, receiver) = oneshot::channel();
            solution
                .service
                .as_ref()
                .and_then(|service| service.requests.send(request).ok())
                .ok_or("Solver service is not running")?;
            solution.pending.insert(request_id, sender);
            receiver
        };
        receiver
            .await
            .map_err(|_| "Solver service stopped before responding".to_owned())?
    }
}

fn handle_protocol_message(solution: &mut Solution, line: &[u8]) {
    let message = match parse_service_message(line) {
        Ok(message) => message,
        Err(error) => {
            solution.fail(error);
            return;
        }
    };
    match message {
        ServiceMessage::Event(event) => match event {
            ServiceEvent::BuildingTree(progress) => {
                solution.status = SolverStatus::BuildingTree(progress)
            }
            ServiceEvent::Solving(progress) => solution.status = SolverStatus::Solving(progress),
            ServiceEvent::Ready(ready) => solution.status = SolverStatus::Ready(ready),
            ServiceEvent::Failed { message } => solution.fail(message),
        },
        ServiceMessage::Response(response) => {
            if let Some(sender) = solution.pending.remove(&response.request_id) {
                let result = if response.ok {
                    let mut report = response.report.into_values();
                    match (report.next(), report.next()) {
                        (Some(value), None) => Ok(value),
                        _ => Err("Successful solver response has no single report".to_owned()),
                    }
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

// Runs one solve's service and routes its messages to the solution of its generation, current
// or reference, until the service exits or the solution is dropped.
fn run_service(
    bridge: &Mutex<BridgeState>,
    generation: u64,
    request: Vec<u8>,
    gpu: Option<Arc<Semaphore>>,
) {
    let mut permit = None;
    if let Some(gpu) = gpu {
        permit = gpu.clone().try_acquire_owned().ok();
        if permit.is_none() {
            let mut state = bridge.lock().unwrap();
            let Some(solution) = state.solution(generation) else {
                return;
            };
            solution.status = SolverStatus::Queued;
            drop(state);
            permit = tauri::async_runtime::block_on(gpu.acquire_owned()).ok();
        }
    }
    {
        let mut state = bridge.lock().unwrap();
        let Some(solution) = state.solution(generation) else {
            return;
        };
        solution.status = SolverStatus::Starting;
    }
    // Starting a process happens without the lock, which status requests and the reference's
    // queries need.
    let spawned = spawn_service(request);
    let stdout = {
        let mut state = bridge.lock().unwrap();
        // Dropping the service of a dropped solution stops it.
        let Some(solution) = state.solution(generation) else {
            return;
        };
        match spawned {
            Ok((service, stdout)) => {
                solution.service = Some(service);
                stdout
            }
            Err(error) => {
                solution.fail(error.to_string());
                return;
            }
        }
    };
    let mut reader = BufReader::new(stdout);
    let mut line = Vec::new();
    let mut service = loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line);
        let mut state = bridge.lock().unwrap();
        let Some(solution) = state.solution(generation) else {
            return;
        };
        if !matches!(read, Ok(1..)) {
            break solution.service.take().unwrap();
        }
        let message = line.trim_ascii_end();
        if !message.is_empty() {
            handle_protocol_message(solution, message);
        }
        // An exported solution no longer uses the GPU, so a reference never holds it.
        if matches!(solution.status, SolverStatus::Ready(_)) {
            permit = None;
        }
    };
    // The service closed its output, or reading it failed. Stop it if it still runs, as its stdin
    // stays open while the service is held here, and hold any permit until it has exited.
    let _ = service.child.kill();
    let exit = service
        .child
        .wait()
        .map_or_else(|error| error.to_string(), |status| status.to_string());
    let message = format!("Solver service exited unexpectedly ({exit})");
    let mut state = bridge.lock().unwrap();
    // A reference only answers queries, so it ends with its service.
    if let Some(mut reference) = state
        .reference
        .take_if(|reference| reference.generation == generation)
    {
        reference.fail(message);
    } else if state.current.generation == generation
        && !matches!(state.current.status, SolverStatus::Failed { .. })
    {
        state.current.fail(message);
    }
    drop(permit);
}

fn spawn_service(scenario: Vec<u8>) -> io::Result<(Service, ChildStdout)> {
    let program = std::env::current_exe()?.with_file_name(service_sidecar());
    let mut child = Command::new(program)
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()?;
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    thread::spawn(move || io::copy(&mut stderr, &mut io::stderr()));
    // A write waits while the service's stdin is full, and the service may be waiting for the
    // output reader, which routes each message under the bridge lock; so this thread writes,
    // never a holder of the lock. The scenario goes first and is dropped once written, not kept
    // for the service's life. A failed write means the service has exited, which its output
    // reader reports.
    let (requests, lines) = mpsc::channel();
    thread::spawn(move || {
        iter::once(scenario)
            .chain(lines)
            .try_for_each(|line| stdin.write_all(&line))
    });
    Ok((Service { child, requests }, stdout))
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
