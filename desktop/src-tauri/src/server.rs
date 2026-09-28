use crate::{
    solver_bridge::SolverBridge,
    solver_protocol::{QueryKind, SolverStatus},
};
use axum::{
    extract::{Path, Query, Request, State},
    http::{header, StatusCode, Uri},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    net::IpAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tauri::utils::{assets::AssetKey, mime_type::MimeType};
use tokio::sync::Semaphore;

// Loopback only: the server has no authentication of its own.
pub(crate) const DEFAULT_ADDRESS: &str = "127.0.0.1:8007";
// A page's session is released after this long without requests; closing the page releases
// it at once.
const IDLE_RELEASE: Duration = Duration::from_secs(10 * 60);
// Ready solutions stay in host memory, about 4 bytes per strategy entry plus the last evaluated
// decision's traversal tables; beyond this many, the least recently used are released.
const MAX_READY_SOLUTIONS: usize = 4;

struct Session {
    bridge: SolverBridge,
    used: Instant,
    // The page has closed. The session stays until the idle release, so that a solve request
    // the network delivers after the close cannot start a solve nobody will read.
    closed: bool,
}

impl Session {
    fn new() -> Self {
        Self {
            bridge: SolverBridge::default(),
            used: Instant::now(),
            closed: false,
        }
    }
}

struct Server {
    sessions: Mutex<HashMap<String, Session>>,
    // One solve trains at a time, in request order.
    gpu: Arc<Semaphore>,
    // The page, which only release builds embed.
    assets: Box<dyn tauri::Assets<tauri::Wry>>,
}

type Shared = State<Arc<Server>>;

impl Server {
    fn session(&self, id: &str) -> Option<SolverBridge> {
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.get_mut(id).filter(|session| !session.closed)?;
        session.used = Instant::now();
        Some(session.bridge.clone())
    }

    fn release(&self) {
        let mut sessions = self.sessions.lock().unwrap();
        for (_, session) in sessions.extract_if(|_, session| session.used.elapsed() >= IDLE_RELEASE)
        {
            session.bridge.stop();
        }
        let mut ready: Vec<_> = sessions
            .iter()
            .filter(|(_, session)| matches!(session.bridge.status(), SolverStatus::Ready(_)))
            .map(|(id, session)| (session.used, id.clone()))
            .collect();
        ready.sort_unstable();
        for (_, id) in ready.iter().rev().skip(MAX_READY_SOLUTIONS) {
            if let Some(session) = sessions.remove(id) {
                session.bridge.stop();
            }
        }
    }
}

pub(crate) async fn serve(
    context: tauri::Context<tauri::Wry>,
    address: &str,
) -> std::io::Result<()> {
    let server = Arc::new(Server {
        sessions: Mutex::default(),
        gpu: Arc::new(Semaphore::new(1)),
        assets: context.assets,
    });
    let sweeper = server.clone();
    tauri::async_runtime::spawn(async move {
        let mut timer = tokio::time::interval(Duration::from_secs(5));
        loop {
            timer.tick().await;
            sweeper.release();
        }
    });
    let app = Router::new()
        .route("/api/sessions/{id}/solve", post(solve))
        .route("/api/sessions/{id}/cancel", post(cancel))
        .route("/api/sessions/{id}/close", post(close))
        .route("/api/sessions/{id}/status", get(status))
        .route("/api/sessions/{id}/reports/{kind}/{node}", get(query))
        .fallback(asset)
        .layer(middleware::from_fn(require_address_host))
        .with_state(server);
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!("Serving 007 Solver on http://{address}");
    axum::serve(listener, app).await
}

// DNS rebinding points another site's name at this server, but it cannot present an IP
// address as that name, so proxies must forward requests with an address Host.
async fn require_address_host(request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|host| host.to_str().ok())
        .unwrap_or_default();
    let name = match host.strip_prefix('[') {
        Some(bracketed) => bracketed.split(']').next(),
        None => host.split(':').next(),
    }
    .unwrap_or_default();
    if name == "localhost" || name.parse::<IpAddr>().is_ok() {
        next.run(request).await
    } else {
        (
            StatusCode::FORBIDDEN,
            "The server answers only requests addressed to an IP address or localhost",
        )
            .into_response()
    }
}

async fn solve(
    State(server): Shared,
    Path(id): Path<String>,
    Json(scenario): Json<Value>,
) -> Response {
    let mut sessions = server.sessions.lock().unwrap();
    let session = sessions.entry(id).or_insert_with(Session::new);
    if session.closed {
        return released();
    }
    session.used = Instant::now();
    // Starting under the sessions lock keeps a concurrent close from missing this solve.
    reply(session.bridge.start(&scenario, Some(server.gpu.clone())))
}

// Like the desktop app, cancelling keeps the session and its generation count.
async fn cancel(State(server): Shared, Path(id): Path<String>) -> StatusCode {
    if let Some(bridge) = server.session(&id) {
        bridge.stop();
    }
    StatusCode::NO_CONTENT
}

async fn close(State(server): Shared, Path(id): Path<String>) -> StatusCode {
    let mut sessions = server.sessions.lock().unwrap();
    let session = sessions.entry(id).or_insert_with(Session::new);
    session.closed = true;
    session.used = Instant::now();
    session.bridge.stop();
    StatusCode::NO_CONTENT
}

async fn status(State(server): Shared, Path(id): Path<String>) -> Response {
    match server.session(&id) {
        Some(bridge) => Json(bridge.status()).into_response(),
        None => released(),
    }
}

#[derive(Deserialize)]
struct Solution {
    generation: u64,
}

async fn query(
    State(server): Shared,
    Path((id, kind, node)): Path<(String, QueryKind, i32)>,
    Query(solution): Query<Solution>,
) -> Response {
    let Some(bridge) = server.session(&id) else {
        return released();
    };
    reply(bridge.query(node, solution.generation, kind).await)
}

async fn asset(State(server): Shared, uri: Uri) -> Response {
    let path = match uri.path() {
        "/" => "/index.html",
        path => path,
    };
    let Some(data) = server.assets.get(&AssetKey::from(path)) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    // Vite names the files under /assets/ by content hash, so only index.html changes per build.
    let cache = if path.starts_with("/assets/") {
        "max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    (
        [
            (header::CONTENT_TYPE, MimeType::parse(&data, path)),
            (header::CACHE_CONTROL, cache.to_owned()),
        ],
        data.into_owned(),
    )
        .into_response()
}

fn reply<T: Serialize>(result: Result<T, String>) -> Response {
    match result {
        Ok(value) => Json(value).into_response(),
        Err(message) => (StatusCode::CONFLICT, message).into_response(),
    }
}

fn released() -> Response {
    (
        StatusCode::NOT_FOUND,
        "The server released this solution; solve again",
    )
        .into_response()
}
