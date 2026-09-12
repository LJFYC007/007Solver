use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "state")]
pub(crate) enum SolverStatus {
    Idle,
    Starting,
    BuildingTree {
        #[serde(rename = "totalIterations")]
        total_iterations: i32,
    },
    Solving {
        #[serde(rename = "completedIterations")]
        completed_iterations: i32,
        #[serde(rename = "totalIterations")]
        total_iterations: i32,
    },
    Ready {
        iterations: i32,
        #[serde(rename = "nodeCount")]
        node_count: i32,
        #[serde(rename = "rootNodeId")]
        root_node_id: i32,
    },
    Failed {
        message: String,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", tag = "event")]
pub(crate) enum ServiceEvent {
    BuildingTree {
        #[serde(rename = "totalIterations")]
        total_iterations: i32,
    },
    Solving {
        #[serde(rename = "completedIterations")]
        completed_iterations: i32,
        #[serde(rename = "totalIterations")]
        total_iterations: i32,
    },
    Ready {
        iterations: i32,
        #[serde(rename = "nodeCount")]
        node_count: i32,
        #[serde(rename = "rootNodeId")]
        root_node_id: i32,
    },
    Failed {
        message: String,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ServiceResponse {
    pub request_id: u64,
    pub ok: bool,
    pub node: Option<Value>,
    pub equity: Option<Value>,
    pub error: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum ServiceMessage {
    Event(ServiceEvent),
    Response(ServiceResponse),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct QueryNodeRequest {
    request_id: u64,
    command: &'static str,
    node_id: i32,
}

pub(crate) fn parse_service_message(line: &[u8]) -> Result<ServiceMessage, String> {
    serde_json::from_slice(line)
        .map_err(|error| format!("Invalid response from solver service: {error}"))
}

pub(crate) fn encode_query(
    request_id: u64,
    node_id: i32,
    command: &'static str,
) -> Result<Vec<u8>, String> {
    serde_json::to_vec(&QueryNodeRequest {
        request_id,
        command,
        node_id,
    })
    .map_err(|error| error.to_string())
}
