//! Cloud GPUs, cheaply and safely (M6).
//!
//! - [`Provider`]: list offers, launch, inspect and terminate instances.
//!   Implementations: [`runpod::RunPod`], [`vast::Vast`] and [`mock::Mock`] (CI).
//! - [`catalog`]: timestamped price snapshots, so estimates use fresh prices.
//! - [`estimate`]: ETA and cost per machine, ranked by the job's objective.
//! - [`guard`]: budget meter, instance leases and the orphan reaper.
//!
//! Every instance Têmpera creates is named/labelled with [`TAG`], so the
//! reaper can find (and terminate) anything a crashed server left behind.
//! Providers are synchronous (blocking HTTP); call them from
//! `tokio::task::spawn_blocking` in async code.
//!
//! See docs/wiki/Cost-and-Estimation.md and docs/wiki/Execution-and-Machines.md.

pub mod catalog;
pub mod estimate;
pub mod guard;
pub mod mock;
pub mod runpod;
pub mod vast;

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Prefix of every instance name/label created by Têmpera.
pub const TAG: &str = "tempera";

#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    #[error("{provider}: missing API key (set {env})")]
    MissingKey { provider: String, env: String },
    #[error("{provider}: HTTP error: {message}")]
    Http { provider: String, message: String },
    #[error("{provider}: unexpected API response: {message}")]
    Api { provider: String, message: String },
    #[error("unknown provider `{0}` (known: runpod, vast, mock)")]
    UnknownProvider(String),
    #[error("{0}")]
    Refused(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CloudError>;

pub fn now_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One rentable machine configuration at a price.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    pub provider: String,
    /// Provider-specific id used by `launch` (GPU type id, ask id…).
    pub id: String,
    pub gpu_name: String,
    pub gpu_count: u32,
    /// VRAM per GPU.
    pub vram_gb: f64,
    /// vCPUs (estimated when the provider does not report them).
    pub cpu_cores: u32,
    pub ram_gb: f64,
    /// Total price per hour in USD (all GPUs).
    pub usd_per_hour: f64,
    /// Interruptible (spot/bid) capacity.
    pub spot: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Host reliability in [0, 1], when the provider reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reliability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inet_down_mbps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_gb: Option<f64>,
    /// When this price was read (unix seconds).
    pub fetched_at: u64,
}

impl Offer {
    pub fn label(&self) -> String {
        let n = if self.gpu_count > 1 {
            format!("{}x ", self.gpu_count)
        } else {
            String::new()
        };
        format!(
            "{}:{n}{}{}",
            self.provider,
            self.gpu_name,
            if self.spot { " (spot)" } else { "" }
        )
    }
}

/// What a job needs from a machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// Minimum VRAM per GPU; `None` when no GPU is needed.
    pub min_vram_gb: Option<f64>,
    pub min_ram_gb: f64,
    pub min_disk_gb: f64,
    /// Allow spot/interruptible offers.
    pub spot: bool,
    /// Hard price ceiling per hour.
    pub max_usd_per_hour: Option<f64>,
    /// Minimum host reliability (marketplaces).
    pub min_reliability: f64,
}

impl Default for ResourceRequest {
    fn default() -> Self {
        Self {
            min_vram_gb: Some(8.0),
            min_ram_gb: 8.0,
            min_disk_gb: 20.0,
            spot: true,
            max_usd_per_hour: None,
            min_reliability: 0.98,
        }
    }
}

impl ResourceRequest {
    /// Client-side filter, applied on top of whatever the provider filtered.
    pub fn accepts(&self, o: &Offer) -> bool {
        (self.spot || !o.spot)
            && self.min_vram_gb.is_none_or(|v| o.vram_gb >= v)
            && o.ram_gb >= self.min_ram_gb
            && o.disk_gb.is_none_or(|d| d >= self.min_disk_gb)
            && self.max_usd_per_hour.is_none_or(|p| o.usd_per_hour <= p)
            && o.reliability.is_none_or(|r| r >= self.min_reliability)
            && o.usd_per_hour > 0.0
    }
}

/// How to boot an instance so the server can reach it over SSH.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootSpec {
    /// Instance name; always starts with [`TAG`].
    pub name: String,
    /// Container image with sshd (and CUDA, for GPU jobs).
    pub image: String,
    /// OpenSSH public key authorized on the instance.
    pub ssh_public_key: String,
    pub disk_gb: u32,
    /// Bid for spot offers (USD/h); defaults to the offer price.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bid_usd_per_hour: Option<f64>,
}

impl BootSpec {
    pub fn for_job(job_id: &str, ssh_public_key: impl Into<String>) -> Self {
        Self {
            name: format!("{TAG}-{job_id}"),
            image: std::env::var("TEMPERA_CLOUD_IMAGE").unwrap_or_else(|_| DEFAULT_IMAGE.to_owned()),
            ssh_public_key: ssh_public_key.into(),
            disk_gb: 30,
            bid_usd_per_hour: None,
        }
    }
}

/// CUDA + Python image whose entrypoint starts sshd with `PUBLIC_KEY`.
pub const DEFAULT_IMAGE: &str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstanceState {
    Starting,
    Running,
    Stopped,
    /// Terminated or no longer known to the provider.
    Gone,
    Unknown,
}

/// A launched instance as the provider reports it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    pub provider: String,
    pub id: String,
    pub name: String,
    pub state: InstanceState,
    /// Public SSH endpoint, once known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usd_per_hour: Option<f64>,
}

impl Instance {
    pub fn is_tempera(&self) -> bool {
        self.name.starts_with(TAG)
    }

    pub fn ssh_ready(&self) -> bool {
        self.state == InstanceState::Running && self.ssh_host.is_some() && self.ssh_port.is_some()
    }
}

/// A cloud GPU provider.
pub trait Provider: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    /// Current offers matching `req` (already filtered and priced).
    fn list_offers(&self, req: &ResourceRequest) -> Result<Vec<Offer>>;
    /// Creates an instance from `offer`; returns its id.
    fn launch(&self, offer: &Offer, boot: &BootSpec) -> Result<String>;
    fn status(&self, id: &str) -> Result<Instance>;
    /// Every instance in the account (the reaper filters by [`TAG`]).
    fn instances(&self) -> Result<Vec<Instance>>;
    /// Idempotent: terminating a gone instance is not an error.
    fn terminate(&self, id: &str) -> Result<()>;
}

/// Builds a provider by name, reading its API key from the environment.
pub fn provider(name: &str) -> Result<Box<dyn Provider>> {
    match name {
        "runpod" => Ok(Box::new(runpod::RunPod::from_env()?)),
        "vast" | "vastai" | "vast.ai" => Ok(Box::new(vast::Vast::from_env()?)),
        "mock" => Ok(Box::new(mock::Mock::from_env())),
        other => Err(CloudError::UnknownProvider(other.to_owned())),
    }
}

/// Names accepted in `compute.allow` that refer to cloud providers.
pub fn is_provider_name(name: &str) -> bool {
    matches!(name, "runpod" | "vast" | "vastai" | "vast.ai" | "mock")
}

/// Shared helper: reads an API key from `env`, never logging it.
pub(crate) fn api_key(provider: &str, env: &str) -> Result<String> {
    match std::env::var(env) {
        Ok(k) if !k.trim().is_empty() => Ok(k.trim().to_owned()),
        _ => Err(CloudError::MissingKey {
            provider: provider.to_owned(),
            env: env.to_owned(),
        }),
    }
}

pub(crate) fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .http_status_as_error(false)
        .build()
        .into()
}

/// Reads a JSON body, turning non-2xx statuses into errors (without echoing keys).
pub(crate) fn read_json(
    provider: &str,
    resp: std::result::Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<serde_json::Value> {
    let mut resp = resp.map_err(|e| CloudError::Http {
        provider: provider.to_owned(),
        message: e.to_string(),
    })?;
    let status = resp.status();
    let text = resp.body_mut().read_to_string().map_err(|e| CloudError::Http {
        provider: provider.to_owned(),
        message: e.to_string(),
    })?;
    if !status.is_success() {
        let snippet: String = text.chars().take(300).collect();
        return Err(CloudError::Http {
            provider: provider.to_owned(),
            message: format!("status {status}: {snippet}"),
        });
    }
    if text.trim().is_empty() {
        return Ok(serde_json::Value::Null);
    }
    Ok(serde_json::from_str(&text)?)
}
