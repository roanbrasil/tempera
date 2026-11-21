//! RunPod: prices from the GraphQL `gpuTypes` query, pods through the REST
//! API (`https://rest.runpod.io/v1`). Auth: `Authorization: Bearer $RUNPOD_API_KEY`.
//!
//! Built from RunPod's public API documentation; not yet exercised against a
//! live account (see docs/wiki/Execution-and-Machines.md).

use serde_json::{Value, json};

use crate::{
    BootSpec, CloudError, Instance, InstanceState, Offer, Provider, ResourceRequest, Result, api_key,
    http_agent, now_s, read_json,
};

const NAME: &str = "runpod";
pub const KEY_ENV: &str = "RUNPOD_API_KEY";
const GRAPHQL: &str = "https://api.runpod.io/graphql";
const REST: &str = "https://rest.runpod.io/v1";
/// RunPod does not report vCPUs per GPU type; community pods get about this many.
const CPU_PER_GPU: u32 = 8;
const RAM_PER_GPU_GB: f64 = 30.0;

const GPU_TYPES_QUERY: &str = "query GpuTypes { gpuTypes { id displayName memoryInGb \
    securePrice communityPrice secureSpotPrice communitySpotPrice } }";

#[derive(Debug)]
pub struct RunPod {
    key: String,
    graphql: String,
    rest: String,
}

impl RunPod {
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(api_key(NAME, KEY_ENV)?))
    }

    pub fn new(key: String) -> Self {
        Self {
            key,
            graphql: GRAPHQL.to_owned(),
            rest: REST.to_owned(),
        }
    }

    /// Points the client at other endpoints (tests, proxies).
    pub fn with_endpoints(mut self, graphql: &str, rest: &str) -> Self {
        graphql.clone_into(&mut self.graphql);
        rest.clone_into(&mut self.rest);
        self
    }

    fn auth(&self) -> String {
        format!("Bearer {}", self.key)
    }
}

fn api(message: impl Into<String>) -> CloudError {
    CloudError::Api {
        provider: NAME.into(),
        message: message.into(),
    }
}

/// `gpuTypes` response → one offer per (GPU type × secure/community × on-demand/spot) with a price.
pub fn parse_gpu_types(v: &Value, fetched_at: u64) -> Result<Vec<Offer>> {
    let types = v["data"]["gpuTypes"]
        .as_array()
        .ok_or_else(|| api("missing data.gpuTypes"))?;
    let mut out = Vec::new();
    for t in types {
        let Some(id) = t["id"].as_str() else { continue };
        let name = t["displayName"].as_str().unwrap_or(id);
        let vram = t["memoryInGb"].as_f64().unwrap_or(0.0);
        for (field, cloud, spot) in [
            ("communityPrice", "COMMUNITY", false),
            ("communitySpotPrice", "COMMUNITY", true),
            ("securePrice", "SECURE", false),
            ("secureSpotPrice", "SECURE", true),
        ] {
            let Some(price) = t[field].as_f64().filter(|p| *p > 0.0) else {
                continue;
            };
            out.push(Offer {
                provider: NAME.into(),
                // `<gpu type id>|<cloud type>` is what `launch` needs.
                id: format!("{id}|{cloud}"),
                gpu_name: name.to_owned(),
                gpu_count: 1,
                vram_gb: vram,
                cpu_cores: CPU_PER_GPU,
                ram_gb: RAM_PER_GPU_GB,
                usd_per_hour: price,
                spot,
                region: Some(cloud.to_lowercase()),
                reliability: None,
                inet_down_mbps: None,
                disk_gb: None,
                fetched_at,
            });
        }
    }
    Ok(out)
}

/// Body of `POST /pods`.
pub fn pod_request(offer: &Offer, boot: &BootSpec) -> Result<Value> {
    let (gpu, cloud) = offer
        .id
        .split_once('|')
        .ok_or_else(|| api(format!("offer id `{}` is not `<gpu>|<cloud>`", offer.id)))?;
    let mut body = json!({
        "name": boot.name,
        "imageName": boot.image,
        "gpuTypeIds": [gpu],
        "gpuCount": offer.gpu_count.max(1),
        "cloudType": cloud,
        "interruptible": offer.spot,
        "containerDiskInGb": boot.disk_gb,
        "ports": ["22/tcp"],
        "supportPublicIp": true,
        "env": { "PUBLIC_KEY": boot.ssh_public_key },
    });
    if offer.spot {
        body["bidPerGpu"] =
            json!(boot.bid_usd_per_hour.unwrap_or(offer.usd_per_hour) / f64::from(offer.gpu_count.max(1)));
    }
    Ok(body)
}

/// A pod object from `GET /pods[/{id}]`.
pub fn parse_pod(v: &Value) -> Option<Instance> {
    let id = v["id"].as_str()?.to_owned();
    let state = match v["desiredStatus"].as_str().unwrap_or("") {
        "RUNNING" => InstanceState::Running,
        "CREATED" | "RESTARTING" => InstanceState::Starting,
        "EXITED" => InstanceState::Stopped,
        "TERMINATED" => InstanceState::Gone,
        _ => InstanceState::Unknown,
    };
    let ssh_port = v["portMappings"]["22"]
        .as_u64()
        .and_then(|p| u16::try_from(p).ok());
    let ssh_host = v["publicIp"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    Some(Instance {
        provider: NAME.into(),
        id,
        name: v["name"].as_str().unwrap_or_default().to_owned(),
        // Running but without a mapped port yet → still booting from our point of view.
        state: if state == InstanceState::Running && (ssh_host.is_none() || ssh_port.is_none()) {
            InstanceState::Starting
        } else {
            state
        },
        ssh_host,
        ssh_port,
        ssh_user: Some("root".into()),
        usd_per_hour: v["costPerHr"].as_f64().or(v["adjustedCostPerHr"].as_f64()),
    })
}

impl Provider for RunPod {
    fn name(&self) -> &str {
        NAME
    }

    fn list_offers(&self, req: &ResourceRequest) -> Result<Vec<Offer>> {
        let resp = http_agent()
            .post(&self.graphql)
            .header("Authorization", &self.auth())
            .send_json(json!({ "query": GPU_TYPES_QUERY }));
        let v = read_json(NAME, resp)?;
        if let Some(err) = v["errors"].as_array().and_then(|e| e.first()) {
            return Err(api(err["message"].as_str().unwrap_or("GraphQL error").to_owned()));
        }
        let mut offers: Vec<Offer> = parse_gpu_types(&v, now_s())?
            .into_iter()
            .filter(|o| req.accepts(o))
            .collect();
        offers.sort_by(|a, b| a.usd_per_hour.total_cmp(&b.usd_per_hour));
        Ok(offers)
    }

    fn launch(&self, offer: &Offer, boot: &BootSpec) -> Result<String> {
        let body = pod_request(offer, boot)?;
        let resp = http_agent()
            .post(&format!("{}/pods", self.rest))
            .header("Authorization", &self.auth())
            .send_json(body);
        let v = read_json(NAME, resp)?;
        v["id"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| api("create pod: no id in response"))
    }

    fn status(&self, id: &str) -> Result<Instance> {
        let resp = http_agent()
            .get(&format!("{}/pods/{id}", self.rest))
            .header("Authorization", &self.auth())
            .call();
        match read_json(NAME, resp) {
            Ok(v) => parse_pod(&v).ok_or_else(|| api("pod without id")),
            Err(CloudError::Http { message, .. }) if message.starts_with("status 404") => Ok(Instance {
                provider: NAME.into(),
                id: id.to_owned(),
                name: String::new(),
                state: InstanceState::Gone,
                ssh_host: None,
                ssh_port: None,
                ssh_user: None,
                usd_per_hour: None,
            }),
            Err(e) => Err(e),
        }
    }

    fn instances(&self) -> Result<Vec<Instance>> {
        let resp = http_agent()
            .get(&format!("{}/pods", self.rest))
            .header("Authorization", &self.auth())
            .call();
        let v = read_json(NAME, resp)?;
        let list = v
            .as_array()
            .or_else(|| v["pods"].as_array())
            .cloned()
            .unwrap_or_default();
        Ok(list.iter().filter_map(parse_pod).collect())
    }

    fn terminate(&self, id: &str) -> Result<()> {
        let resp = http_agent()
            .delete(&format!("{}/pods/{id}", self.rest))
            .header("Authorization", &self.auth())
            .call();
        match read_json(NAME, resp) {
            Ok(_) => Ok(()),
            Err(CloudError::Http { message, .. }) if message.starts_with("status 404") => Ok(()),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn fixture(name: &str) -> Value {
        let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    }

    #[test]
    fn gpu_types_become_priced_offers() {
        let offers = parse_gpu_types(&fixture("runpod_gpu_types.json"), 42).unwrap();
        // 4090: community, community spot, secure; A100: secure, secure spot; null prices skipped.
        assert_eq!(offers.len(), 5);
        let spot = offers
            .iter()
            .find(|o| o.spot && o.gpu_name == "RTX 4090")
            .unwrap();
        assert_eq!(spot.id, "NVIDIA GeForce RTX 4090|COMMUNITY");
        assert!((spot.usd_per_hour - 0.2).abs() < 1e-9);
        assert_eq!(spot.vram_gb, 24.0);
        assert!(offers.iter().all(|o| o.fetched_at == 42));
    }

    #[test]
    fn pod_request_has_ssh_and_tag() {
        let offers = parse_gpu_types(&fixture("runpod_gpu_types.json"), 0).unwrap();
        let o = offers.iter().find(|o| o.spot).unwrap();
        let body = pod_request(o, &BootSpec::for_job("0000beef", "job-1", "ssh-ed25519 AAA test")).unwrap();
        assert_eq!(body["name"], "tempera-0000beef-job-1");
        assert_eq!(body["ports"][0], "22/tcp");
        assert_eq!(body["env"]["PUBLIC_KEY"], "ssh-ed25519 AAA test");
        assert_eq!(body["interruptible"], true);
        assert_eq!(body["cloudType"], "COMMUNITY");
    }

    #[test]
    fn pods_parse_with_ssh_endpoint() {
        let v = fixture("runpod_pods.json");
        let pods: Vec<Instance> = v.as_array().unwrap().iter().filter_map(parse_pod).collect();
        assert_eq!(pods.len(), 2);
        assert!(pods[0].ssh_ready() && pods[0].is_tempera());
        assert_eq!(pods[0].ssh_port, Some(40123));
        assert_eq!(pods[1].state, InstanceState::Starting, "no port mapping yet");
        assert!(!pods[1].is_tempera());
    }
}
