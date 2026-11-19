//! Vast.ai: the cheapest marketplace, with a reliability filter so flaky
//! hosts never get picked. Auth: `Authorization: Bearer $VAST_API_KEY`.
//!
//! Endpoints (`https://console.vast.ai/api/v0`): `POST /bundles/` (search),
//! `PUT /asks/{id}/` (rent), `GET /instances/`, `DELETE /instances/{id}/`.
//! Built from Vast's public API documentation; not yet exercised against a
//! live account (see docs/wiki/Execution-and-Machines.md).

use serde_json::{Value, json};

use crate::{
    BootSpec, CloudError, Instance, InstanceState, Offer, Provider, ResourceRequest, Result, api_key,
    http_agent, now_s, read_json,
};

const NAME: &str = "vast";
pub const KEY_ENV: &str = "VAST_API_KEY";
const BASE: &str = "https://console.vast.ai/api/v0";
/// Hosts below this download bandwidth make data/image transfer dominate.
const MIN_INET_DOWN_MBPS: f64 = 100.0;

#[derive(Debug)]
pub struct Vast {
    key: String,
    base: String,
}

impl Vast {
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(api_key(NAME, KEY_ENV)?))
    }

    pub fn new(key: String) -> Self {
        Self {
            key,
            base: BASE.to_owned(),
        }
    }

    pub fn with_base(mut self, base: &str) -> Self {
        base.clone_into(&mut self.base);
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

/// Search query: verified, rentable, reliable hosts with enough VRAM, bandwidth and disk.
pub fn search_query(req: &ResourceRequest, spot: bool) -> Value {
    let mut q = json!({
        "verified": {"eq": true},
        "rentable": {"eq": true},
        "rented": {"eq": false},
        "reliability2": {"gte": req.min_reliability},
        "inet_down": {"gte": MIN_INET_DOWN_MBPS},
        "disk_space": {"gte": req.min_disk_gb},
        "cpu_ram": {"gte": req.min_ram_gb * 1024.0},
        "num_gpus": {"gte": 1},
        "type": if spot { "bid" } else { "on-demand" },
        "order": [["dph_total", "asc"]],
        "limit": 64,
    });
    if let Some(v) = req.min_vram_gb {
        q["gpu_ram"] = json!({"gte": v * 1024.0});
    }
    q
}

/// `bundles` response → offers. Spot offers are priced at the host's minimum bid.
pub fn parse_bundles(v: &Value, spot: bool, fetched_at: u64) -> Result<Vec<Offer>> {
    let list = v["offers"].as_array().ok_or_else(|| api("missing `offers`"))?;
    let mut out = Vec::new();
    for o in list {
        let Some(id) = o["id"].as_u64() else { continue };
        let gpus = o["num_gpus"].as_u64().unwrap_or(1).max(1) as u32;
        let price = if spot {
            o["min_bid"].as_f64()
        } else {
            o["dph_total"].as_f64()
        };
        let Some(price) = price.filter(|p| *p > 0.0) else {
            continue;
        };
        out.push(Offer {
            provider: NAME.into(),
            id: format!("{id}|{}", if spot { "bid" } else { "ondemand" }),
            gpu_name: o["gpu_name"].as_str().unwrap_or("?").to_owned(),
            gpu_count: gpus,
            vram_gb: o["gpu_ram"].as_f64().unwrap_or(0.0) / 1024.0,
            cpu_cores: o["cpu_cores_effective"].as_f64().unwrap_or(4.0).floor() as u32,
            ram_gb: o["cpu_ram"].as_f64().unwrap_or(0.0) / 1024.0,
            usd_per_hour: price,
            spot,
            region: o["geolocation"].as_str().map(str::to_owned),
            reliability: o["reliability2"].as_f64().or(o["reliability"].as_f64()),
            inet_down_mbps: o["inet_down"].as_f64(),
            disk_gb: o["disk_space"].as_f64(),
            fetched_at,
        });
    }
    Ok(out)
}

/// Body of `PUT /asks/{id}/`.
pub fn rent_request(offer: &Offer, boot: &BootSpec) -> Value {
    let mut body = json!({
        "client_id": "me",
        "image": boot.image,
        "disk": boot.disk_gb,
        "label": boot.name,
        "runtype": "ssh",
        "env": { "PUBLIC_KEY": boot.ssh_public_key },
        "onstart": "mkdir -p ~/.ssh && echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys",
    });
    if offer.spot {
        body["price"] = json!(boot.bid_usd_per_hour.unwrap_or(offer.usd_per_hour));
    }
    body
}

pub fn parse_instance(v: &Value) -> Option<Instance> {
    let id = v["id"].as_u64()?.to_string();
    let state = match v["actual_status"].as_str().unwrap_or("") {
        "running" => InstanceState::Running,
        "loading" | "created" | "scheduling" => InstanceState::Starting,
        "exited" | "stopped" | "offline" => InstanceState::Stopped,
        _ => InstanceState::Unknown,
    };
    let ssh_host = v["ssh_host"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let ssh_port = v["ssh_port"].as_u64().and_then(|p| u16::try_from(p).ok());
    Some(Instance {
        provider: NAME.into(),
        id,
        name: v["label"].as_str().unwrap_or_default().to_owned(),
        state: if state == InstanceState::Running && (ssh_host.is_none() || ssh_port.is_none()) {
            InstanceState::Starting
        } else {
            state
        },
        ssh_host,
        ssh_port,
        ssh_user: Some("root".into()),
        usd_per_hour: v["dph_total"].as_f64(),
    })
}

impl Provider for Vast {
    fn name(&self) -> &str {
        NAME
    }

    fn list_offers(&self, req: &ResourceRequest) -> Result<Vec<Offer>> {
        let mut offers = Vec::new();
        for spot in [false, true] {
            if spot && !req.spot {
                continue;
            }
            let resp = http_agent()
                .post(&format!("{}/bundles/", self.base))
                .header("Authorization", &self.auth())
                .send_json(search_query(req, spot));
            offers.extend(parse_bundles(&read_json(NAME, resp)?, spot, now_s())?);
        }
        offers.retain(|o| req.accepts(o));
        offers.sort_by(|a, b| a.usd_per_hour.total_cmp(&b.usd_per_hour));
        Ok(offers)
    }

    fn launch(&self, offer: &Offer, boot: &BootSpec) -> Result<String> {
        let ask = offer.id.split('|').next().unwrap_or(&offer.id);
        let resp = http_agent()
            .put(&format!("{}/asks/{ask}/", self.base))
            .header("Authorization", &self.auth())
            .send_json(rent_request(offer, boot));
        let v = read_json(NAME, resp)?;
        if v["success"].as_bool() == Some(false) {
            return Err(api(v["msg"].as_str().unwrap_or("rent refused").to_owned()));
        }
        v["new_contract"]
            .as_u64()
            .map(|id| id.to_string())
            .ok_or_else(|| api("rent: no `new_contract` in response"))
    }

    fn status(&self, id: &str) -> Result<Instance> {
        Ok(self
            .instances()?
            .into_iter()
            .find(|i| i.id == id)
            .unwrap_or(Instance {
                provider: NAME.into(),
                id: id.to_owned(),
                name: String::new(),
                state: InstanceState::Gone,
                ssh_host: None,
                ssh_port: None,
                ssh_user: None,
                usd_per_hour: None,
            }))
    }

    fn instances(&self) -> Result<Vec<Instance>> {
        let resp = http_agent()
            .get(&format!("{}/instances/", self.base))
            .header("Authorization", &self.auth())
            .call();
        let v = read_json(NAME, resp)?;
        Ok(v["instances"]
            .as_array()
            .map(|a| a.iter().filter_map(parse_instance).collect())
            .unwrap_or_default())
    }

    fn terminate(&self, id: &str) -> Result<()> {
        let resp = http_agent()
            .delete(&format!("{}/instances/{id}/", self.base))
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
    fn bundles_parse_and_reliability_filter_applies() {
        let v = fixture("vast_bundles.json");
        let offers = parse_bundles(&v, false, 7).unwrap();
        assert_eq!(offers.len(), 2);
        assert_eq!(offers[0].id, "111|ondemand");
        assert!((offers[0].vram_gb - 24.0).abs() < 1e-9);
        let req = ResourceRequest::default(); // min reliability 0.98
        let kept: Vec<_> = offers.iter().filter(|o| req.accepts(o)).collect();
        assert_eq!(kept.len(), 1, "the 0.90-reliability host is filtered out");
        let spot = parse_bundles(&v, true, 7).unwrap();
        assert!((spot[0].usd_per_hour - 0.09).abs() < 1e-9 && spot[0].spot);
    }

    #[test]
    fn query_and_rent_request() {
        let q = search_query(&ResourceRequest::default(), true);
        assert_eq!(q["type"], "bid");
        assert_eq!(q["reliability2"]["gte"], 0.98);
        assert_eq!(q["gpu_ram"]["gte"], 8.0 * 1024.0);
        let offers = parse_bundles(&fixture("vast_bundles.json"), true, 0).unwrap();
        let body = rent_request(&offers[0], &BootSpec::for_job("j", "ssh-ed25519 K"));
        assert_eq!(body["label"], "tempera-j");
        assert_eq!(body["price"], 0.09);
    }

    #[test]
    fn instances_parse() {
        let v = fixture("vast_instances.json");
        let list: Vec<Instance> = v["instances"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(parse_instance)
            .collect();
        assert_eq!(list.len(), 2);
        assert!(list[0].ssh_ready() && list[0].is_tempera());
        assert_eq!(list[1].state, InstanceState::Starting);
    }
}
