//! Pricing catalog (#54): timestamped snapshots of each provider's offers in
//! `<home>/pricing/<provider>-<unix seconds>.json`. Estimates read the newest
//! snapshot and show its age; old snapshots stay as price history.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Offer, Provider, ResourceRequest, Result, now_s};

/// Snapshots older than this are reported as stale.
pub const STALE_AFTER_S: u64 = 6 * 3600;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub provider: String,
    pub fetched_at: u64,
    pub offers: Vec<Offer>,
}

impl Snapshot {
    pub fn age_s(&self, now: u64) -> u64 {
        now.saturating_sub(self.fetched_at)
    }

    pub fn is_stale(&self, now: u64) -> bool {
        self.age_s(now) > STALE_AFTER_S
    }

    pub fn cheapest(&self) -> Option<&Offer> {
        self.offers
            .iter()
            .min_by(|a, b| a.usd_per_hour.total_cmp(&b.usd_per_hour))
    }
}

pub fn pricing_dir(home: &Path) -> PathBuf {
    home.join("pricing")
}

/// Writes a snapshot; returns its path.
pub fn save(home: &Path, snap: &Snapshot) -> Result<PathBuf> {
    let dir = pricing_dir(home);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}-{}.json", snap.provider, snap.fetched_at));
    std::fs::write(&path, serde_json::to_vec_pretty(snap)?)?;
    Ok(path)
}

/// Fetches the provider's offers (broad request: every GPU, spot allowed) and saves them.
pub fn sync(home: &Path, provider: &dyn Provider) -> Result<Snapshot> {
    let req = ResourceRequest {
        min_vram_gb: None,
        min_ram_gb: 0.0,
        min_disk_gb: 0.0,
        spot: true,
        max_usd_per_hour: None,
        min_reliability: 0.0,
    };
    let snap = Snapshot {
        provider: provider.name().to_owned(),
        fetched_at: now_s(),
        offers: provider.list_offers(&req)?,
    };
    save(home, &snap)?;
    Ok(snap)
}

/// `(fetched_at, path)` of every snapshot of `provider`, oldest first.
pub fn history(home: &Path, provider: &str) -> Vec<(u64, PathBuf)> {
    let prefix = format!("{provider}-");
    let mut out: Vec<(u64, PathBuf)> = std::fs::read_dir(pricing_dir(home))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let ts = name.strip_prefix(&prefix)?.strip_suffix(".json")?.parse().ok()?;
            Some((ts, e.path()))
        })
        .collect();
    out.sort_by_key(|(ts, _)| *ts);
    out
}

pub fn load(path: &Path) -> Result<Snapshot> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// Newest snapshot of `provider`, if any.
pub fn latest(home: &Path, provider: &str) -> Option<Snapshot> {
    history(home, provider).last().and_then(|(_, p)| load(p).ok())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::mock::Mock;

    #[test]
    fn snapshots_are_timestamped_and_latest_wins() {
        let home = tempfile::tempdir().unwrap();
        let mut a = Snapshot {
            provider: "mock".into(),
            fetched_at: 100,
            offers: Mock::catalog(100),
        };
        save(home.path(), &a).unwrap();
        a.fetched_at = 200;
        a.offers[0].usd_per_hour = 0.07;
        save(home.path(), &a).unwrap();
        save(
            home.path(),
            &Snapshot {
                provider: "other".into(),
                fetched_at: 300,
                offers: vec![],
            },
        )
        .unwrap();
        assert_eq!(history(home.path(), "mock").len(), 2);
        let l = latest(home.path(), "mock").unwrap();
        assert_eq!(l.fetched_at, 200);
        assert!((l.cheapest().unwrap().usd_per_hour - 0.07).abs() < 1e-9);
        assert!(l.is_stale(200 + STALE_AFTER_S + 1) && !l.is_stale(250));
        assert!(latest(home.path(), "runpod").is_none());
    }

    #[test]
    fn sync_uses_the_provider() {
        let home = tempfile::tempdir().unwrap();
        let m = Mock::with_state(home.path().join("state.json"));
        let s = sync(home.path(), &m).unwrap();
        assert_eq!(s.offers.len(), 4, "broad request keeps CPU-only offers too");
        assert_eq!(latest(home.path(), "mock").unwrap().offers.len(), 4);
    }
}
