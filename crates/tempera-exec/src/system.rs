//! Hardware inventory sent in `Register`.

use tempera_proto::v1::{Gpu, Hardware};

pub fn hardware() -> Hardware {
    Hardware {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu_cores: std::thread::available_parallelism().map_or(1, |n| n.get() as u32),
        ram_bytes: ram_bytes().unwrap_or(0),
        disk_free_bytes: 0,
        gpus: nvidia_gpus(),
        python_available: which("uv") || which("python3"),
    }
}

fn ram_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Uses `nvidia-smi` when present (NVML integration comes in M4).
fn nvidia_gpus() -> Vec<Gpu> {
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut parts = l.split(',').map(str::trim);
            let name = parts.next()?.to_owned();
            let mib: u64 = parts.next()?.parse().ok()?;
            let driver = parts.next().unwrap_or("").to_owned();
            Some(Gpu {
                name,
                vram_bytes: mib * 1024 * 1024,
                driver_version: driver,
                cuda_version: String::new(),
            })
        })
        .collect()
}

pub fn which(bin: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| std::env::split_paths(&paths).any(|p| p.join(bin).is_file()))
}
