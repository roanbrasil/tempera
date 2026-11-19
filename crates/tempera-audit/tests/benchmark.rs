//! Audit benchmark (#36): datasets with *planted* problems; every check must
//! find its plant (recall) and a clean dataset must stay quiet (precision).
//! Runs in CI with `cargo test`; prints a scorecard with `--nocapture`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Write as _;
use std::path::Path;

use tempera_audit::{AuditReport, audit};
use tempera_core::spec::{JobSpec, TaskType};
use tempera_data::{infer_task, ingest, make_split, profile_dataset};

/// Deterministic pseudo-random numbers in [0, 1).
fn rng(i: u64, salt: u64) -> f64 {
    let mut z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

struct Case {
    name: &'static str,
    /// Finding id prefix that must appear.
    expect: &'static str,
    csv: String,
    spec_extra: &'static str,
    target: Option<&'static str>,
}

/// A clean binary dataset: 2 informative numeric features, a categorical, balanced classes.
fn base_rows(n: u64, extra_header: &str, extra: impl Fn(u64, bool) -> String) -> String {
    let mut s = format!("x1,x2,plan,label{extra_header}\n");
    for i in 0..n {
        let x1 = rng(i, 1) * 10.0;
        let x2 = rng(i, 2) * 5.0;
        let label = x1 + rng(i, 3) * 8.0 > 9.0;
        let plan = ["basic", "pro", "team"][(rng(i, 4) * 3.0) as usize];
        let _ = writeln!(
            s,
            "{x1:.3},{x2:.3},{plan},{}{}",
            if label { "yes" } else { "no" },
            extra(i, label)
        );
    }
    s
}

/// A valid CPF built from a seed (check digits computed).
fn cpf(i: u64) -> String {
    let mut d: Vec<u32> = format!("{:09}", (i * 7_919 + 123_456_789) % 1_000_000_000)
        .chars()
        .filter_map(|c| c.to_digit(10))
        .collect();
    for n in [9usize, 10] {
        let sum: u32 = (0..n).map(|k| d[k] * (n as u32 + 1 - k as u32)).sum();
        let r = (sum * 10) % 11;
        d.push(if r == 10 { 0 } else { r });
    }
    let s: String = d
        .iter()
        .map(|x| char::from_digit(*x, 10).unwrap_or('0'))
        .collect();
    format!("{}.{}.{}-{}", &s[0..3], &s[3..6], &s[6..9], &s[9..11])
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "A1 target leakage",
            expect: "A1:refund",
            csv: base_rows(2000, ",refund", |i, y| {
                format!(",{:.2}", if y { 10.0 + rng(i, 9) * 50.0 } else { 0.0 })
            }),
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A2 duplicate rows",
            expect: "A2",
            csv: {
                let mut s = base_rows(1000, "", |_, _| String::new());
                let dup: Vec<&str> = s.lines().skip(1).take(50).collect();
                let extra = dup.join("\n") + "\n";
                s.push_str(&extra);
                s
            },
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A3 repeated entity",
            expect: "A3:customer_id",
            csv: base_rows(2000, ",customer_id", |i, _| format!(",C{}", i % 400)),
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A4 date column, random split",
            expect: "A4",
            csv: base_rows(1500, ",when", |i, _| {
                format!(",2024-{:02}-{:02}", 1 + i % 12, 1 + i % 28)
            }),
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A5 class imbalance",
            expect: "A5",
            csv: {
                let mut s = String::from("x1,x2,label\n");
                for i in 0..2000u64 {
                    let pos = rng(i, 7) < 0.04;
                    let _ = writeln!(
                        s,
                        "{:.3},{:.3},{}",
                        rng(i, 1) + f64::from(u8::from(pos)),
                        rng(i, 2),
                        if pos { "yes" } else { "no" }
                    );
                }
                s
            },
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A6 train/test drift",
            expect: "A6:x1",
            csv: {
                // With a temporal split the last 20% (test) has shifted x1.
                let mut s = String::from("x1,x2,day,label\n");
                for i in 0..2000u64 {
                    let late = i >= 1600;
                    let x1 = rng(i, 1) * 10.0 + if late { 8.0 } else { 0.0 };
                    let _ = writeln!(
                        s,
                        "{x1:.3},{:.3},2024-01-01T00:00:{:02},{}",
                        rng(i, 2),
                        0,
                        if rng(i, 3) > 0.5 { "yes" } else { "no" }
                    );
                }
                // make time increase with the row
                s.lines()
                    .enumerate()
                    .map(|(k, l)| {
                        if k == 0 {
                            l.to_owned()
                        } else {
                            l.replace(
                                "2024-01-01T00:00:00",
                                &format!(
                                    "2024-01-01T{:02}:{:02}:{:02}",
                                    (k / 3600) % 24,
                                    (k / 60) % 60,
                                    k % 60
                                ),
                            )
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n"
            },
            spec_extra: ", time_column: day",
            target: Some("label"),
        },
        Case {
            name: "A10 identifier column",
            expect: "A10:uuid",
            csv: base_rows(1500, ",uuid", |i, _| {
                format!(",u-{:08x}", i.wrapping_mul(2_654_435_761))
            }),
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A10 constant column",
            expect: "A10:country",
            csv: base_rows(1000, ",country", |_, _| ",BR".into()),
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A12 tiny rare class",
            expect: "A12:class",
            csv: {
                let mut s = String::from("x1,x2,label\n");
                for i in 0..400u64 {
                    let pos = i % 40 == 0;
                    let _ = writeln!(
                        s,
                        "{:.3},{:.3},{}",
                        rng(i, 1),
                        rng(i, 2),
                        if pos { "yes" } else { "no" }
                    );
                }
                s
            },
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A13 informative missingness",
            expect: "A13:income",
            csv: base_rows(2000, ",income", |i, y| {
                if y && rng(i, 11) < 0.6 {
                    ",".into()
                } else {
                    format!(",{:.0}", 1000.0 + rng(i, 12) * 9000.0)
                }
            }),
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A11 e-mail column",
            expect: "A11:contact",
            csv: base_rows(800, ",contact", |i, _| format!(",user{i}@example.com")),
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A11 CPF column",
            expect: "A11:document",
            csv: base_rows(800, ",document", |i, _| format!(",{}", cpf(i))),
            spec_extra: "",
            target: Some("label"),
        },
        Case {
            name: "A9 sensitive proxy",
            expect: "A9:title",
            csv: base_rows(1500, ",gender,title", |i, _| {
                let female = rng(i, 21) < 0.5;
                let title = if rng(i, 22) < 0.95 {
                    if female { "ms" } else { "mr" }
                } else {
                    "dr"
                };
                format!(",{},{title}", if female { "f" } else { "m" })
            }),
            spec_extra: ", sensitive: [gender]",
            target: Some("label"),
        },
    ]
}

fn run(dir: &Path, name: &str, csv: &str, spec_extra: &str, target: Option<&str>) -> AuditReport {
    let p = dir.join(format!("{}.csv", name.replace([' ', '/'], "_")));
    std::fs::write(&p, csv).unwrap();
    let ds = ingest(&p, dir).unwrap();
    let profile = profile_dataset(&ds).unwrap();
    let target = target.map(|t| format!(", target: {t}")).unwrap_or_default();
    let spec = JobSpec::from_yaml(&format!(
        "version: 1\nname: b\ndata: {{ source: x.csv{target}{spec_extra} }}\n"
    ))
    .unwrap();
    let inference = infer_task(&spec, &profile).unwrap();
    let mut report = audit(&ds, &spec, &profile, &inference).unwrap();
    // A6 runs after the split, as in the orchestrator.
    if inference.task != TaskType::Clustering {
        let mut plan = report.plan.clone();
        plan.dedupe = true;
        if let Ok(split) = make_split(&ds, &spec, inference.task, &plan) {
            let drift = tempera_audit::advanced::drift(
                &split.path(tempera_data::Partition::Train),
                &split.path(tempera_data::Partition::Test),
                &split.features,
                split.strategy,
            )
            .unwrap();
            tempera_audit::add_findings(&mut report, &spec, drift);
        }
    }
    report
}

#[test]
fn planted_issues_recall_and_clean_precision() {
    let tmp = tempfile::tempdir().unwrap();
    let mut found = 0;
    let mut card = String::from("\naudit benchmark\n");
    let all = cases();
    for c in &all {
        let r = run(tmp.path(), c.name, &c.csv, c.spec_extra, c.target);
        let hit = r.findings.iter().any(|f| f.id.starts_with(c.expect));
        found += usize::from(hit);
        let _ = writeln!(
            card,
            "  {} {:<30} expects {}",
            if hit { "✓" } else { "✗" },
            c.name,
            c.expect
        );
    }
    let recall = found as f64 / all.len() as f64;
    let _ = writeln!(card, "  recall: {found}/{} = {:.0}%", all.len(), recall * 100.0);

    let clean = run(
        tmp.path(),
        "clean",
        &base_rows(2000, "", |_, _| String::new()),
        "",
        Some("label"),
    );
    let noise: Vec<&str> = clean.findings.iter().map(|f| f.id.as_str()).collect();
    let _ = writeln!(card, "  clean dataset findings: {noise:?}");
    println!("{card}");
    assert!(recall >= 0.9, "{card}");
    assert!(noise.is_empty(), "false positives on clean data: {noise:?}");
}
