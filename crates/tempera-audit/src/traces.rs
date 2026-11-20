//! Audit of agent traces (#67), run before fine-tuning an LLM on tool use.
//! Same contract as the tabular audit: red findings block training until a
//! human decides.
//!
//! | id  | check |
//! |-----|-------|
//! | T1  | same prompt (exact or near duplicate) in train and test |
//! | T2  | duplicate conversations inside train |
//! | T3  | contamination with a declared benchmark (n-gram overlap) |
//! | T4  | personal data in messages or tool results |
//! | T5  | malformed tool calls (not JSON, undeclared tool, schema violations) |
//! | T6  | too few conversations |
//! | T7  | one tool dominates the calls |
//! | T8  | no conversation without tool calls (never learns when *not* to call) |
//! | T9  | unreadable lines |

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempera_data::split::Partition;
use tempera_data::traces::{Conversation, Role, TraceProfile, TraceSet, normalize, profile_traces};

use crate::privacy::{PiiKind, classify};
use crate::{Finding, Resolution, Severity};

/// Jaccard similarity (word 3-shingles) above which two prompts are near duplicates.
pub const NEAR_DUP: f64 = 0.8;
/// Word n-gram length for benchmark contamination (prompts are short, so 8, not 13).
pub const CONTAM_N: usize = 8;
/// Share of a benchmark item's n-grams found in training that makes it contaminated.
pub const CONTAM_SHARE: f64 = 0.5;
/// Share of a benchmark's items that, once contaminated, makes the finding red (Tülu 3).
pub const CONTAM_BENCH_SHARE: f64 = 0.02;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceAudit {
    pub profile: TraceProfile,
    pub findings: Vec<Finding>,
    pub blocked: bool,
    /// Conversations per partition.
    pub split: HashMap<String, usize>,
}

/// One benchmark the model will be evaluated on (prompts only).
#[derive(Debug, Clone)]
pub struct Benchmark {
    pub name: String,
    pub prompts: Vec<String>,
}

fn finding(id: &str, sev: Severity, title: &str, evidence: String, suggestion: &str) -> Finding {
    Finding {
        id: id.to_owned(),
        check: id.to_owned(),
        severity: sev,
        column: None,
        title: title.to_owned(),
        evidence,
        suggestion: suggestion.to_owned(),
        resolution: Resolution::Unresolved,
    }
}

fn lines(convs: &[&Conversation], max: usize) -> String {
    let mut l: Vec<String> = convs.iter().take(max).map(|c| c.line.to_string()).collect();
    if convs.len() > max {
        l.push("…".into());
    }
    l.join(", ")
}

/// Runs every trace check. `acknowledged` lists finding ids the user accepted.
pub fn audit_traces(
    set: &TraceSet,
    split: &[Partition],
    benchmarks: &[Benchmark],
    acknowledged: &[String],
) -> TraceAudit {
    let convs = &set.conversations;
    let mut f = Vec::new();
    f.extend(cross_split(convs, split));
    f.extend(duplicates(convs, split));
    f.extend(contamination(convs, benchmarks));
    f.extend(personal_data(convs));
    f.extend(malformed_calls(convs));
    f.extend(sample_size(convs.len()));
    f.extend(balance(convs));
    if !set.invalid_lines.is_empty() {
        let share = set.invalid_lines.len() as f64 / (set.invalid_lines.len() + convs.len()) as f64;
        let first = &set.invalid_lines[0];
        f.push(finding(
            "T9",
            if share > 0.05 {
                Severity::Red
            } else {
                Severity::Yellow
            },
            "Unreadable lines",
            format!(
                "{} lines ({:.1}%) could not be read; first: line {}: {}",
                set.invalid_lines.len(),
                share * 100.0,
                first.0,
                first.1
            ),
            "Fix or remove them; they are skipped.",
        ));
    }
    for x in &mut f {
        if acknowledged.contains(&x.id) {
            x.resolution = Resolution::Acknowledged;
        }
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for p in split {
        *counts.entry(format!("{p:?}").to_lowercase()).or_default() += 1;
    }
    TraceAudit {
        profile: profile_traces(set),
        blocked: f.iter().any(Finding::blocks),
        findings: f,
        split: counts,
    }
}

// ---------- T1/T2: duplicates ----------

fn shingles(text: &str) -> HashSet<u64> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return [fnv(&words.join(" "))].into();
    }
    words.windows(3).map(|w| fnv(&w.join(" "))).collect()
}

/// FNV-1a: stable across runs and platforms (unlike `DefaultHasher`).
fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

const PERMS: usize = 64;
const BANDS: usize = 16; // 16 bands × 4 rows: ~0.8 Jaccard → ~98% chance to collide

fn minhash(sh: &HashSet<u64>) -> [u64; PERMS] {
    let mut sig = [u64::MAX; PERMS];
    for &x in sh {
        for (i, s) in sig.iter_mut().enumerate() {
            let a = 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 * 2 + 1);
            let v = (x ^ a).wrapping_mul(0xbf58_476d_1ce4_e5b9).rotate_left(31) ^ (i as u64);
            if v < *s {
                *s = v;
            }
        }
    }
    sig
}

fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

/// Pairs (i, j), i < j, whose prompts are near duplicates (MinHash/LSH candidates, verified exactly).
pub fn near_duplicate_pairs(prompts: &[String]) -> Vec<(usize, usize)> {
    let sh: Vec<HashSet<u64>> = prompts.iter().map(|p| shingles(p)).collect();
    let rows = PERMS / BANDS;
    let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    for (i, s) in sh.iter().enumerate() {
        let sig = minhash(s);
        for b in 0..BANDS {
            let key = sig[b * rows..(b + 1) * rows]
                .iter()
                .fold(0u64, |h, &x| h.rotate_left(13) ^ x);
            buckets.entry((b, key)).or_default().push(i);
        }
    }
    let mut seen = HashSet::new();
    let mut out = vec![];
    for ids in buckets.values().filter(|v| v.len() > 1) {
        for (k, &i) in ids.iter().enumerate() {
            for &j in &ids[k + 1..] {
                let (i, j) = (i.min(j), i.max(j));
                if seen.insert((i, j)) && jaccard(&sh[i], &sh[j]) >= NEAR_DUP {
                    out.push((i, j));
                }
            }
        }
    }
    out.sort_unstable();
    out
}

fn prompt_of(c: &Conversation) -> String {
    normalize(c.first_user().unwrap_or(""))
}

fn cross_split(convs: &[Conversation], split: &[Partition]) -> Option<Finding> {
    let prompts: Vec<String> = convs.iter().map(prompt_of).collect();
    let mut leaked: HashSet<usize> = HashSet::new();
    for (i, j) in near_duplicate_pairs(&prompts) {
        let (pi, pj) = (split[i], split[j]);
        let train_test = matches!(
            (pi, pj),
            (Partition::Train, Partition::Test) | (Partition::Test, Partition::Train)
        );
        if train_test && !prompts[i].is_empty() {
            leaked.insert(if pi == Partition::Test { i } else { j });
        }
    }
    let n_test = split.iter().filter(|p| **p == Partition::Test).count().max(1);
    if leaked.is_empty() {
        return None;
    }
    let share = leaked.len() as f64 / n_test as f64;
    let mut ex: Vec<&Conversation> = leaked.iter().map(|&i| &convs[i]).collect();
    ex.sort_by_key(|c| c.line);
    Some(finding(
        "T1",
        if share >= 0.01 {
            Severity::Red
        } else {
            Severity::Yellow
        },
        "Same prompt in train and test",
        format!(
            "{} test conversations ({:.1}% of test) repeat or nearly repeat a train prompt \
             (Jaccard ≥ {NEAR_DUP}). Lines: {}. The test score would measure memorization.",
            leaked.len(),
            share * 100.0,
            lines(&ex, 8)
        ),
        "Give related conversations the same `task_id` so they share a split, or remove the copies.",
    ))
}

fn duplicates(convs: &[Conversation], split: &[Partition]) -> Option<Finding> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut dups = vec![];
    for (c, p) in convs.iter().zip(split) {
        if *p != Partition::Train {
            continue;
        }
        let key: String = c
            .messages
            .iter()
            .map(|m| normalize(&m.content))
            .collect::<Vec<_>>()
            .join("\u{1}");
        let n = seen.entry(key).or_default();
        *n += 1;
        if *n > 1 {
            dups.push(c);
        }
    }
    let n_train = split.iter().filter(|p| **p == Partition::Train).count().max(1);
    (!dups.is_empty()).then(|| {
        let share = dups.len() as f64 / n_train as f64;
        finding(
            "T2",
            if share >= 0.3 {
                Severity::Red
            } else {
                Severity::Yellow
            },
            "Duplicate conversations in train",
            format!(
                "{} exact copies ({:.1}% of train), e.g. lines {}. Duplicates are memorized and \
                 waste compute.",
                dups.len(),
                share * 100.0,
                lines(&dups, 8)
            ),
            "Deduplicate before training.",
        )
    })
}

// ---------- T3: contamination ----------

fn ngrams(text: &str, n: usize) -> HashSet<u64> {
    let w: Vec<&str> = text.split_whitespace().collect();
    if w.len() < n {
        // Short prompt: the whole normalized text is the unit.
        return if w.is_empty() {
            HashSet::new()
        } else {
            [fnv(&w.join(" "))].into()
        };
    }
    w.windows(n).map(|g| fnv(&g.join(" "))).collect()
}

fn contamination(convs: &[Conversation], benchmarks: &[Benchmark]) -> Vec<Finding> {
    // Tülu 3 decontamination (Lambert et al. 2024): 8-grams of user turns; a benchmark item
    // is contaminated when more than half of its n-grams occur in the training data, and the
    // benchmark is flagged red when more than 2% of its items are.
    let mut owner: HashMap<u64, usize> = HashMap::new(); // n-gram → first conversation index
    for (i, c) in convs.iter().enumerate() {
        for m in c.messages.iter().filter(|m| m.role == Role::User) {
            for g in ngrams(&normalize(&m.content), CONTAM_N) {
                owner.entry(g).or_insert(i);
            }
        }
    }
    let mut out = vec![];
    for b in benchmarks {
        let mut items = 0usize;
        let mut hit_convs: HashSet<usize> = HashSet::new();
        for p in &b.prompts {
            let g = ngrams(&normalize(p), CONTAM_N);
            if g.is_empty() {
                continue;
            }
            let found: Vec<usize> = g.iter().filter_map(|x| owner.get(x).copied()).collect();
            if found.len() as f64 / g.len() as f64 > CONTAM_SHARE {
                items += 1;
                hit_convs.extend(found);
            }
        }
        if items == 0 {
            continue;
        }
        let share = items as f64 / b.prompts.len().max(1) as f64;
        let mut ex: Vec<&Conversation> = hit_convs.iter().map(|&i| &convs[i]).collect();
        ex.sort_by_key(|c| c.line);
        out.push(finding(
            &format!("T3:{}", b.name),
            if share > CONTAM_BENCH_SHARE {
                Severity::Red
            } else {
                Severity::Yellow
            },
            "Benchmark contamination",
            format!(
                "{items} of {} `{}` items ({:.1}%) appear in the training conversations \
                 ({CONTAM_N}-gram share > {CONTAM_SHARE}), e.g. lines {}. Scores on `{}` would be inflated.",
                b.prompts.len(),
                b.name,
                share * 100.0,
                lines(&ex, 8),
                b.name
            ),
            "Remove those conversations, or do not report this benchmark for this model.",
        ));
    }
    out
}

// ---------- T4: personal data ----------

static EMAIL_IN_TEXT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\w.+-]+@[\w-]+(\.[\w-]+)+").unwrap_or_else(|_| unreachable!()));
static NUMBERISH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\+?\(?\d[\d\s().\-/]{6,22}\d").unwrap_or_else(|_| unreachable!()));

/// Personal data found in free text.
pub fn pii_in_text(text: &str) -> Vec<PiiKind> {
    let mut out = vec![];
    if EMAIL_IN_TEXT.is_match(text) {
        out.push(PiiKind::Email);
    }
    for m in NUMBERISH.find_iter(text) {
        if let Some(k) = classify(m.as_str()) {
            out.push(k);
        }
    }
    for tok in text.split(|c: char| c.is_whitespace() || ",;\"'()[]{}<>".contains(c)) {
        let tok = tok.trim_matches(|c: char| ".:!?".contains(c));
        if tok.len() >= 7
            && let Some(k @ PiiKind::IpAddress) = classify(tok)
        {
            out.push(k);
        }
    }
    out.sort_by_key(|k| k.label());
    out.dedup();
    out
}

fn personal_data(convs: &[Conversation]) -> Option<Finding> {
    let mut kinds: HashMap<&'static str, usize> = HashMap::new();
    let mut hit = vec![];
    for c in convs {
        let mut found = false;
        for m in &c.messages {
            let mut text = m.content.clone();
            for t in &m.tool_calls {
                text.push(' ');
                text.push_str(&t.raw);
            }
            for k in pii_in_text(&text) {
                *kinds.entry(k.label()).or_default() += 1;
                found = true;
            }
        }
        if found {
            hit.push(c);
        }
    }
    if hit.is_empty() {
        return None;
    }
    let share = hit.len() as f64 / convs.len() as f64;
    let mut k: Vec<_> = kinds.into_iter().collect();
    k.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let kinds = k
        .iter()
        .map(|(l, n)| format!("{l} ({n})"))
        .collect::<Vec<_>>()
        .join(", ");
    Some(finding(
        "T4",
        if share >= 0.01 {
            Severity::Red
        } else {
            Severity::Yellow
        },
        "Personal data in the conversations",
        format!(
            "{} conversations ({:.1}%) contain {kinds}, e.g. lines {}. Fine-tuned LLMs can \
             memorize and repeat training text (LGPD/GDPR).",
            hit.len(),
            share * 100.0,
            lines(&hit, 8)
        ),
        "Mask or synthesize the values before training.",
    ))
}

// ---------- T5: malformed tool calls ----------

/// Why a call does not match its declared schema, if it does not.
pub fn schema_violation(schema: &Value, args: &Value) -> Option<String> {
    let Some(obj) = args.as_object() else {
        return Some("arguments are not an object".into());
    };
    let props = schema.get("properties").and_then(Value::as_object);
    if let Some(req) = schema.get("required").and_then(Value::as_array) {
        for r in req.iter().filter_map(Value::as_str) {
            if !obj.contains_key(r) {
                return Some(format!("missing required `{r}`"));
            }
        }
    }
    if let Some(props) = props {
        for (k, v) in obj {
            match props.get(k) {
                None if schema.get("additionalProperties") == Some(&Value::Bool(false)) => {
                    return Some(format!("unknown argument `{k}`"));
                }
                Some(p) => {
                    if let Some(t) = p.get("type").and_then(Value::as_str) {
                        let ok = match t {
                            "string" => v.is_string(),
                            "integer" => v.is_i64() || v.is_u64(),
                            "number" => v.is_number(),
                            "boolean" => v.is_boolean(),
                            "array" => v.is_array(),
                            "object" => v.is_object(),
                            "null" => v.is_null(),
                            _ => true,
                        };
                        if !ok {
                            return Some(format!("`{k}` should be {t}"));
                        }
                    }
                    if let Some(allowed) = p.get("enum").and_then(Value::as_array)
                        && !allowed.contains(v)
                    {
                        return Some(format!("`{k}` not in enum"));
                    }
                }
                None => {}
            }
        }
    }
    None
}

fn malformed_calls(convs: &[Conversation]) -> Option<Finding> {
    let (mut total, mut bad) = (0usize, 0usize);
    let mut reasons: HashMap<String, usize> = HashMap::new();
    let mut hit = vec![];
    for c in convs {
        let declared: HashMap<&str, &Value> =
            c.tools.iter().map(|t| (t.name.as_str(), &t.parameters)).collect();
        let mut any = false;
        for m in &c.messages {
            for _ in &m.broken_calls {
                total += 1;
                bad += 1;
                any = true;
                *reasons.entry("tool-call block is not JSON".into()).or_default() += 1;
            }
            for t in &m.tool_calls {
                total += 1;
                let why = if t.name.is_empty() {
                    Some("no tool name".to_owned())
                } else if t.arguments.is_none() {
                    Some("arguments are not JSON".to_owned())
                } else if !declared.is_empty() && !declared.contains_key(t.name.as_str()) {
                    Some("tool not declared".to_owned())
                } else {
                    declared
                        .get(t.name.as_str())
                        .and_then(|s| schema_violation(s, t.arguments.as_ref().unwrap_or(&Value::Null)))
                        .map(|w| {
                            // Group by kind, not by argument name: drop the `quoted` parts.
                            w.split('`')
                                .step_by(2)
                                .collect::<String>()
                                .split_whitespace()
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                };
                if let Some(w) = why {
                    bad += 1;
                    any = true;
                    *reasons.entry(w).or_default() += 1;
                }
            }
        }
        if any {
            hit.push(c);
        }
    }
    if bad == 0 {
        return None;
    }
    let share = bad as f64 / total.max(1) as f64;
    let mut r: Vec<_> = reasons.into_iter().collect();
    r.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Some(finding(
        "T5",
        if share >= 0.02 {
            Severity::Red
        } else {
            Severity::Yellow
        },
        "Malformed tool calls",
        format!(
            "{bad} of {total} tool calls ({:.1}%) are invalid: {}. Lines: {}. The model would \
             learn to emit them.",
            share * 100.0,
            r.iter()
                .map(|(w, n)| format!("{w} ({n})"))
                .collect::<Vec<_>>()
                .join(", "),
            lines(&hit, 8)
        ),
        "Fix or drop these conversations.",
    ))
}

// ---------- T6–T8 ----------

fn sample_size(n: usize) -> Option<Finding> {
    let sev = match n {
        0..50 => Severity::Red,
        50..300 => Severity::Yellow,
        _ => return None,
    };
    Some(finding(
        "T6",
        sev,
        "Few conversations",
        format!("{n} conversations. Below ~300, a held-out test split is too small to tell models apart."),
        "Collect more conversations, or treat the scores as indicative only.",
    ))
}

fn balance(convs: &[Conversation]) -> Vec<Finding> {
    let mut out = vec![];
    let mut per_tool: HashMap<&str, usize> = HashMap::new();
    for t in convs.iter().flat_map(Conversation::tool_calls) {
        *per_tool.entry(t.name.as_str()).or_default() += 1;
    }
    let total: usize = per_tool.values().sum();
    let declared: HashSet<&str> = convs
        .iter()
        .flat_map(|c| c.tools.iter().map(|t| t.name.as_str()))
        .collect();
    if let Some((name, n)) = per_tool.iter().max_by_key(|(k, v)| (**v, std::cmp::Reverse(**k)))
        && total >= 20
        && declared.len().max(per_tool.len()) >= 3
        && *n as f64 / total as f64 >= 0.8
    {
        out.push(finding(
            "T7",
            Severity::Yellow,
            "One tool dominates",
            format!(
                "`{name}` is {:.0}% of {total} tool calls across {} tools. The model may call it \
                 by default.",
                *n as f64 / total as f64 * 100.0,
                declared.len().max(per_tool.len())
            ),
            "Add examples for the other tools.",
        ));
    }
    let without = convs.iter().filter(|c| c.tool_calls().next().is_none()).count();
    if total > 0 && without == 0 && convs.len() >= 20 {
        out.push(finding(
            "T8",
            Severity::Yellow,
            "Every conversation calls a tool",
            "No example where the right answer is to not call a tool; the model will tend to call \
             one even for irrelevant requests."
                .into(),
            "Add conversations answered without tools (irrelevance examples).",
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use tempera_data::traces::{parse_line, split_by_group};

    use super::*;

    fn conv(line: usize, task: &str, prompt: &str, call: Option<&str>) -> Conversation {
        let call = call.map_or(String::new(), |c| {
            format!(r#",{{"role":"assistant","tool_calls":[{{"function":{c}}}]}}"#)
        });
        parse_line(
            &format!(
                r#"{{"task_id":"{task}","messages":[{{"role":"user","content":{}}}{call}],"tools":[{{"type":"function","function":{{"name":"get_weather","parameters":{{"type":"object","properties":{{"city":{{"type":"string"}},"days":{{"type":"integer"}}}},"required":["city"]}}}}}},{{"type":"function","function":{{"name":"search","parameters":{{}}}}}},{{"type":"function","function":{{"name":"book","parameters":{{}}}}}}]}}"#,
                serde_json::to_string(prompt).unwrap()
            ),
            line,
        )
        .unwrap()
    }

    fn set(convs: Vec<Conversation>) -> TraceSet {
        TraceSet {
            conversations: convs,
            invalid_lines: vec![],
        }
    }

    fn ids(a: &TraceAudit) -> Vec<String> {
        a.findings.iter().map(|f| f.id.clone()).collect()
    }

    /// 400 distinct, clean conversations over 3 tools, some without calls.
    fn clean() -> Vec<Conversation> {
        let tools = ["get_weather", "search", "book"];
        (0..400)
            .map(|i| {
                let call = match i % 4 {
                    3 => None,
                    k if tools[k] == "get_weather" => Some(format!(
                        r#"{{"name":"get_weather","arguments":"{{\"city\":\"c{i}\"}}"}}"#
                    )),
                    k => Some(format!(r#"{{"name":"{}","arguments":"{{}}"}}"#, tools[k])),
                };
                conv(
                    i + 1,
                    &format!("t{i}"),
                    &format!(
                        "request number {i} about topic {} with detail {}",
                        i * 7919 % 1000,
                        i * 31
                    ),
                    call.as_deref(),
                )
            })
            .collect()
    }

    fn audit(convs: Vec<Conversation>, bench: &[Benchmark]) -> TraceAudit {
        let s = set(convs);
        let split = split_by_group(&s.conversations, 0.1, 0.2, 7);
        audit_traces(&s, &split, bench, &[])
    }

    #[test]
    fn clean_dataset_has_no_findings() {
        let a = audit(clean(), &[]);
        assert!(a.findings.is_empty(), "{:#?}", a.findings);
        assert!(!a.blocked);
        assert_eq!(a.split.values().sum::<usize>(), 400);
    }

    #[test]
    fn t1_same_prompt_in_train_and_test_blocks() {
        let mut c = clean();
        // 40 paraphrase pairs under different task ids → they land in different splits.
        for i in 0..40 {
            c.push(conv(
                1000 + i,
                &format!("copy{i}"),
                &format!(
                    "request number {i} about topic {} with detail {}",
                    i * 7919 % 1000,
                    i * 31
                ),
                None,
            ));
        }
        let a = audit(c, &[]);
        assert!(ids(&a).contains(&"T1".into()), "{:?}", ids(&a));
        assert!(a.blocked);
    }

    #[test]
    fn near_duplicates_are_found_and_distinct_prompts_are_not() {
        let p = |s: &str| normalize(s);
        let prompts = vec![
            p("what is the weather in recife tomorrow morning please"),
            p("what is the weather in recife tomorrow morning please now"),
            p("book a table for two at an italian restaurant tonight"),
        ];
        assert_eq!(near_duplicate_pairs(&prompts), vec![(0, 1)]);
    }

    #[test]
    fn t2_duplicates_in_train() {
        let mut c = clean();
        let split = split_by_group(&c, 0.1, 0.2, 7);
        let i = split.iter().position(|p| *p == Partition::Train).unwrap();
        for k in 0..5 {
            let mut d = c[i].clone();
            d.line = 2000 + k;
            c.push(d);
        }
        let a = audit(c, &[]);
        let t2 = a.findings.iter().find(|f| f.id == "T2").expect("T2");
        assert!(t2.evidence.starts_with("5 exact copies"), "{}", t2.evidence);
        assert!(!ids(&a).contains(&"T1".into()), "same task id never leaks");
    }

    #[test]
    fn t3_benchmark_contamination_is_red() {
        let mut c = clean();
        let bench_prompt =
            "Find the cheapest direct flight from Recife to Lisbon departing next Friday evening";
        c.push(conv(999, "x", &format!("{bench_prompt}."), None));
        let b = Benchmark {
            name: "bfcl".into(),
            prompts: vec![bench_prompt.into(), "something else entirely here".into()],
        };
        let a = audit(c, std::slice::from_ref(&b));
        let t3 = a.findings.iter().find(|f| f.id == "T3:bfcl").expect("T3");
        assert_eq!(t3.severity, Severity::Red);
        assert!(
            t3.evidence.contains("999") && t3.evidence.starts_with("1 of 2"),
            "{}",
            t3.evidence
        );
        // One contaminated item in a large benchmark is a warning, not a block.
        let mut big = b.clone();
        big.prompts
            .extend((0..99).map(|k| format!("unrelated benchmark question number {k} about cooking pasta")));
        let mut c2 = clean();
        c2.push(conv(999, "x", bench_prompt, None));
        let a = audit(c2, std::slice::from_ref(&big));
        let t3 = a.findings.iter().find(|f| f.id == "T3:bfcl").expect("T3");
        assert_eq!(t3.severity, Severity::Yellow);
        // The clean set alone does not overlap.
        assert!(audit(clean(), &[b]).findings.is_empty());
    }

    #[test]
    fn t4_personal_data_in_text() {
        assert_eq!(
            pii_in_text("mail me at ana.silva@example.com ok"),
            vec![PiiKind::Email]
        );
        assert_eq!(pii_in_text("meu CPF é 529.982.247-25."), vec![PiiKind::Cpf]);
        assert_eq!(
            pii_in_text("card 4111 1111 1111 1111 thanks"),
            vec![PiiKind::CreditCard]
        );
        assert_eq!(
            pii_in_text("server 192.168.10.20 is down"),
            vec![PiiKind::IpAddress]
        );
        assert!(pii_in_text("order 12345 shipped on 2024-01-02, total 199.90").is_empty());
        let mut c = clean();
        for i in 0..10 {
            c.push(conv(
                3000 + i,
                &format!("p{i}"),
                &format!("my email is user{i}@corp.com, cancel it"),
                None,
            ));
        }
        let a = audit(c, &[]);
        let t4 = a.findings.iter().find(|f| f.id == "T4").expect("T4");
        assert_eq!(t4.severity, Severity::Red);
        assert!(
            !t4.evidence.contains("@corp.com"),
            "evidence must not echo the PII"
        );
    }

    #[test]
    fn t5_malformed_calls() {
        let mut c = clean();
        c.push(conv(
            4001,
            "m1",
            "weather?",
            Some(r#"{"name":"get_weather","arguments":"{city"}"#),
        ));
        c.push(conv(
            4002,
            "m2",
            "weather?",
            Some(r#"{"name":"get_weather","arguments":"{\"days\":3}"}"#),
        ));
        c.push(conv(
            4003,
            "m3",
            "weather?",
            Some(r#"{"name":"get_weather","arguments":"{\"city\":5}"}"#),
        ));
        c.push(conv(
            4004,
            "m4",
            "weather?",
            Some(r#"{"name":"launch_rocket","arguments":"{}"}"#),
        ));
        c.push(conv(
            4005,
            "m5",
            "weather?",
            Some(r#"{"name":"get_weather","arguments":"{\"city\":\"x\",\"days\":2}"}"#),
        ));
        let a = audit(c, &[]);
        let t5 = a.findings.iter().find(|f| f.id == "T5").expect("T5");
        assert!(t5.evidence.starts_with("4 of"), "{}", t5.evidence);
        for why in [
            "arguments are not JSON",
            "missing required",
            "should be string",
            "tool not declared",
        ] {
            assert!(t5.evidence.contains(why), "{why}: {}", t5.evidence);
        }
        assert!(!t5.evidence.contains("4005"));
    }

    #[test]
    fn schema_violation_rules() {
        let s = serde_json::json!({"type":"object","properties":{"n":{"type":"integer"},"u":{"enum":["c","f"]}},"required":["n"],"additionalProperties":false});
        assert_eq!(schema_violation(&s, &serde_json::json!({"n": 1, "u": "c"})), None);
        assert!(schema_violation(&s, &serde_json::json!({"n": 1.5})).is_some());
        assert!(schema_violation(&s, &serde_json::json!({"n": 1, "u": "k"})).is_some());
        assert!(schema_violation(&s, &serde_json::json!({"n": 1, "z": 0})).is_some());
        assert!(schema_violation(&s, &serde_json::json!([1])).is_some());
    }

    #[test]
    fn t6_t7_t8_and_acknowledge() {
        let few: Vec<_> = clean().into_iter().take(30).collect();
        let a = audit(few, &[]);
        assert!(a.blocked && ids(&a).contains(&"T6".into()));

        let all_weather: Vec<_> = (0..100)
            .map(|i| {
                conv(
                    i + 1,
                    &format!("w{i}"),
                    &format!("forecast {i} for city {}", i * 13),
                    Some(r#"{"name":"get_weather","arguments":"{\"city\":\"x\"}"}"#),
                )
            })
            .collect();
        let s = set(all_weather);
        let split = split_by_group(&s.conversations, 0.1, 0.2, 7);
        let a = audit_traces(&s, &split, &[], &["T6".into()]);
        let got = ids(&a);
        assert!(
            got.contains(&"T7".into()) && got.contains(&"T8".into()),
            "{got:?}"
        );
        let t6 = a.findings.iter().find(|f| f.id == "T6").unwrap();
        assert_eq!(t6.resolution, Resolution::Acknowledged);
        assert!(!a.blocked, "only yellow and acknowledged findings left");
    }

    #[test]
    fn t9_unreadable_lines() {
        let mut s = set(clean());
        s.invalid_lines = (0..50).map(|i| (i, "not JSON".to_owned())).collect();
        let split = split_by_group(&s.conversations, 0.1, 0.2, 7);
        let a = audit_traces(&s, &split, &[], &[]);
        let t9 = a.findings.iter().find(|f| f.id == "T9").unwrap();
        assert_eq!(t9.severity, Severity::Red);
    }
}
