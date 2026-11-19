//! A11 personal data (LGPD/GDPR) and A9 proxies of sensitive attributes (#33).

use std::sync::LazyLock;

use polars::prelude::*;
use regex::Regex;
use tempera_core::spec::JobSpec;
use tempera_data::ingest::Dataset;
use tempera_data::profile::{ColumnKind, Profile};
use tempera_data::split::{FeatureKind, FeaturePlan};

use crate::{Finding, Resolution, Result, Severity};

/// Values sampled per string column.
const SAMPLE: u32 = 2000;
/// Share of sampled values that must match for a column to be flagged.
const MIN_SHARE: f64 = 0.3;
/// Single-feature predictiveness of a sensitive attribute that makes a proxy.
const PROXY: f64 = 0.8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiKind {
    Email,
    Phone,
    Cpf,
    Cnpj,
    Ssn,
    CreditCard,
    IpAddress,
}

impl PiiKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Email => "e-mail addresses",
            Self::Phone => "phone numbers",
            Self::Cpf => "CPF numbers (Brazil)",
            Self::Cnpj => "CNPJ numbers (Brazil)",
            Self::Ssn => "US Social Security numbers",
            Self::CreditCard => "payment card numbers",
            Self::IpAddress => "IP addresses",
        }
    }
}

static EMAIL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$").unwrap_or_else(|_| unreachable!()));
static PHONE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\+?[\d\s().-]{8,20}$").unwrap_or_else(|_| unreachable!()));
static SSN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d{3}-\d{2}-\d{4}$").unwrap_or_else(|_| unreachable!()));
static IPV4: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$").unwrap_or_else(|_| unreachable!())
});

fn digits(s: &str) -> Vec<u32> {
    s.chars().filter_map(|c| c.to_digit(10)).collect()
}

/// CPF with both check digits (rejects 000.000.000-00 style repeats).
pub fn is_cpf(s: &str) -> bool {
    let d = digits(s);
    if d.len() != 11
        || d.iter().all(|&x| x == d[0])
        || s.chars().any(|c| !c.is_ascii_digit() && !".- ".contains(c))
    {
        return false;
    }
    let check = |n: usize| {
        let sum: u32 = (0..n).map(|i| d[i] * (n as u32 + 1 - i as u32)).sum();
        let r = (sum * 10) % 11;
        if r == 10 { 0 } else { r }
    };
    check(9) == d[9] && check(10) == d[10]
}

/// CNPJ with both check digits.
pub fn is_cnpj(s: &str) -> bool {
    let d = digits(s);
    if d.len() != 14
        || d.iter().all(|&x| x == d[0])
        || s.chars().any(|c| !c.is_ascii_digit() && !"./- ".contains(c))
    {
        return false;
    }
    let check = |weights: &[u32]| {
        let sum: u32 = weights.iter().zip(&d).map(|(w, x)| w * x).sum();
        let r = sum % 11;
        if r < 2 { 0 } else { 11 - r }
    };
    check(&[5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]) == d[12]
        && check(&[6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]) == d[13]
}

/// Payment card: 13–19 digits passing the Luhn check.
pub fn is_card(s: &str) -> bool {
    let d = digits(s);
    if !(13..=19).contains(&d.len()) || s.chars().any(|c| !c.is_ascii_digit() && c != ' ' && c != '-') {
        return false;
    }
    let sum: u32 = d
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &x)| {
            if i % 2 == 1 {
                if x * 2 > 9 { x * 2 - 9 } else { x * 2 }
            } else {
                x
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

fn is_ip(s: &str) -> bool {
    IPV4.captures(s)
        .is_some_and(|c| (1..=4).all(|i| c[i].parse::<u16>().is_ok_and(|n| n <= 255)))
}

fn is_phone(s: &str) -> bool {
    let n = digits(s).len();
    let looks_like_date =
        s.len() == 10 && s.as_bytes().get(4) == Some(&b'-') && s.as_bytes().get(7) == Some(&b'-');
    PHONE.is_match(s)
        && !looks_like_date
        && ((10..=15).contains(&n) || (s.starts_with('+') && n >= 8))
        && s.chars().any(|c| " ()-+".contains(c))
}

/// Classifies one value (most specific first).
pub fn classify(v: &str) -> Option<PiiKind> {
    let v = v.trim();
    if EMAIL.is_match(v) {
        Some(PiiKind::Email)
    } else if is_cpf(v) {
        Some(PiiKind::Cpf)
    } else if is_cnpj(v) {
        Some(PiiKind::Cnpj)
    } else if SSN.is_match(v) {
        Some(PiiKind::Ssn)
    } else if is_card(v) {
        Some(PiiKind::CreditCard)
    } else if is_ip(v) {
        Some(PiiKind::IpAddress)
    } else if is_phone(v) {
        Some(PiiKind::Phone)
    } else {
        None
    }
}

/// A11: scans string-like columns for personal data.
pub fn personal_data(ds: &Dataset, profile: &Profile) -> Result<Vec<Finding>> {
    let mut out = Vec::new();
    for c in &profile.columns {
        let stringy = c.dtype.starts_with("str") || c.dtype.starts_with('i');
        if !stringy || c.kind == ColumnKind::Constant {
            continue;
        }
        let df = ds
            .scan()?
            .select([col(c.name.as_str()).cast(DataType::String)])
            .filter(col(c.name.as_str()).is_not_null())
            .limit(SAMPLE)
            .collect()?;
        let values: Vec<String> = df
            .column(c.name.as_str())?
            .str()?
            .iter()
            .flatten()
            .map(str::to_owned)
            .collect();
        if values.len() < 10 {
            continue;
        }
        let mut counts: Vec<(PiiKind, usize)> = Vec::new();
        for k in values.iter().filter_map(|v| classify(v)) {
            match counts.iter_mut().find(|(kind, _)| *kind == k) {
                Some(e) => e.1 += 1,
                None => counts.push((k, 1)),
            }
        }
        let Some((kind, n)) = counts.iter().max_by_key(|(_, n)| *n).copied() else {
            continue;
        };
        let share = n as f64 / values.len() as f64;
        if share >= MIN_SHARE {
            out.push(Finding {
                id: format!("A11:{}", c.name),
                check: "A11".into(),
                severity: Severity::Yellow,
                column: Some(c.name.clone()),
                title: "Personal data".into(),
                evidence: format!("{:.0}% of sampled values in `{}` look like {}.", share * 100.0, c.name, kind.label()),
                suggestion: format!(
                    "Personal data is rarely a useful feature and raises LGPD/GDPR obligations. Add `{}` to `data.drop` or mask it upstream; acknowledge `A11:{}` if you must keep it.",
                    c.name, c.name
                ),
                resolution: Resolution::Unresolved,
            });
        }
    }
    Ok(out)
}

/// A9: features that predict a sensitive attribute almost on their own.
pub fn sensitive_proxies(ds: &Dataset, spec: &JobSpec, plan: &FeaturePlan) -> Result<Vec<Finding>> {
    let mut out = Vec::new();
    for s in &spec.data.sensitive {
        let mut cols: Vec<Expr> = plan
            .features
            .iter()
            .filter(|f| &f.name != s)
            .map(|f| col(f.name.as_str()))
            .collect();
        cols.push(col(s.as_str()).cast(DataType::String));
        let df = match ds
            .scan()?
            .select(cols)
            .filter(col(s.as_str()).is_not_null())
            .limit(200_000)
            .collect()
        {
            Ok(df) => df,
            Err(_) => continue, // sensitive column missing: spec validation reports it elsewhere
        };
        let n_classes = df.column(s.as_str())?.n_unique()?;
        for f in plan.features.iter().filter(|f| &f.name != s) {
            let score = if n_classes == 2 {
                crate::binary_auc(&df, &f.name, f.kind, s)?
            } else if f.kind == FeatureKind::Numeric {
                None
            } else {
                crate::multiclass_purity(&df, &f.name, s)?
            };
            if let Some(score) = score.filter(|v| *v >= PROXY) {
                out.push(Finding {
                    id: format!("A9:{}", f.name),
                    check: "A9".into(),
                    severity: Severity::Yellow,
                    column: Some(f.name.clone()),
                    title: "Proxy of a sensitive attribute".into(),
                    evidence: format!("`{}` alone predicts the sensitive attribute `{s}` (score {score:.2}).", f.name),
                    suggestion: format!(
                        "Dropping `{s}` is not enough while `{}` encodes it. Review the fairness report; consider dropping or coarsening `{}`.",
                        f.name, f.name
                    ),
                    resolution: Resolution::Unresolved,
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validators() {
        assert!(is_cpf("529.982.247-25"));
        assert!(is_cpf("52998224725"));
        assert!(!is_cpf("529.982.247-26"));
        assert!(!is_cpf("111.111.111-11"));
        assert!(is_cnpj("11.222.333/0001-81"));
        assert!(!is_cnpj("11.222.333/0001-82"));
        assert!(is_card("4111 1111 1111 1111"));
        assert!(!is_card("4111 1111 1111 1112"));
        assert_eq!(classify("ana.silva+x@example.com.br"), Some(PiiKind::Email));
        assert_eq!(classify("+55 (11) 98765-4321"), Some(PiiKind::Phone));
        assert_eq!(classify("123-45-6789"), Some(PiiKind::Ssn));
        assert_eq!(classify("192.168.0.42"), Some(PiiKind::IpAddress));
        assert_eq!(classify("999.1.1.1"), None);
        // plain numbers and ordinary text are not personal data
        assert_eq!(classify("12345"), None);
        assert_eq!(classify("basic"), None);
        assert_eq!(classify("2024-01-02"), None);
    }
}
