//! Statistical profile of a dataset (`profile.json`).
//!
//! Columns are read one at a time from Parquet (columnar projection), so
//! memory stays at roughly one column even for wide datasets.

use std::path::Path;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ingest::Dataset;

/// Bump when the profile format or semantics change.
pub const PROFILE_VERSION: u32 = 1;
const TOP_K: usize = 10;
const HIST_BINS: usize = 10;
const SAMPLE_ROWS: usize = 5;

/// Semantic kind of a column, inferred from dtype and cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnKind {
    Numeric,
    Boolean,
    Categorical,
    Datetime,
    /// Long free text.
    Text,
    /// String with nearly one distinct value per row (likely an identifier).
    HighCardinality,
    /// A single distinct value (or all null).
    Constant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub version: u32,
    pub dataset_id: String,
    pub rows: u64,
    pub columns: Vec<ColumnProfile>,
    /// First rows, as JSON values, for previews.
    pub sample: Vec<serde_json::Map<String, serde_json::Value>>,
}

impl Profile {
    pub fn column(&self, name: &str) -> Option<&ColumnProfile> {
        self.columns.iter().find(|c| c.name == name)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnProfile {
    pub name: String,
    pub dtype: String,
    pub kind: ColumnKind,
    pub null_count: u64,
    pub null_fraction: f64,
    pub n_unique: u64,
    /// Distinct non-null values / non-null rows.
    pub unique_ratio: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric: Option<NumericStats>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_values: Vec<ValueCount>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_length: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_time: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_time: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub p01: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p99: f64,
    /// `HIST_BINS` equal-width bins over [min, max].
    pub histogram: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueCount {
    pub value: String,
    pub count: u64,
}

/// Profiles a dataset and writes `profile.json` next to it (cached).
pub fn profile_dataset(ds: &Dataset) -> Result<Profile> {
    let path = ds.dir.join("profile.json");
    if let Ok(text) = std::fs::read_to_string(&path)
        && let Ok(p) = serde_json::from_str::<Profile>(&text)
        && p.version == PROFILE_VERSION
    {
        return Ok(p);
    }
    let profile = profile_parquet(&ds.data_path(), &ds.id)?;
    let tmp = ds.dir.join("profile.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&profile)?)?;
    std::fs::rename(tmp, path)?;
    Ok(profile)
}

pub fn profile_parquet(path: &Path, dataset_id: &str) -> Result<Profile> {
    let lf = crate::ingest::scan_parquet(path)?;
    let schema = lf.clone().collect_schema()?;
    let mut columns = Vec::with_capacity(schema.len());
    let mut rows = 0u64;
    for (name, _) in schema.iter() {
        let df = lf.clone().select([col(name.clone())]).collect()?;
        let s = df.column(name.as_str())?.as_materialized_series().clone();
        rows = s.len() as u64;
        columns.push(profile_series(&s)?);
    }
    let sample_df = lf.limit(SAMPLE_ROWS as u32).collect()?;
    Ok(Profile {
        version: PROFILE_VERSION,
        dataset_id: dataset_id.to_owned(),
        rows,
        columns,
        sample: rows_as_json(&sample_df),
    })
}

pub fn profile_series(s: &Series) -> Result<ColumnProfile> {
    let len = s.len() as u64;
    let null_count = s.null_count() as u64;
    let non_null = len - null_count;
    let n_unique_with_null = s.n_unique()? as u64;
    let n_unique = n_unique_with_null - u64::from(null_count > 0);
    let unique_ratio = if non_null == 0 {
        0.0
    } else {
        n_unique as f64 / non_null as f64
    };
    let dtype = s.dtype().clone();

    let mut p = ColumnProfile {
        name: s.name().to_string(),
        dtype: dtype.to_string(),
        kind: ColumnKind::Constant,
        null_count,
        null_fraction: if len == 0 {
            0.0
        } else {
            null_count as f64 / len as f64
        },
        n_unique,
        unique_ratio,
        numeric: None,
        top_values: Vec::new(),
        mean_length: None,
        min_time: None,
        max_time: None,
    };

    if n_unique <= 1 {
        p.top_values = top_values(s)?;
        return Ok(p);
    }

    if dtype.is_bool() {
        p.kind = ColumnKind::Boolean;
        p.top_values = top_values(s)?;
    } else if dtype.is_primitive_numeric() {
        p.kind = ColumnKind::Numeric;
        p.numeric = numeric_stats(s)?;
    } else if dtype.is_temporal() {
        p.kind = ColumnKind::Datetime;
        let as_str = s.cast(&DataType::String)?;
        let sorted = as_str.drop_nulls().sort(SortOptions::default())?;
        let ca = sorted.str()?;
        p.min_time = ca.get(0).map(str::to_owned);
        p.max_time = ca.get(ca.len().saturating_sub(1)).map(str::to_owned);
    } else if dtype.is_string() || dtype.is_categorical() {
        let st = s.cast(&DataType::String)?;
        let ca = st.str()?;
        let (total, count) = ca
            .iter()
            .flatten()
            .fold((0usize, 0usize), |(t, c), v| (t + v.chars().count(), c + 1));
        let mean_len = if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        };
        p.mean_length = Some(mean_len);
        p.top_values = top_values(&st)?;
        p.kind = classify_string(n_unique, non_null, mean_len);
    } else {
        // Nested or unusual types: treat as text so they are not used as features.
        p.kind = ColumnKind::Text;
    }
    Ok(p)
}

/// Strings: categorical when the vocabulary is small relative to the rows.
fn classify_string(n_unique: u64, non_null: u64, mean_len: f64) -> ColumnKind {
    let ratio = if non_null == 0 {
        0.0
    } else {
        n_unique as f64 / non_null as f64
    };
    if mean_len >= 40.0 && ratio > 0.5 {
        ColumnKind::Text
    } else if ratio > 0.9 && non_null >= 20 {
        ColumnKind::HighCardinality
    } else if n_unique <= 100 || ratio <= 0.05 {
        ColumnKind::Categorical
    } else if mean_len >= 40.0 {
        ColumnKind::Text
    } else {
        ColumnKind::HighCardinality
    }
}

fn numeric_stats(s: &Series) -> Result<Option<NumericStats>> {
    let f = s.cast(&DataType::Float64)?;
    let ca = f.f64()?;
    let (Some(min), Some(max)) = (ca.min(), ca.max()) else {
        return Ok(None);
    };
    let q = |x: f64| {
        ca.quantile(x, QuantileMethod::Linear)
            .ok()
            .flatten()
            .unwrap_or(f64::NAN)
    };
    let mut histogram = vec![0u64; HIST_BINS];
    let width = (max - min) / HIST_BINS as f64;
    for v in ca.iter().flatten() {
        let idx = if width > 0.0 {
            (((v - min) / width) as usize).min(HIST_BINS - 1)
        } else {
            0
        };
        histogram[idx] += 1;
    }
    Ok(Some(NumericStats {
        min,
        max,
        mean: ca.mean().unwrap_or(f64::NAN),
        std: ca.std(1).unwrap_or(0.0),
        p01: q(0.01),
        p25: q(0.25),
        p50: q(0.5),
        p75: q(0.75),
        p99: q(0.99),
        histogram,
    }))
}

fn top_values(s: &Series) -> Result<Vec<ValueCount>> {
    let st = s.cast(&DataType::String)?.drop_nulls();
    let counts = st.value_counts(true, false, "count".into(), false)?;
    let values = counts.column(st.name().as_str())?.str()?.clone();
    let n = counts.column("count")?.cast(&DataType::UInt64)?;
    let n = n.u64()?;
    Ok(values
        .iter()
        .zip(n.iter())
        .take(TOP_K)
        .filter_map(|(v, c)| {
            Some(ValueCount {
                value: v?.to_owned(),
                count: c?,
            })
        })
        .collect())
}

fn rows_as_json(df: &DataFrame) -> Vec<serde_json::Map<String, serde_json::Value>> {
    let mut rows = vec![serde_json::Map::new(); df.height()];
    for c in df.columns() {
        let s = c.as_materialized_series();
        for (i, row) in rows.iter_mut().enumerate() {
            let v = match s.get(i) {
                Ok(AnyValue::Null) | Err(_) => serde_json::Value::Null,
                Ok(AnyValue::Boolean(b)) => b.into(),
                Ok(av) if s.dtype().is_primitive_numeric() => {
                    av.extract::<f64>().map_or(serde_json::Value::Null, |x| {
                        serde_json::Number::from_f64(x).map_or(serde_json::Value::Null, Into::into)
                    })
                }
                Ok(av) => av.str_value().to_string().into(),
            };
            row.insert(s.name().to_string(), v);
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn kinds_and_stats() {
        let n = 200;
        let df = df!(
            "num" => (0..n).map(|i| i as f64).collect::<Vec<_>>(),
            "flag" => (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>(),
            "plan" => (0..n).map(|i| ["basic", "pro", "team"][i % 3]).collect::<Vec<_>>(),
            "user_id" => (0..n).map(|i| format!("u{i:05}")).collect::<Vec<_>>(),
            "same" => vec![7i64; n],
            "note" => (0..n).map(|i| format!("customer {i} wrote a long complaint about the delivery")).collect::<Vec<_>>(),
        )
        .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("d.parquet");
        let mut f = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(&mut f).finish(&mut df.clone()).unwrap();

        let p = profile_parquet(&path, "x").unwrap();
        assert_eq!(p.rows, n as u64);
        let kind = |c: &str| p.column(c).unwrap().kind;
        assert_eq!(kind("num"), ColumnKind::Numeric);
        assert_eq!(kind("flag"), ColumnKind::Boolean);
        assert_eq!(kind("plan"), ColumnKind::Categorical);
        assert_eq!(kind("user_id"), ColumnKind::HighCardinality);
        assert_eq!(kind("same"), ColumnKind::Constant);
        assert_eq!(kind("note"), ColumnKind::Text);

        let num = p.column("num").unwrap().numeric.clone().unwrap();
        assert_eq!((num.min, num.max), (0.0, 199.0));
        assert!((num.p50 - 99.5).abs() < 1e-9);
        assert_eq!(num.histogram.iter().sum::<u64>(), n as u64);
        assert_eq!(p.column("plan").unwrap().top_values.len(), 3);
        assert_eq!(p.sample.len(), SAMPLE_ROWS);
    }

    #[test]
    fn nulls_are_counted() {
        let s = Series::new("x".into(), &[Some(1.0), None, Some(3.0), None]);
        let p = profile_series(&s).unwrap();
        assert_eq!(p.null_count, 2);
        assert_eq!(p.n_unique, 2);
        assert!((p.null_fraction - 0.5).abs() < 1e-12);
    }
}
