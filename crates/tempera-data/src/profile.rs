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
pub const PROFILE_VERSION: u32 = 2;
const TOP_K: usize = 10;
const HIST_BINS: usize = 10;
const SAMPLE_ROWS: usize = 5;
/// Above this many rows, distribution details come from a uniform sample.
const EXACT_LIMIT: u64 = 2_000_000;
const SAMPLE_LIMIT: u64 = 1_000_000;

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
    /// For large datasets: rows in the uniform sample used for quantiles, histograms
    /// and top values. Counts, min/max/mean/std are always exact; distinct counts
    /// are approximate (HyperLogLog) when this is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampled_rows: Option<u64>,
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
    let rows = count_rows(&lf)?;
    let sample_df = lf.clone().limit(SAMPLE_ROWS as u32).collect()?;
    let sample = rows_as_json(&sample_df);
    if rows > EXACT_LIMIT {
        return profile_large(lf, &schema, rows, dataset_id, sample);
    }
    let mut columns = Vec::with_capacity(schema.len());
    for (name, _) in schema.iter() {
        let df = lf.clone().select([col(name.clone())]).collect()?;
        let s = df.column(name.as_str())?.as_materialized_series().clone();
        columns.push(profile_series(&s)?);
    }
    Ok(Profile {
        version: PROFILE_VERSION,
        dataset_id: dataset_id.to_owned(),
        rows,
        columns,
        sample,
        sampled_rows: None,
    })
}

fn collect_streaming(lf: LazyFrame) -> Result<DataFrame> {
    // A plain select/filter always yields a single frame.
    Ok(lf.collect_with_engine(Engine::Streaming)?.unwrap_single())
}

fn count_rows(lf: &LazyFrame) -> Result<u64> {
    let df = lf.clone().select([len().alias("n")]).collect()?;
    Ok(df
        .column("n")?
        .cast(&DataType::UInt64)?
        .u64()?
        .get(0)
        .unwrap_or(0))
}

/// Exact aggregates over all rows (streaming), distribution details from a uniform sample.
fn profile_large(
    lf: LazyFrame,
    schema: &Schema,
    rows: u64,
    dataset_id: &str,
    sample: Vec<serde_json::Map<String, serde_json::Value>>,
) -> Result<Profile> {
    // One streaming query per column keeps memory at a few batches, even if an
    // aggregation falls back to the in-memory engine for that column.
    let mut full: Vec<DataFrame> = Vec::with_capacity(schema.len());
    for (name, dtype) in schema.iter() {
        let c = || col(name.clone());
        let mut aggs = vec![
            c().null_count()
                .cast(DataType::UInt64)
                .alias(format!("{name}::nulls")),
            c().approx_n_unique()
                .cast(DataType::UInt64)
                .alias(format!("{name}::distinct")),
        ];
        if dtype.is_primitive_numeric() {
            let f = || c().cast(DataType::Float64);
            aggs.push(f().min().alias(format!("{name}::min")));
            aggs.push(f().max().alias(format!("{name}::max")));
            aggs.push(f().mean().alias(format!("{name}::mean")));
            aggs.push(f().std(1).alias(format!("{name}::std")));
        } else if dtype.is_temporal() {
            aggs.push(c().min().cast(DataType::String).alias(format!("{name}::tmin")));
            aggs.push(c().max().cast(DataType::String).alias(format!("{name}::tmax")));
        }
        full.push(collect_streaming(lf.clone().select(aggs))?);
    }
    let find = |k: &str| full.iter().find_map(|df| df.column(k).ok().cloned());
    let get_u = |k: &str| -> Option<u64> { find(k)?.u64().ok()?.get(0) };
    let get_f = |k: &str| -> Option<f64> { find(k)?.f64().ok()?.get(0) };
    let get_s = |k: &str| -> Option<String> { find(k)?.str().ok()?.get(0).map(str::to_owned) };

    // Uniform sample: every `step`-th row, filtered while streaming.
    let step = rows.div_ceil(SAMPLE_LIMIT).max(1);
    let sampled = collect_streaming(
        lf.with_row_index("__row__", None)
            .filter((col("__row__").cast(DataType::UInt64) % lit(step)).eq(lit(0u64)))
            .drop(by_name(["__row__"], true, false)),
    )?;
    let mut columns = Vec::with_capacity(schema.len());
    for (name, dtype) in schema.iter() {
        let s = sampled.column(name.as_str())?.as_materialized_series().clone();
        let mut p = profile_series(&s)?;
        let nulls = get_u(&format!("{name}::nulls")).unwrap_or(p.null_count);
        let non_null = rows - nulls.min(rows);
        let distinct = get_u(&format!("{name}::distinct"))
            .map(|d| d.saturating_sub(u64::from(nulls > 0)).min(non_null))
            .unwrap_or(p.n_unique);
        p.null_count = nulls;
        p.null_fraction = nulls as f64 / rows as f64;
        p.n_unique = distinct;
        p.unique_ratio = if non_null == 0 {
            0.0
        } else {
            distinct as f64 / non_null as f64
        };
        if distinct <= 1 {
            p.kind = ColumnKind::Constant;
        } else if dtype.is_primitive_numeric() {
            p.kind = ColumnKind::Numeric;
            if let Some(n) = p.numeric.as_mut() {
                n.min = get_f(&format!("{name}::min")).unwrap_or(n.min);
                n.max = get_f(&format!("{name}::max")).unwrap_or(n.max);
                n.mean = get_f(&format!("{name}::mean")).unwrap_or(n.mean);
                n.std = get_f(&format!("{name}::std")).unwrap_or(n.std);
            }
        } else if dtype.is_temporal() {
            p.kind = ColumnKind::Datetime;
            p.min_time = get_s(&format!("{name}::tmin")).or(p.min_time);
            p.max_time = get_s(&format!("{name}::tmax")).or(p.max_time);
        } else if dtype.is_string() || dtype.is_categorical() {
            p.kind = classify_string(distinct, non_null, p.mean_length.unwrap_or(0.0));
        }
        columns.push(p);
    }
    Ok(Profile {
        version: PROFILE_VERSION,
        dataset_id: dataset_id.to_owned(),
        rows,
        columns,
        sample,
        sampled_rows: Some(sampled.height() as u64),
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
    fn large_datasets_use_exact_aggregates_and_a_sample() {
        let n = EXACT_LIMIT as usize + 100_000;
        let df = df!(
            "x" => (0..n).map(|i| if i % 10 == 0 { None } else { Some(i as f64) }).collect::<Vec<_>>(),
            "plan" => (0..n).map(|i| ["a", "b", "c"][i % 3]).collect::<Vec<_>>(),
        )
        .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("big.parquet");
        ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .finish(&mut df.clone())
            .unwrap();
        let p = profile_parquet(&path, "big").unwrap();
        assert_eq!(p.rows, n as u64);
        assert!(p.sampled_rows.unwrap() <= SAMPLE_LIMIT);
        let x = p.column("x").unwrap();
        assert_eq!(x.null_count, (n as u64).div_ceil(10));
        let num = x.numeric.as_ref().unwrap();
        assert_eq!((num.min, num.max), (1.0, (n - 1) as f64));
        let plan = p.column("plan").unwrap();
        assert_eq!((plan.kind, plan.n_unique), (ColumnKind::Categorical, 3));
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
