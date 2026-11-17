//! Trivial baselines computed in-process: the floor every model must beat.

use std::path::Path;

use polars::prelude::*;
use tempera_data::ingest::scan_parquet;
use tempera_data::split::TARGET;

/// Majority class: predicts the most frequent train label, with train class
/// frequencies as probabilities.
pub fn majority(train: &Path, classes: &[String], input: &Path, output: &Path) -> anyhow::Result<()> {
    let y = scan_parquet(train)?.select([col(TARGET)]).collect()?;
    let labels = y.column(TARGET)?.cast(&DataType::String)?;
    let total = labels.len().max(1) as f64;
    let freq: Vec<f64> = classes
        .iter()
        .map(|c| {
            labels
                .str()
                .map(|s| s.iter().filter(|v| *v == Some(c.as_str())).count())
                .unwrap_or(0) as f64
                / total
        })
        .collect();
    let best = freq
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i);
    let n = rows(input)?;
    let mut cols = vec![Column::new("prediction".into(), vec![classes[best].as_str(); n])];
    for (c, f) in classes.iter().zip(&freq) {
        cols.push(Column::new(format!("proba__{c}").into(), vec![*f; n]));
    }
    write(output, DataFrame::new_infer_height(cols)?)
}

/// Mean of the train target.
pub fn mean(train: &Path, input: &Path, output: &Path) -> anyhow::Result<()> {
    let y = scan_parquet(train)?
        .select([col(TARGET).cast(DataType::Float64).mean()])
        .collect()?;
    let m = y.column(TARGET)?.f64()?.get(0).unwrap_or(0.0);
    let n = rows(input)?;
    write(
        output,
        DataFrame::new_infer_height(vec![Column::new("prediction".into(), vec![m; n])])?,
    )
}

fn rows(p: &Path) -> anyhow::Result<usize> {
    let df = scan_parquet(p)?.select([len().alias("n")]).collect()?;
    Ok(df
        .column("n")?
        .cast(&DataType::UInt64)?
        .u64()?
        .get(0)
        .unwrap_or(0) as usize)
}

fn write(p: &Path, mut df: DataFrame) -> anyhow::Result<()> {
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir)?;
    }
    ParquetWriter::new(std::fs::File::create(p)?).finish(&mut df)?;
    Ok(())
}
