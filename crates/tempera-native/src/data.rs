//! Model-ready Parquet in, design matrix out, predictions Parquet back.
//!
//! Preprocessing (fitted on train, saved with the model, reused in predict):
//! - numeric: nulls/NaN imputed with the train median, then standardized
//!   (train mean / std of the imputed column; a constant column keeps std 1);
//! - boolean: 0/1, nulls imputed with the train median (i.e. the majority value);
//! - categorical: one-hot of the [`MAX_CATEGORIES`] most frequent train
//!   categories, plus an `other` column when train had more; nulls and values
//!   never seen in train encode as all zeros.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, RecordBatch, RecordBatchReader, StringArray,
};
use arrow::compute::{cast, concat_batches};
use arrow::datatypes::{DataType, Field, Schema};
use ndarray::Array2;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::error::NativeError;
use crate::manifest::{FeatureKind, FeatureSpec};

/// Column holding the target in the model-ready Parquet.
pub const TARGET_COLUMN: &str = "__target__";
/// Columns that are never features.
pub const RESERVED_COLUMNS: [&str; 3] = [TARGET_COLUMN, "__group__", "__time__"];
/// Maximum number of one-hot columns per categorical feature (plus `other`).
pub const MAX_CATEGORIES: usize = 50;

/// A Parquet file loaded into memory as one record batch.
#[derive(Debug)]
pub(crate) struct Table {
    batch: RecordBatch,
}

impl Table {
    pub(crate) fn read(path: &Path) -> Result<Self, NativeError> {
        let file = File::open(path)
            .map_err(|e| NativeError::Data(format!("cannot open `{}`: {e}", path.display())))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        let batch = concat_batches(&schema, &batches)?;
        Ok(Self { batch })
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    fn column(&self, name: &str) -> Result<&ArrayRef, NativeError> {
        self.batch
            .column_by_name(name)
            .ok_or_else(|| NativeError::Data(format!("column `{name}` not found")))
    }

    fn f64_column(&self, name: &str) -> Result<Vec<Option<f64>>, NativeError> {
        let col = self.column(name)?;
        if !col.data_type().is_numeric() {
            return Err(NativeError::Data(format!(
                "column `{name}` should be numeric, found {}",
                col.data_type()
            )));
        }
        let casted = cast(col, &DataType::Float64)?;
        let arr = downcast::<Float64Array>(&casted, name)?;
        Ok(arr.iter().map(|v| v.filter(|x| x.is_finite())).collect())
    }

    fn string_column(&self, name: &str) -> Result<Vec<Option<String>>, NativeError> {
        let col = self.column(name)?;
        let casted = cast(col, &DataType::Utf8)?;
        let arr = downcast::<StringArray>(&casted, name)?;
        Ok(arr.iter().map(|v| v.map(str::to_owned)).collect())
    }

    fn bool_column(&self, name: &str) -> Result<Vec<Option<f64>>, NativeError> {
        let col = self.column(name)?;
        if col.data_type() != &DataType::Boolean {
            return Err(NativeError::Data(format!(
                "column `{name}` should be boolean, found {}",
                col.data_type()
            )));
        }
        let arr = downcast::<BooleanArray>(col, name)?;
        Ok(arr.iter().map(|v| v.map(|b| if b { 1.0 } else { 0.0 })).collect())
    }

    /// Classification target as class indices into `classes`.
    pub(crate) fn class_target(&self, classes: &[String]) -> Result<Vec<usize>, NativeError> {
        let index: HashMap<&str, usize> = classes.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
        self.string_column(TARGET_COLUMN)?
            .into_iter()
            .enumerate()
            .map(|(row, v)| {
                let v = v.ok_or_else(|| NativeError::Data(format!("null `{TARGET_COLUMN}` at row {row}")))?;
                index
                    .get(v.as_str())
                    .copied()
                    .ok_or_else(|| NativeError::Data(format!("label `{v}` at row {row} is not in `classes`")))
            })
            .collect()
    }

    /// Regression target.
    pub(crate) fn numeric_target(&self) -> Result<Vec<f64>, NativeError> {
        self.f64_column(TARGET_COLUMN)?
            .into_iter()
            .enumerate()
            .map(|(row, v)| {
                v.ok_or_else(|| {
                    NativeError::Data(format!("null or non-finite `{TARGET_COLUMN}` at row {row}"))
                })
            })
            .collect()
    }
}

fn downcast<'a, T: 'static>(arr: &'a ArrayRef, name: &str) -> Result<&'a T, NativeError> {
    arr.as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| NativeError::Data(format!("column `{name}` has an unexpected type")))
}

/// Fitted transform of one feature column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ColumnTransform {
    Numeric {
        name: String,
        median: f64,
        mean: f64,
        std: f64,
    },
    Boolean {
        name: String,
        fill: f64,
    },
    Categorical {
        name: String,
        categories: Vec<String>,
        /// Whether an `other` column collects train categories beyond the top ones.
        other: bool,
    },
}

impl ColumnTransform {
    pub fn name(&self) -> &str {
        match self {
            Self::Numeric { name, .. } | Self::Boolean { name, .. } | Self::Categorical { name, .. } => name,
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::Numeric { .. } | Self::Boolean { .. } => 1,
            Self::Categorical {
                categories, other, ..
            } => categories.len() + usize::from(*other),
        }
    }
}

/// Preprocessing fitted on train and saved with the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preprocessor {
    pub columns: Vec<ColumnTransform>,
}

impl Preprocessor {
    pub(crate) fn fit(table: &Table, features: &[FeatureSpec]) -> Result<Self, NativeError> {
        let mut columns = Vec::with_capacity(features.len());
        for f in features {
            let name = f.name.clone();
            columns.push(match f.kind {
                FeatureKind::Numeric => {
                    let values = table.f64_column(&f.name)?;
                    let median = median(&values);
                    let imputed: Vec<f64> = values.iter().map(|v| v.unwrap_or(median)).collect();
                    let (mean, std) = mean_std(&imputed);
                    ColumnTransform::Numeric {
                        name,
                        median,
                        mean,
                        std,
                    }
                }
                FeatureKind::Boolean => {
                    let values = table.bool_column(&f.name)?;
                    ColumnTransform::Boolean {
                        name,
                        fill: median(&values),
                    }
                }
                FeatureKind::Categorical => {
                    let values = table.string_column(&f.name)?;
                    let mut counts: HashMap<&str, usize> = HashMap::new();
                    for v in values.iter().flatten() {
                        *counts.entry(v.as_str()).or_default() += 1;
                    }
                    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
                    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                    let other = ranked.len() > MAX_CATEGORIES;
                    let categories = ranked
                        .into_iter()
                        .take(MAX_CATEGORIES)
                        .map(|(c, _)| c.to_owned())
                        .collect();
                    ColumnTransform::Categorical {
                        name,
                        categories,
                        other,
                    }
                }
            });
        }
        Ok(Self { columns })
    }

    /// Number of design-matrix columns.
    pub fn width(&self) -> usize {
        self.columns.iter().map(ColumnTransform::width).sum()
    }

    /// For each design-matrix column, the index of the original feature.
    pub fn column_owner(&self) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .flat_map(|(i, c)| std::iter::repeat_n(i, c.width()))
            .collect()
    }

    pub(crate) fn transform(&self, table: &Table) -> Result<Array2<f64>, NativeError> {
        let n = table.num_rows();
        let mut x = Array2::<f64>::zeros((n, self.width()));
        let mut offset = 0;
        for col in &self.columns {
            match col {
                ColumnTransform::Numeric {
                    name,
                    median,
                    mean,
                    std,
                } => {
                    for (row, v) in table.f64_column(name)?.into_iter().enumerate() {
                        x[[row, offset]] = (v.unwrap_or(*median) - mean) / std;
                    }
                }
                ColumnTransform::Boolean { name, fill } => {
                    for (row, v) in table.bool_column(name)?.into_iter().enumerate() {
                        x[[row, offset]] = v.unwrap_or(*fill);
                    }
                }
                ColumnTransform::Categorical {
                    name,
                    categories,
                    other,
                } => {
                    let index: HashMap<&str, usize> = categories
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (c.as_str(), i))
                        .collect();
                    // With an `other` column, any non-null value outside the top
                    // categories lands there (rare train values and unseen ones alike
                    // cannot be told apart without storing every train value).
                    for (row, v) in table.string_column(name)?.into_iter().enumerate() {
                        let Some(v) = v else { continue };
                        match index.get(v.as_str()) {
                            Some(&i) => x[[row, offset + i]] = 1.0,
                            None if *other => x[[row, offset + categories.len()]] = 1.0,
                            None => {}
                        }
                    }
                }
            }
            offset += col.width();
        }
        Ok(x)
    }
}

fn median(values: &[Option<f64>]) -> f64 {
    let mut v: Vec<f64> = values.iter().flatten().copied().collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    (mean, if std > 1e-12 { std } else { 1.0 })
}

/// Model output for a set of rows.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Predictions {
    /// Predicted class index per row and probabilities (rows × classes, in
    /// manifest `classes` order).
    Classification {
        predicted: Vec<usize>,
        proba: Array2<f64>,
    },
    Regression(Vec<f64>),
}

/// Writes the predictions Parquet (Contracts §7).
pub(crate) fn write_predictions(
    path: &Path,
    preds: &Predictions,
    classes: Option<&[String]>,
) -> Result<(), NativeError> {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    match preds {
        Predictions::Regression(values) => {
            fields.push(Field::new("prediction", DataType::Float64, false));
            arrays.push(Arc::new(Float64Array::from(values.clone())));
        }
        Predictions::Classification { predicted, proba } => {
            let classes =
                classes.ok_or_else(|| NativeError::Model("classification model without classes".into()))?;
            fields.push(Field::new("prediction", DataType::Utf8, false));
            arrays.push(Arc::new(StringArray::from_iter_values(
                predicted.iter().map(|&i| classes[i].as_str()),
            )));
            for (k, class) in classes.iter().enumerate() {
                fields.push(Field::new(format!("proba__{class}"), DataType::Float64, false));
                arrays.push(Arc::new(Float64Array::from(proba.column(k).to_vec())));
            }
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_ignores_nulls() {
        assert_eq!(median(&[Some(3.0), None, Some(1.0), Some(2.0)]), 2.0);
        assert_eq!(median(&[Some(4.0), Some(1.0), Some(2.0), Some(3.0)]), 2.5);
        assert_eq!(median(&[None, None]), 0.0);
    }

    #[test]
    fn constant_column_keeps_unit_std() {
        assert_eq!(mean_std(&[5.0, 5.0, 5.0]), (5.0, 1.0));
        let (m, s) = mean_std(&[1.0, 3.0]);
        assert_eq!((m, s), (2.0, 1.0));
    }
}
