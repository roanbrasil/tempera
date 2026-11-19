//! Algorithms: linear models use this crate's own solvers (`solve.rs`), ridge
//! and trees use linfa. Everything is stored as plain parameters so a saved
//! model is independent of any library's internal types.

use linfa::prelude::*;
use linfa_elasticnet::ElasticNet;
use linfa_trees::{DecisionTree, SplitQuality, TreeNode};
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tempera_core::spec::TaskType;

use crate::data::Predictions;
use crate::error::NativeError;
use crate::solve;

/// Algorithms implemented by this backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    LinearRegression,
    Ridge,
    LogisticRegression,
    DecisionTree,
}

impl Algorithm {
    pub fn parse(name: &str) -> Result<Self, NativeError> {
        Ok(match name {
            "linear_regression" => Self::LinearRegression,
            "ridge" => Self::Ridge,
            "logistic_regression" => Self::LogisticRegression,
            "decision_tree" => Self::DecisionTree,
            other => return Err(NativeError::UnknownAlgorithm(other.to_owned())),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::LinearRegression => "linear_regression",
            Self::Ridge => "ridge",
            Self::LogisticRegression => "logistic_regression",
            Self::DecisionTree => "decision_tree",
        }
    }

    /// Whether this backend can run the algorithm on `task`.
    pub fn supports(self, task: TaskType) -> bool {
        use TaskType::*;
        match self {
            Self::LinearRegression | Self::Ridge => task == Regression,
            // linfa-trees 0.8 only ships a classification tree.
            Self::LogisticRegression | Self::DecisionTree => {
                matches!(task, BinaryClassification | MulticlassClassification)
            }
        }
    }

    pub(crate) fn check(self, task: TaskType) -> Result<(), NativeError> {
        if self.supports(task) {
            Ok(())
        } else {
            Err(NativeError::UnsupportedTask {
                algorithm: self.name().to_owned(),
                task: serde_json::to_value(task)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_else(|| format!("{task:?}")),
            })
        }
    }
}

/// A flattened decision-tree node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum FlatNode {
    /// Goes `left` when `x[feature] < threshold`, else `right` (linfa's rule).
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    Leaf {
        class: usize,
        /// Train class frequencies in the leaf, manifest `classes` order.
        proba: Vec<f64>,
    },
}

/// A fitted model, as saved in `native.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FittedModel {
    /// `y = x·coef + intercept` (linear regression and ridge).
    Linear {
        coef: Vec<f64>,
        intercept: f64,
    },
    /// `p(pos) = sigmoid(x·coef + intercept)`; class indices into `classes`.
    BinaryLogistic {
        coef: Vec<f64>,
        intercept: f64,
        pos: usize,
        neg: usize,
        n_classes: usize,
    },
    /// Softmax over `x·coef[k] + intercept[k]` for every fitted class `classes[k]`.
    MultiLogistic {
        /// One row per fitted class.
        coef: Vec<Vec<f64>>,
        intercept: Vec<f64>,
        classes: Vec<usize>,
        n_classes: usize,
    },
    Tree {
        nodes: Vec<FlatNode>,
    },
}

/// What fitting produced besides the model.
#[derive(Debug)]
pub(crate) struct FitOutcome {
    pub model: FittedModel,
    /// Importance per design-matrix column.
    pub column_importance: Vec<f64>,
}

/// Reads a typed hyper-parameter, with a default when missing or null.
fn param<T: serde::de::DeserializeOwned>(
    params: &Map<String, Value>,
    key: &str,
    default: T,
) -> Result<T, NativeError> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| NativeError::Manifest(format!("param `{key}`: {e}"))),
    }
}

fn training(e: impl std::fmt::Display) -> NativeError {
    NativeError::Training(e.to_string())
}

pub(crate) fn fit_regression(
    algorithm: Algorithm,
    x: &Array2<f64>,
    y: &[f64],
    params: &Map<String, Value>,
) -> Result<FitOutcome, NativeError> {
    let ds = Dataset::new(x.clone(), Array1::from(y.to_vec()));
    let (coef, intercept) = match algorithm {
        Algorithm::LinearRegression => {
            let fit_intercept = param(params, "fit_intercept", true)?;
            solve::least_squares(x, y, 0.0, fit_intercept)
                .ok_or_else(|| training("least squares did not converge (non-finite data?)"))?
        }
        Algorithm::Ridge => {
            // sklearn-style `alpha`: minimizes ||y - Xw||² + alpha·||w||². linfa's
            // elastic net minimizes 1/(2n)·||y - Xw||² + penalty/2·||w||² when
            // l1_ratio = 0, so penalty = alpha / n.
            let alpha: f64 = param(params, "alpha", 1.0)?;
            if !(alpha.is_finite() && alpha >= 0.0) {
                return Err(NativeError::Manifest("param `alpha` must be >= 0".into()));
            }
            let n = x.nrows().max(1) as f64;
            let m = ElasticNet::params()
                .penalty(alpha / n)
                .l1_ratio(0.0)
                .with_intercept(param(params, "fit_intercept", true)?)
                .max_iterations(param(params, "max_iterations", 1000u32)?)
                .tolerance(param(params, "tolerance", 1e-6)?)
                .fit(&ds)
                .map_err(training)?;
            (m.hyperplane().to_vec(), m.intercept())
        }
        _ => return Err(NativeError::Model("not a regression algorithm".into())),
    };
    let importance = coef.iter().map(|c| c.abs()).collect();
    Ok(FitOutcome {
        model: FittedModel::Linear { coef, intercept },
        column_importance: importance,
    })
}

pub(crate) fn fit_classification(
    algorithm: Algorithm,
    x: &Array2<f64>,
    y: &[usize],
    n_classes: usize,
    params: &Map<String, Value>,
) -> Result<FitOutcome, NativeError> {
    let ds = Dataset::new(x.clone(), Array1::from(y.to_vec()));
    let mut seen: Vec<usize> = y.to_vec();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() < 2 {
        return Err(NativeError::Data(
            "train contains fewer than two distinct classes".into(),
        ));
    }
    match algorithm {
        Algorithm::LogisticRegression => {
            let alpha: f64 = param(params, "alpha", 1.0)?;
            let max_iter: u64 = param(params, "max_iterations", 100)?;
            let tol: f64 = param(params, "gradient_tolerance", 1e-4)?;
            if !(alpha.is_finite() && alpha >= 0.0) {
                return Err(NativeError::Manifest("param `alpha` must be >= 0".into()));
            }
            if seen.len() == 2 {
                let (neg, pos) = (seen[0], seen[1]);
                let yb: Vec<bool> = y.iter().map(|&c| c == pos).collect();
                let (coef, intercept) = solve::logistic_binary(x, &yb, alpha, max_iter, tol);
                let importance = coef.iter().map(|c| c.abs()).collect();
                Ok(FitOutcome {
                    model: FittedModel::BinaryLogistic {
                        coef,
                        intercept,
                        pos,
                        neg,
                        n_classes,
                    },
                    column_importance: importance,
                })
            } else {
                // Fit over the classes present in train, indexed 0..k.
                let index: std::collections::HashMap<usize, usize> =
                    seen.iter().enumerate().map(|(i, &c)| (c, i)).collect();
                let yk: Vec<usize> = y.iter().map(|c| index[c]).collect();
                let (coef, intercept) = solve::logistic_multinomial(x, &yk, seen.len(), alpha, max_iter, tol);
                let importance = (0..x.ncols())
                    .map(|j| coef.iter().map(|w| w[j].abs()).sum::<f64>() / coef.len() as f64)
                    .collect();
                Ok(FitOutcome {
                    model: FittedModel::MultiLogistic {
                        coef,
                        intercept,
                        classes: seen,
                        n_classes,
                    },
                    column_importance: importance,
                })
            }
        }
        Algorithm::DecisionTree => {
            let quality = match param(params, "criterion", "gini".to_owned())?.as_str() {
                "gini" => SplitQuality::Gini,
                "entropy" => SplitQuality::Entropy,
                other => {
                    return Err(NativeError::Manifest(format!(
                        "param `criterion` must be `gini` or `entropy`, got `{other}`"
                    )));
                }
            };
            let tree = DecisionTree::params()
                .split_quality(quality)
                .max_depth(param(params, "max_depth", None::<usize>)?)
                .min_weight_split(param(params, "min_weight_split", 2.0f32)?)
                .min_weight_leaf(param(params, "min_weight_leaf", 1.0f32)?)
                .min_impurity_decrease(param(params, "min_impurity_decrease", 1e-5)?)
                .fit(&ds)
                .map_err(training)?;
            let mut nodes = Vec::new();
            flatten(tree.root_node(), &mut nodes)?;
            // Leaf class frequencies from the train rows reaching each leaf.
            let mut counts: Vec<Vec<f64>> = vec![vec![0.0; n_classes]; nodes.len()];
            for (row, &label) in x.rows().into_iter().zip(y) {
                counts[leaf_of(&nodes, row)][label] += 1.0;
            }
            for (node, c) in nodes.iter_mut().zip(counts) {
                if let FlatNode::Leaf { class, proba } = node {
                    let total: f64 = c.iter().sum();
                    *proba = if total > 0.0 {
                        c.iter().map(|v| v / total).collect()
                    } else {
                        (0..n_classes).map(|k| f64::from(u8::from(k == *class))).collect()
                    };
                }
            }
            let importance = tree
                .feature_importance()
                .into_iter()
                .map(|v| if v.is_finite() { v } else { 0.0 })
                .collect::<Vec<f64>>();
            let mut column_importance = vec![0.0; x.ncols()];
            for (i, v) in importance.into_iter().enumerate().take(x.ncols()) {
                column_importance[i] = v;
            }
            Ok(FitOutcome {
                model: FittedModel::Tree { nodes },
                column_importance,
            })
        }
        _ => Err(NativeError::Model("not a classification algorithm".into())),
    }
}

/// Pre-order flattening; returns the index of `node` in `out`.
fn flatten(node: &TreeNode<f64, usize>, out: &mut Vec<FlatNode>) -> Result<usize, NativeError> {
    let idx = out.len();
    if let Some(class) = node.prediction() {
        out.push(FlatNode::Leaf {
            class,
            proba: Vec::new(),
        });
        return Ok(idx);
    }
    let (feature, threshold, _) = node.split();
    out.push(FlatNode::Leaf {
        class: 0,
        proba: Vec::new(),
    });
    let children = node.children();
    let (Some(l), Some(r)) = (children[0], children[1]) else {
        return Err(NativeError::Model(
            "internal tree node without two children".into(),
        ));
    };
    let left = flatten(l, out)?;
    let right = flatten(r, out)?;
    out[idx] = FlatNode::Split {
        feature,
        threshold,
        left,
        right,
    };
    Ok(idx)
}

fn leaf_of(nodes: &[FlatNode], row: ArrayView1<'_, f64>) -> usize {
    let mut i = 0;
    loop {
        match &nodes[i] {
            FlatNode::Leaf { .. } => return i,
            FlatNode::Split {
                feature,
                threshold,
                left,
                right,
            } => i = if row[*feature] < *threshold { *left } else { *right },
        }
    }
}

fn dot(row: ArrayView1<'_, f64>, coef: &[f64]) -> f64 {
    row.iter().zip(coef).map(|(a, b)| a * b).sum()
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn argmax(v: ArrayView1<'_, f64>) -> usize {
    let mut best = 0;
    for (i, &p) in v.iter().enumerate() {
        if p > v[best] {
            best = i;
        }
    }
    best
}

impl FittedModel {
    /// Number of design-matrix columns the model expects.
    pub fn n_inputs(&self) -> Option<usize> {
        match self {
            Self::Linear { coef, .. } | Self::BinaryLogistic { coef, .. } => Some(coef.len()),
            Self::MultiLogistic { coef, .. } => coef.first().map(Vec::len),
            Self::Tree { .. } => None,
        }
    }

    pub(crate) fn predict(&self, x: &Array2<f64>) -> Result<Predictions, NativeError> {
        if let Some(p) = self.n_inputs()
            && p != x.ncols()
        {
            return Err(NativeError::Model(format!(
                "model expects {p} input columns, got {}",
                x.ncols()
            )));
        }
        let n = x.nrows();
        Ok(match self {
            Self::Linear { coef, intercept } => {
                Predictions::Regression(x.rows().into_iter().map(|r| dot(r, coef) + intercept).collect())
            }
            Self::BinaryLogistic {
                coef,
                intercept,
                pos,
                neg,
                n_classes,
            } => {
                let mut proba = Array2::zeros((n, *n_classes));
                let mut predicted = Vec::with_capacity(n);
                for (i, r) in x.rows().into_iter().enumerate() {
                    let p = sigmoid(dot(r, coef) + intercept);
                    proba[[i, *pos]] = p;
                    proba[[i, *neg]] = 1.0 - p;
                    predicted.push(if p >= 0.5 { *pos } else { *neg });
                }
                Predictions::Classification { predicted, proba }
            }
            Self::MultiLogistic {
                coef,
                intercept,
                classes,
                n_classes,
            } => {
                let mut proba = Array2::zeros((n, *n_classes));
                for (i, r) in x.rows().into_iter().enumerate() {
                    let z: Vec<f64> = coef.iter().zip(intercept).map(|(w, b)| dot(r, w) + b).collect();
                    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
                    let sum: f64 = e.iter().sum();
                    for (k, &c) in classes.iter().enumerate() {
                        proba[[i, c]] = e[k] / sum;
                    }
                }
                let predicted = proba.rows().into_iter().map(argmax).collect();
                Predictions::Classification { predicted, proba }
            }
            Self::Tree { nodes } => {
                let n_classes = nodes
                    .iter()
                    .find_map(|nd| match nd {
                        FlatNode::Leaf { proba, .. } => Some(proba.len()),
                        FlatNode::Split { .. } => None,
                    })
                    .unwrap_or(0);
                let mut proba = Array2::zeros((n, n_classes));
                let mut predicted = Vec::with_capacity(n);
                for (i, r) in x.rows().into_iter().enumerate() {
                    if let FlatNode::Leaf { class, proba: p } = &nodes[leaf_of(nodes, r)] {
                        for (k, v) in p.iter().enumerate() {
                            proba[[i, k]] = *v;
                        }
                        predicted.push(*class);
                    }
                }
                Predictions::Classification { predicted, proba }
            }
        })
    }

    /// Validates indices so a corrupted model file cannot panic in `predict`.
    pub(crate) fn validate(&self, n_classes: usize, width: usize) -> Result<(), NativeError> {
        let bad = |m: &str| Err(NativeError::Model(format!("invalid model file: {m}")));
        match self {
            Self::Linear { coef, .. } if coef.len() != width => bad("coefficient count"),
            Self::BinaryLogistic {
                pos,
                neg,
                n_classes: k,
                coef,
                ..
            } => {
                if *k != n_classes || *pos >= *k || *neg >= *k || coef.len() != width {
                    bad("binary logistic shape")
                } else {
                    Ok(())
                }
            }
            Self::MultiLogistic {
                coef,
                intercept,
                classes,
                n_classes: k,
            } => {
                if *k != n_classes
                    || coef.len() != classes.len()
                    || intercept.len() != classes.len()
                    || classes.iter().any(|c| c >= k)
                    || coef.iter().any(|w| w.len() != width)
                {
                    bad("multinomial logistic shape")
                } else {
                    Ok(())
                }
            }
            Self::Tree { nodes } => {
                if nodes.is_empty() {
                    return bad("empty tree");
                }
                for (i, nd) in nodes.iter().enumerate() {
                    match nd {
                        FlatNode::Split {
                            feature, left, right, ..
                        } => {
                            // Children come after their parent in pre-order, so walks terminate.
                            if *feature >= width
                                || *left <= i
                                || *right <= i
                                || *left >= nodes.len()
                                || *right >= nodes.len()
                            {
                                return bad("tree split");
                            }
                        }
                        FlatNode::Leaf { class, proba } => {
                            if *class >= n_classes || proba.len() != n_classes {
                                return bad("tree leaf");
                            }
                        }
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Mean squared error (regression) or log-loss (classification) on train.
pub(crate) fn train_loss(preds: &Predictions, target: &Target<'_>) -> f64 {
    match (preds, target) {
        (Predictions::Regression(p), Target::Regression(y)) => {
            p.iter().zip(*y).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / p.len().max(1) as f64
        }
        (Predictions::Classification { proba, .. }, Target::Classes(y)) => {
            let eps = 1e-15;
            y.iter()
                .enumerate()
                .map(|(i, &c)| -proba[[i, c]].clamp(eps, 1.0 - eps).ln())
                .sum::<f64>()
                / y.len().max(1) as f64
        }
        _ => f64::NAN,
    }
}

#[derive(Debug)]
pub(crate) enum Target<'a> {
    Regression(&'a [f64]),
    Classes(&'a [usize]),
}
