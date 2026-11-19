//! Hyperparameter search (#28): successive halving with the training-set
//! fraction as the budget (1/9 → 1/3 → 1, keep the best third each rung).
//!
//! On a single local agent trials run one after another, so the asynchronous
//! variant (ASHA) and the synchronous one make the same decisions; the
//! schedule is ASHA's. Every trial is a tracked run with its parameters; only
//! the best full-data trial per algorithm enters the leaderboard.

use serde_json::{Map, Value, json};

/// Rungs as fractions of the training rows.
pub const RUNGS: [f64; 3] = [1.0 / 9.0, 1.0 / 3.0, 1.0];
/// Keep the best `1 / REDUCTION` of the configurations at each rung.
pub const REDUCTION: usize = 3;

#[derive(Debug, Clone)]
pub enum Dist {
    LogUniform(f64, f64),
    Uniform(f64, f64),
    Int(i64, i64),
    Choice(Vec<Value>),
}

/// Search space for an algorithm on a backend; `None` when it has no tunable space.
pub fn space(algorithm: &str, backend: &str) -> Option<Vec<(&'static str, Dist)>> {
    use Dist::*;
    let s = match (algorithm, backend) {
        ("lightgbm", _) => vec![
            ("num_leaves", Int(15, 127)),
            ("learning_rate", LogUniform(0.01, 0.3)),
            ("min_child_samples", Int(5, 100)),
            ("feature_fraction", Uniform(0.5, 1.0)),
            ("lambda_l2", LogUniform(1e-3, 10.0)),
        ],
        ("xgboost", _) => vec![
            ("max_depth", Int(3, 10)),
            ("eta", LogUniform(0.01, 0.3)),
            ("min_child_weight", LogUniform(0.5, 20.0)),
            ("subsample", Uniform(0.5, 1.0)),
            ("colsample_bytree", Uniform(0.5, 1.0)),
            ("lambda", LogUniform(1e-3, 10.0)),
        ],
        ("catboost", _) => vec![
            ("depth", Int(4, 10)),
            ("learning_rate", LogUniform(0.01, 0.3)),
            ("l2_leaf_reg", LogUniform(1.0, 10.0)),
        ],
        ("random_forest" | "extra_trees", "python") => vec![
            ("n_estimators", Int(100, 400)),
            ("min_samples_leaf", Int(1, 20)),
            (
                "max_features",
                Choice(vec![json!("sqrt"), json!("log2"), json!(0.5), json!(1.0)]),
            ),
        ],
        ("hist_gradient_boosting", _) => vec![
            ("learning_rate", LogUniform(0.01, 0.3)),
            ("max_leaf_nodes", Int(15, 127)),
            ("l2_regularization", LogUniform(1e-3, 10.0)),
        ],
        ("logistic_regression", "native") | ("ridge", "native") => vec![("alpha", LogUniform(1e-4, 10.0))],
        ("logistic_regression", "python") => vec![("C", LogUniform(1e-3, 100.0))],
        ("ridge" | "lasso", "python") => vec![("alpha", LogUniform(1e-4, 10.0))],
        ("elastic_net", _) => vec![
            ("alpha", LogUniform(1e-4, 10.0)),
            ("l1_ratio", Uniform(0.05, 0.95)),
        ],
        ("decision_tree", "native") => vec![("max_depth", Int(2, 20)), ("min_weight_leaf", Int(1, 30))],
        ("knn", _) => vec![
            ("n_neighbors", Int(3, 50)),
            ("weights", Choice(vec![json!("uniform"), json!("distance")])),
        ],
        ("svm", _) => vec![("C", LogUniform(0.1, 100.0))],
        ("mlp", _) => vec![
            ("alpha", LogUniform(1e-5, 1e-1)),
            ("learning_rate_init", LogUniform(1e-4, 1e-2)),
            (
                "hidden_layer_sizes",
                Choice(vec![json!([64]), json!([128]), json!([64, 32]), json!([128, 64])]),
            ),
        ],
        _ => return None,
    };
    Some(s)
}

/// Deterministic sampling (SplitMix64), stable across versions.
#[derive(Debug)]
pub struct Sampler(u64);

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x5DEE_CE66_D1CE_4E5B)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn sample(&mut self, space: &[(&'static str, Dist)]) -> Map<String, Value> {
        let mut m = Map::new();
        for (name, d) in space {
            let u = self.next_f64();
            let v = match d {
                Dist::LogUniform(lo, hi) => json!(round((lo.ln() + u * (hi.ln() - lo.ln())).exp())),
                Dist::Uniform(lo, hi) => json!(round(lo + u * (hi - lo))),
                Dist::Int(lo, hi) => json!(lo + ((u * (hi - lo + 1) as f64) as i64).min(hi - lo)),
                Dist::Choice(options) => {
                    options[((u * options.len() as f64) as usize).min(options.len() - 1)].clone()
                }
            };
            m.insert((*name).to_owned(), v);
        }
        m
    }
}

/// Four significant digits keep reports readable and runs reproducible.
fn round(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mag = 10f64.powi(3 - x.abs().log10().floor() as i32);
    (x * mag).round() / mag
}

/// How many configurations survive a rung.
pub fn survivors(n: usize) -> usize {
    n.div_ceil(REDUCTION).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_is_deterministic_and_in_range() {
        let space = space("lightgbm", "python").unwrap_or_default();
        let a: Vec<_> = (0..20)
            .map({
                let mut s = Sampler::new(7);
                move |_| s.sample(&space)
            })
            .collect();
        let space2 = super::space("lightgbm", "python").unwrap_or_default();
        let mut s = Sampler::new(7);
        let b: Vec<_> = (0..20).map(|_| s.sample(&space2)).collect();
        assert_eq!(a, b);
        for m in &a {
            let lr = m["learning_rate"].as_f64().unwrap_or(-1.0);
            assert!((0.0099..=0.3001).contains(&lr), "{lr}");
            let leaves = m["num_leaves"].as_i64().unwrap_or(-1);
            assert!((15..=127).contains(&leaves), "{leaves}");
        }
    }

    #[test]
    fn halving_schedule() {
        assert_eq!(survivors(30), 10);
        assert_eq!(survivors(10), 4);
        assert_eq!(survivors(1), 1);
        assert!(space("majority", "builtin").is_none());
        assert!(space("logistic_regression", "native").is_some());
    }
}
