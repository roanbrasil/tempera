//! Evaluation metrics, computed in one place for every backend.
//!
//! Backends only write predictions; scoring here guarantees that a LightGBM
//! model and a linfa model are measured with exactly the same code.

use crate::spec::Metric;

/// Whether larger values of the metric are better.
pub fn higher_is_better(m: Metric) -> bool {
    !matches!(
        m,
        Metric::LogLoss | Metric::Rmse | Metric::Mae | Metric::Mape | Metric::Mase | Metric::Perplexity
    )
}

/// ROC-AUC via the rank statistic (ties get average ranks).
/// `y` is 1 for the positive class. Returns NaN if only one class is present.
pub fn roc_auc(y: &[bool], score: &[f64]) -> f64 {
    let n_pos = y.iter().filter(|&&p| p).count() as f64;
    let n_neg = y.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    let ranks = average_ranks(score);
    let rank_sum: f64 = y.iter().zip(&ranks).filter(|(p, _)| **p).map(|(_, r)| r).sum();
    (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Average precision (area under the precision-recall curve, step-wise),
/// matching scikit-learn's `average_precision_score`.
pub fn pr_auc(y: &[bool], score: &[f64]) -> f64 {
    let n_pos = y.iter().filter(|&&p| p).count();
    if n_pos == 0 {
        return f64::NAN;
    }
    let mut idx: Vec<usize> = (0..y.len()).collect();
    idx.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    let (mut tp, mut fp, mut ap, mut prev_recall) = (0usize, 0usize, 0.0, 0.0);
    let mut i = 0;
    while i < idx.len() {
        // Process tied scores as one threshold.
        let s = score[idx[i]];
        while i < idx.len() && score[idx[i]] == s {
            if y[idx[i]] {
                tp += 1
            } else {
                fp += 1
            }
            i += 1;
        }
        let recall = tp as f64 / n_pos as f64;
        let precision = tp as f64 / (tp + fp) as f64;
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    ap
}

/// Multiclass log loss. `proba[i][k]` is the probability of class `k` for row `i`.
pub fn log_loss(y: &[usize], proba: &[Vec<f64>]) -> f64 {
    const EPS: f64 = 1e-15;
    let n = y.len().max(1) as f64;
    -y.iter()
        .zip(proba)
        .map(|(&k, p)| {
            let total: f64 = p.iter().sum();
            let pk = p.get(k).copied().unwrap_or(0.0) / if total > 0.0 { total } else { 1.0 };
            pk.clamp(EPS, 1.0 - EPS).ln()
        })
        .sum::<f64>()
        / n
}

pub fn accuracy(y: &[usize], pred: &[usize]) -> f64 {
    if y.is_empty() {
        return f64::NAN;
    }
    y.iter().zip(pred).filter(|(a, b)| a == b).count() as f64 / y.len() as f64
}

/// Mean per-class recall over classes present in `y`.
pub fn balanced_accuracy(y: &[usize], pred: &[usize], n_classes: usize) -> f64 {
    let cm = confusion_matrix(y, pred, n_classes);
    let recalls: Vec<f64> = (0..n_classes)
        .filter_map(|k| {
            let support: u64 = cm[k].iter().sum();
            (support > 0).then(|| cm[k][k] as f64 / support as f64)
        })
        .collect();
    if recalls.is_empty() {
        f64::NAN
    } else {
        recalls.iter().sum::<f64>() / recalls.len() as f64
    }
}

/// Binary F1 of class 1 when `n_classes == 2`, macro F1 otherwise.
pub fn f1(y: &[usize], pred: &[usize], n_classes: usize) -> f64 {
    let cm = confusion_matrix(y, pred, n_classes);
    let f1_of = |k: usize| {
        let tp = cm[k][k] as f64;
        let fp: f64 = (0..n_classes).filter(|&j| j != k).map(|j| cm[j][k] as f64).sum();
        let fn_: f64 = (0..n_classes).filter(|&j| j != k).map(|j| cm[k][j] as f64).sum();
        if tp == 0.0 {
            0.0
        } else {
            2.0 * tp / (2.0 * tp + fp + fn_)
        }
    };
    if n_classes == 2 {
        f1_of(1)
    } else {
        (0..n_classes).map(f1_of).sum::<f64>() / n_classes.max(1) as f64
    }
}

/// `cm[true][pred]` counts.
pub fn confusion_matrix(y: &[usize], pred: &[usize], n_classes: usize) -> Vec<Vec<u64>> {
    let mut cm = vec![vec![0u64; n_classes]; n_classes];
    for (&t, &p) in y.iter().zip(pred) {
        if t < n_classes && p < n_classes {
            cm[t][p] += 1;
        }
    }
    cm
}

pub fn rmse(y: &[f64], pred: &[f64]) -> f64 {
    mean(y.iter().zip(pred).map(|(a, b)| (a - b).powi(2))).sqrt()
}

pub fn mae(y: &[f64], pred: &[f64]) -> f64 {
    mean(y.iter().zip(pred).map(|(a, b)| (a - b).abs()))
}

/// Mean absolute percentage error, skipping rows where the target is 0.
pub fn mape(y: &[f64], pred: &[f64]) -> f64 {
    mean(
        y.iter()
            .zip(pred)
            .filter(|(a, _)| **a != 0.0)
            .map(|(a, b)| ((a - b) / a).abs()),
    )
}

pub fn r2(y: &[f64], pred: &[f64]) -> f64 {
    let m = mean(y.iter().copied());
    let ss_tot: f64 = y.iter().map(|a| (a - m).powi(2)).sum();
    let ss_res: f64 = y.iter().zip(pred).map(|(a, b)| (a - b).powi(2)).sum();
    if ss_tot == 0.0 {
        f64::NAN
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// ROC curve points `(fpr, tpr)`, at most `max_points` (for reports).
pub fn roc_curve(y: &[bool], score: &[f64], max_points: usize) -> Vec<(f64, f64)> {
    curve(y, score, max_points, |tp, fp, n_pos, n_neg| {
        (fp / n_neg, tp / n_pos)
    })
}

/// Precision-recall curve points `(recall, precision)`, at most `max_points`.
pub fn pr_curve(y: &[bool], score: &[f64], max_points: usize) -> Vec<(f64, f64)> {
    curve(y, score, max_points, |tp, fp, n_pos, _| {
        (tp / n_pos, if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 })
    })
}

fn curve(
    y: &[bool],
    score: &[f64],
    max_points: usize,
    point: impl Fn(f64, f64, f64, f64) -> (f64, f64),
) -> Vec<(f64, f64)> {
    let n_pos = y.iter().filter(|&&p| p).count() as f64;
    let n_neg = y.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..y.len()).collect();
    idx.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    let mut pts = vec![point(0.0, 0.0, n_pos, n_neg)];
    let (mut tp, mut fp) = (0.0, 0.0);
    let mut i = 0;
    while i < idx.len() {
        let s = score[idx[i]];
        while i < idx.len() && score[idx[i]] == s {
            if y[idx[i]] {
                tp += 1.0
            } else {
                fp += 1.0
            }
            i += 1;
        }
        pts.push(point(tp, fp, n_pos, n_neg));
    }
    downsample(pts, max_points)
}

fn downsample(pts: Vec<(f64, f64)>, max: usize) -> Vec<(f64, f64)> {
    if pts.len() <= max || max < 2 {
        return pts;
    }
    let step = (pts.len() - 1) as f64 / (max - 1) as f64;
    (0..max)
        .map(|i| pts[((i as f64 * step).round() as usize).min(pts.len() - 1)])
        .collect()
}

fn mean(it: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = it.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn average_ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut ranks = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // Reference values computed with scikit-learn 1.5.
    #[test]
    fn binary_ranking_metrics_match_sklearn() {
        let y = [false, false, true, true];
        let s = [0.1, 0.4, 0.35, 0.8];
        assert!(close(roc_auc(&y, &s), 0.75));
        assert!(close(pr_auc(&y, &s), 0.8333333333333333));
        // ties
        let s = [0.5, 0.5, 0.5, 0.9];
        assert!(close(roc_auc(&y, &s), 0.75));
        assert!(roc_auc(&[true, true], &[0.1, 0.2]).is_nan());
    }

    #[test]
    fn classification_metrics() {
        let y = [0, 0, 1, 1, 2, 2];
        let p = [0, 1, 1, 1, 2, 0];
        assert!(close(accuracy(&y, &p), 4.0 / 6.0));
        assert!(close(balanced_accuracy(&y, &p, 3), (0.5 + 1.0 + 0.5) / 3.0));
        // sklearn f1_score(average="macro") = 0.6555555555555556
        assert!(close(f1(&y, &p, 3), 0.6555555555555556));
        assert!(close(f1(&[0, 1, 1, 0], &[0, 1, 0, 0], 2), 2.0 / 3.0));
        let ll = log_loss(&[0, 1], &[vec![0.9, 0.1], vec![0.2, 0.8]]);
        assert!(close(ll, -(0.9f64.ln() + 0.8f64.ln()) / 2.0));
    }

    #[test]
    fn regression_metrics() {
        let y = [3.0, -0.5, 2.0, 7.0];
        let p = [2.5, 0.0, 2.0, 8.0];
        assert!(close(rmse(&y, &p), 0.6123724356957945));
        assert!(close(mae(&y, &p), 0.5));
        assert!(close(r2(&y, &p), 0.9486081370449679));
        assert!(close(mape(&[1.0, 0.0, 2.0], &[1.5, 3.0, 2.0]), 0.25));
    }

    #[test]
    fn curves_are_bounded() {
        let y: Vec<bool> = (0..1000).map(|i| i % 3 == 0).collect();
        let s: Vec<f64> = (0..1000).map(|i| ((i * 37) % 101) as f64).collect();
        let roc = roc_curve(&y, &s, 50);
        assert!(roc.len() <= 50);
        assert_eq!(*roc.first().unwrap_or(&(9.0, 9.0)), (0.0, 0.0));
        assert!(close(roc.last().map_or(0.0, |p| p.0), 1.0));
    }
}
