//! Small numerical solvers owned by this crate: least squares through the
//! normal equations (Cholesky) and L2-regularized logistic regression with
//! L-BFGS. They replace `linfa-linear`/`linfa-logistic`, whose optimizer
//! (argmin) depends on the unmaintained `paste` crate (RUSTSEC-2024-0436).

use ndarray::{Array1, Array2, ArrayView1, Axis};

/// Solves `a·x = b` for symmetric positive (semi-)definite `a`. Adds a tiny
/// ridge when `a` is singular (collinear features), which returns the
/// minimum-norm-like solution instead of failing.
pub(crate) fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    let scale = (0..n).map(|i| a[[i, i]].abs()).fold(0.0, f64::max).max(1.0);
    for jitter in [0.0, 1e-12, 1e-10, 1e-8, 1e-6] {
        if let Some(l) = cholesky(a, jitter * scale) {
            // L·z = b, then Lᵀ·x = z.
            let mut z = Array1::zeros(n);
            for i in 0..n {
                let s: f64 = (0..i).map(|k| l[[i, k]] * z[k]).sum();
                z[i] = (b[i] - s) / l[[i, i]];
            }
            let mut x = Array1::zeros(n);
            for i in (0..n).rev() {
                let s: f64 = (i + 1..n).map(|k| l[[k, i]] * x[k]).sum();
                x[i] = (z[i] - s) / l[[i, i]];
            }
            if x.iter().all(|v: &f64| v.is_finite()) {
                return Some(x);
            }
        }
    }
    None
}

fn cholesky(a: &Array2<f64>, jitter: f64) -> Option<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[[i, k]] * l[[j, k]]).sum();
            if i == j {
                let d = a[[i, i]] + jitter - s;
                if d <= 1e-300 || !d.is_finite() {
                    return None;
                }
                l[[i, i]] = d.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - s) / l[[j, j]];
            }
        }
    }
    Some(l)
}

/// Ordinary (`alpha = 0`) or ridge least squares; the intercept is never
/// penalized. Returns `(coef, intercept)`.
pub(crate) fn least_squares(
    x: &Array2<f64>,
    y: &[f64],
    alpha: f64,
    fit_intercept: bool,
) -> Option<(Vec<f64>, f64)> {
    let y = ArrayView1::from(y);
    let (xc, yc, x_mean, y_mean) = if fit_intercept {
        let xm = x.mean_axis(Axis(0))?;
        let ym = y.mean()?;
        (x - &xm, y.mapv(|v| v - ym), xm, ym)
    } else {
        (x.clone(), y.to_owned(), Array1::zeros(x.ncols()), 0.0)
    };
    let mut gram = xc.t().dot(&xc);
    for i in 0..gram.nrows() {
        gram[[i, i]] += alpha;
    }
    let w = solve_spd(&gram, &xc.t().dot(&yc))?;
    let intercept = y_mean - x_mean.dot(&w);
    Some((w.to_vec(), intercept))
}

/// Minimizes `f` with L-BFGS (memory 10) and a backtracking Armijo line search.
/// Stops when the gradient's max-norm drops below `tol`.
fn lbfgs<F>(f: F, mut x: Vec<f64>, max_iter: u64, tol: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    const M: usize = 10;
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
    let (mut fx, mut g) = f(&x);
    let mut hist: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> = Default::default();
    for _ in 0..max_iter {
        if g.iter().fold(0.0f64, |m, v| m.max(v.abs())) < tol {
            break;
        }
        // Two-loop recursion: d = -H·g.
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(hist.len());
        for (s, y, rho) in hist.iter().rev() {
            let a = rho * dot(s, &q);
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
            alphas.push(a);
        }
        if let Some((s, y, _)) = hist.back() {
            let gamma = dot(s, y) / dot(y, y);
            q.iter_mut().for_each(|v| *v *= gamma);
        }
        for ((s, y, rho), a) in hist.iter().zip(alphas.into_iter().rev()) {
            let b = rho * dot(y, &q);
            q.iter_mut().zip(s).for_each(|(qi, si)| *qi += (a - b) * si);
        }
        let mut d: Vec<f64> = q.into_iter().map(|v| -v).collect();
        let mut slope = dot(&g, &d);
        if slope >= 0.0 {
            // Not a descent direction: restart from steepest descent.
            hist.clear();
            d = g.iter().map(|v| -v).collect();
            slope = dot(&g, &d);
        }
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let xn: Vec<f64> = x.iter().zip(&d).map(|(xi, di)| xi + step * di).collect();
            let (fn_, gn) = f(&xn);
            if fn_.is_finite() && fn_ <= fx + 1e-4 * step * slope {
                accepted = Some((xn, fn_, gn));
                break;
            }
            step *= 0.5;
        }
        let Some((xn, fn_, gn)) = accepted else { break };
        let s: Vec<f64> = xn.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = gn.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if hist.len() == M {
                hist.pop_front();
            }
            hist.push_back((s, y, 1.0 / sy));
        }
        let improved = (fx - fn_).abs() > 1e-14 * fx.abs().max(1.0);
        x = xn;
        fx = fn_;
        g = gn;
        if !improved {
            break;
        }
    }
    x
}

fn log1p_exp(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// Binary logistic regression; `y[i]` is true for the positive class.
/// Objective: `Σ log-loss + alpha/2·||w||²` (intercept unpenalized).
pub(crate) fn logistic_binary(
    x: &Array2<f64>,
    y: &[bool],
    alpha: f64,
    max_iter: u64,
    tol: f64,
) -> (Vec<f64>, f64) {
    let p = x.ncols();
    let f = |theta: &[f64]| {
        let w = ArrayView1::from(&theta[..p]);
        let b = theta[p];
        let z = x.dot(&w) + b;
        let mut loss = 0.5 * alpha * w.dot(&w);
        let mut r = Array1::zeros(x.nrows());
        for (i, (&zi, &yi)) in z.iter().zip(y).enumerate() {
            loss += if yi { log1p_exp(-zi) } else { log1p_exp(zi) };
            r[i] = 1.0 / (1.0 + (-zi).exp()) - f64::from(u8::from(yi));
        }
        let mut grad = x.t().dot(&r) + &w.mapv(|v| alpha * v);
        let gb = r.sum();
        let mut out = std::mem::take(&mut grad).to_vec();
        out.push(gb);
        (loss, out)
    };
    let theta = lbfgs(f, vec![0.0; p + 1], max_iter, tol);
    (theta[..p].to_vec(), theta[p])
}

/// Multinomial (softmax) logistic regression over `k` classes `0..k`.
/// Returns one coefficient row and one intercept per class.
pub(crate) fn logistic_multinomial(
    x: &Array2<f64>,
    y: &[usize],
    k: usize,
    alpha: f64,
    max_iter: u64,
    tol: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let p = x.ncols();
    let f = |theta: &[f64]| {
        // theta = [W (k×p, row-major), b (k)]
        let Ok(w) = ndarray::ArrayView2::from_shape((k, p), &theta[..k * p]) else {
            return (f64::INFINITY, vec![0.0; theta.len()]);
        };
        let b = ArrayView1::from(&theta[k * p..]);
        let mut z = x.dot(&w.t()) + b; // n×k
        let mut loss = 0.5 * alpha * w.iter().map(|v| v * v).sum::<f64>();
        for (i, mut row) in z.rows_mut().into_iter().enumerate() {
            let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - m).exp());
            let s = row.sum();
            loss -= (row[y[i]] / s).ln();
            row.mapv_inplace(|v| v / s);
            row[y[i]] -= 1.0; // residual: softmax - onehot
        }
        let gw = z.t().dot(x) + &w.mapv(|v| alpha * v); // k×p
        let gb = z.sum_axis(Axis(0));
        let mut out = gw.iter().copied().collect::<Vec<_>>();
        out.extend(gb.iter());
        (loss, out)
    };
    let theta = lbfgs(f, vec![0.0; k * p + k], max_iter, tol);
    let coef = (0..k).map(|c| theta[c * p..(c + 1) * p].to_vec()).collect();
    (coef, theta[k * p..].to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;
    use ndarray::array;

    #[test]
    fn ols_recovers_exact_line() {
        let x = array![[1.0, 0.0], [2.0, 1.0], [3.0, 5.0], [4.0, 2.0], [5.0, 3.0]];
        let y: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|r| 2.0 * r[0] - 3.0 * r[1] + 7.0)
            .collect();
        let (w, b) = least_squares(&x, &y, 0.0, true).expect("solves");
        assert!((w[0] - 2.0).abs() < 1e-9 && (w[1] + 3.0).abs() < 1e-9 && (b - 7.0).abs() < 1e-9);
    }

    #[test]
    fn collinear_columns_do_not_fail() {
        let x = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
        let y = [1.0, 2.0, 3.0, 4.0];
        let (w, b) = least_squares(&x, &y, 0.0, true).expect("solves");
        let pred: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|r| r[0] * w[0] + r[1] * w[1] + b)
            .collect();
        assert!(pred.iter().zip(y).all(|(p, t)| (p - t).abs() < 1e-4));
    }

    #[test]
    fn logistic_separates_and_gradient_vanishes() {
        let x = array![[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [0.2], [-0.3]];
        let y = [false, false, false, true, true, true, false, true];
        let (w, b) = logistic_binary(&x, &y, 1.0, 200, 1e-8);
        assert!(w[0] > 0.5, "{w:?} {b}");
        // Optimality: gradient of the bias is Σ(p - y) = 0.
        let s: f64 = x
            .column(0)
            .iter()
            .zip(&y)
            .map(|(xi, &yi)| 1.0 / (1.0 + (-(w[0] * xi + b)).exp()) - f64::from(u8::from(yi)))
            .sum();
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn multinomial_learns_three_clusters() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.2],
            [5.0, 0.0],
            [5.2, 0.1],
            [0.0, 5.0],
            [0.2, 5.1]
        ];
        let y = [0, 0, 1, 1, 2, 2];
        let (w, b) = logistic_multinomial(&x, &y, 3, 0.1, 500, 1e-8);
        for (row, &t) in x.rows().into_iter().zip(&y) {
            let z: Vec<f64> = (0..3).map(|c| row.dot(&ArrayView1::from(&w[c])) + b[c]).collect();
            let best = (0..3).max_by(|&a, &c| z[a].total_cmp(&z[c])).unwrap_or(0);
            assert_eq!(best, t);
        }
    }
}
