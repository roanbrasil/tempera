//! Inline SVG chart primitives, generated in Rust (no JavaScript).
//!
//! Charts use CSS classes (`c-*`) defined in the page stylesheet so they follow
//! light/dark mode. Every text argument is raw and is escaped here. Non-finite
//! points are dropped; empty inputs yield an empty string (the caller then
//! shows a note instead of a chart).

use std::fmt::Write as _;

use crate::escape;
use crate::format::truncate;

/// Linear map from a data domain to a pixel range.
#[derive(Debug, Clone, Copy)]
pub struct Scale {
    d0: f64,
    d1: f64,
    r0: f64,
    r1: f64,
}

impl Scale {
    pub fn new(domain: (f64, f64), range: (f64, f64)) -> Self {
        let (mut d0, mut d1) = domain;
        if !d0.is_finite() || !d1.is_finite() {
            (d0, d1) = (0.0, 1.0);
        }
        if (d1 - d0).abs() < f64::EPSILON {
            let pad = if d0.abs() > 0.0 { d0.abs() * 0.1 } else { 1.0 };
            (d0, d1) = (d0 - pad, d1 + pad);
        }
        Self {
            d0,
            d1,
            r0: range.0,
            r1: range.1,
        }
    }

    pub fn map(&self, v: f64) -> f64 {
        let t = (v - self.d0) / (self.d1 - self.d0);
        self.r0 + t.clamp(-0.05, 1.05) * (self.r1 - self.r0)
    }

    pub fn domain(&self) -> (f64, f64) {
        (self.d0, self.d1)
    }
}

/// A coordinate with one decimal; non-finite values become `0`.
pub fn num(v: f64) -> String {
    if v.is_finite() {
        let s = format!("{v:.1}");
        s.strip_suffix(".0").map(str::to_owned).unwrap_or(s)
    } else {
        "0".to_owned()
    }
}

/// "Nice" tick positions (steps of 1, 2 or 5 × 10^k) covering `[lo, hi]`.
pub fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<f64> {
    if !lo.is_finite() || !hi.is_finite() || hi <= lo || target == 0 {
        return Vec::new();
    }
    let raw = (hi - lo) / target as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = mag
        * if norm < 1.5 {
            1.0
        } else if norm < 3.0 {
            2.0
        } else if norm < 7.0 {
            5.0
        } else {
            10.0
        };
    let first = (lo / step).ceil() * step;
    let mut ticks = Vec::new();
    let mut i = 0.0;
    while first + i * step <= hi + step * 1e-9 && ticks.len() < 50 {
        let t = first + i * step;
        ticks.push(if t.abs() < step * 1e-9 { 0.0 } else { t });
        i += 1.0;
    }
    ticks
}

/// Tick label with as many decimals as the step needs.
fn tick_label(v: f64, ticks: &[f64]) -> String {
    let step = if ticks.len() > 1 {
        (ticks[1] - ticks[0]).abs()
    } else {
        1.0
    };
    let decimals = if step >= 1.0 {
        0
    } else {
        (-step.log10().floor()) as usize
    };
    let a = v.abs();
    if a >= 10_000.0 {
        let (div, suffix) = if a >= 1e9 {
            (1e9, "G")
        } else if a >= 1e6 {
            (1e6, "M")
        } else {
            (1e3, "k")
        };
        let s = format!("{:.1}", v / div);
        format!("{}{suffix}", s.strip_suffix(".0").unwrap_or(&s))
    } else {
        format!("{:.*}", decimals.min(4), v)
    }
}

/// Min and max of the finite values, if any.
pub fn extent(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    values.filter(|v| v.is_finite()).fold(None, |acc, v| match acc {
        None => Some((v, v)),
        Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
    })
}

fn open(out: &mut String, w: f64, h: f64, title: &str, desc: &str) {
    let _ = write!(
        out,
        "<svg class=\"chart\" viewBox=\"0 0 {} {}\" role=\"img\" style=\"max-width:{}px\" preserveAspectRatio=\"xMidYMid meet\">\
         <title>{}</title><desc>{}</desc>",
        num(w),
        num(h),
        num(w),
        escape(title),
        escape(desc)
    );
}

/// Plot area in pixels.
#[derive(Debug, Clone, Copy)]
struct Plot {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

/// Grid, axes, tick labels and axis titles.
fn axes(out: &mut String, p: Plot, xs: &Scale, ys: &Scale, x_label: &str, y_label: &str) {
    let xt = nice_ticks(xs.domain().0, xs.domain().1, 5);
    let yt = nice_ticks(ys.domain().0, ys.domain().1, 5);
    out.push_str("<g class=\"c-grid\">");
    for &t in &yt {
        let y = ys.map(t);
        let _ = write!(
            out,
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>",
            num(p.left),
            num(y),
            num(p.right),
            num(y)
        );
    }
    for &t in &xt {
        let x = xs.map(t);
        let _ = write!(
            out,
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>",
            num(x),
            num(p.top),
            num(x),
            num(p.bottom)
        );
    }
    out.push_str("</g>");
    let _ = write!(
        out,
        "<g class=\"c-axis\"><line x1=\"{l}\" y1=\"{b}\" x2=\"{r}\" y2=\"{b}\"/><line x1=\"{l}\" y1=\"{t}\" x2=\"{l}\" y2=\"{b}\"/></g>",
        l = num(p.left),
        r = num(p.right),
        t = num(p.top),
        b = num(p.bottom)
    );
    out.push_str("<g class=\"c-text\">");
    for &t in &xt {
        let _ = write!(
            out,
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\">{}</text>",
            num(xs.map(t)),
            num(p.bottom + 16.0),
            escape(&tick_label(t, &xt))
        );
    }
    for &t in &yt {
        let _ = write!(
            out,
            "<text x=\"{}\" y=\"{}\" text-anchor=\"end\">{}</text>",
            num(p.left - 6.0),
            num(ys.map(t) + 4.0),
            escape(&tick_label(t, &yt))
        );
    }
    let _ = write!(
        out,
        "<text class=\"c-label\" x=\"{}\" y=\"{}\" text-anchor=\"middle\">{}</text>",
        num((p.left + p.right) / 2.0),
        num(p.bottom + 36.0),
        escape(x_label)
    );
    let cy = (p.top + p.bottom) / 2.0;
    let _ = write!(
        out,
        "<text class=\"c-label\" x=\"14\" y=\"{}\" text-anchor=\"middle\" transform=\"rotate(-90 14 {})\">{}</text>",
        num(cy),
        num(cy),
        escape(y_label)
    );
    out.push_str("</g>");
}

/// Reference line drawn behind a curve.
#[derive(Debug, Clone, Copy)]
pub enum Reference {
    None,
    /// `y = x` (chance level for ROC).
    Diagonal,
    /// Horizontal line at this y (e.g. PR baseline = prevalence).
    Horizontal(f64),
}

/// A curve on the unit square (ROC or PR).
#[derive(Debug, Clone, Copy)]
pub struct Curve<'a> {
    pub title: &'a str,
    pub desc: &'a str,
    pub x_label: &'a str,
    pub y_label: &'a str,
    pub points: &'a [(f64, f64)],
    pub reference: Reference,
    pub reference_label: &'a str,
    /// Text shown in the lower-right corner, e.g. `AUC 0.912`.
    pub annotation: Option<&'a str>,
}

/// Unit-square curve chart. Returns an empty string when no finite point exists.
pub fn curve(c: &Curve<'_>) -> String {
    let pts: Vec<(f64, f64)> = c
        .points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    if pts.is_empty() {
        return String::new();
    }
    let (w, h) = (400.0, 340.0);
    let p = Plot {
        left: 52.0,
        top: 14.0,
        right: w - 16.0,
        bottom: h - 52.0,
    };
    let xs = Scale::new((0.0, 1.0), (p.left, p.right));
    let ys = Scale::new((0.0, 1.0), (p.bottom, p.top));
    let mut out = String::new();
    open(&mut out, w, h, c.title, c.desc);
    axes(&mut out, p, &xs, &ys, c.x_label, c.y_label);
    match c.reference {
        Reference::None => {}
        Reference::Diagonal => {
            let _ = write!(
                out,
                "<line class=\"c-ref\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"><title>{}</title></line>",
                num(xs.map(0.0)),
                num(ys.map(0.0)),
                num(xs.map(1.0)),
                num(ys.map(1.0)),
                escape(c.reference_label)
            );
        }
        Reference::Horizontal(y) if y.is_finite() => {
            let _ = write!(
                out,
                "<line class=\"c-ref\" x1=\"{}\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\"><title>{}</title></line>",
                num(xs.map(0.0)),
                num(xs.map(1.0)),
                escape(c.reference_label),
                y = num(ys.map(y.clamp(0.0, 1.0)))
            );
        }
        Reference::Horizontal(_) => {}
    }
    let path: Vec<String> = pts
        .iter()
        .map(|&(x, y)| format!("{},{}", num(xs.map(x)), num(ys.map(y))))
        .collect();
    let _ = write!(out, "<polyline class=\"c-line\" points=\"{}\"/>", path.join(" "));
    if let Some(a) = c.annotation {
        let _ = write!(
            out,
            "<text class=\"c-note\" x=\"{}\" y=\"{}\" text-anchor=\"end\">{}</text>",
            num(p.right - 8.0),
            num(p.bottom - 10.0),
            escape(a)
        );
    }
    out.push_str("</svg>");
    out
}

/// Predicted-vs-actual scatter with the `y = x` line. Empty when no finite point.
pub fn scatter(title: &str, desc: &str, points: &[(f64, f64)], x_label: &str, y_label: &str) -> String {
    let pts: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    let Some((lo, hi)) = extent(pts.iter().flat_map(|&(x, y)| [x, y])) else {
        return String::new();
    };
    let pad = ((hi - lo) * 0.04).max(f64::EPSILON);
    let (lo, hi) = (lo - pad, hi + pad);
    let (w, h) = (440.0, 400.0);
    let p = Plot {
        left: 60.0,
        top: 14.0,
        right: w - 16.0,
        bottom: h - 52.0,
    };
    let xs = Scale::new((lo, hi), (p.left, p.right));
    let ys = Scale::new((lo, hi), (p.bottom, p.top));
    let (lo, hi) = xs.domain();
    let mut out = String::new();
    open(&mut out, w, h, title, desc);
    axes(&mut out, p, &xs, &ys, x_label, y_label);
    let _ = write!(
        out,
        "<line class=\"c-ref\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"><title>Perfect prediction (y = x)</title></line>",
        num(xs.map(lo)),
        num(ys.map(lo)),
        num(xs.map(hi)),
        num(ys.map(hi))
    );
    out.push_str("<g class=\"c-dot\">");
    for &(x, y) in &pts {
        let _ = write!(
            out,
            "<circle cx=\"{}\" cy=\"{}\" r=\"3\"/>",
            num(xs.map(x)),
            num(ys.map(y))
        );
    }
    out.push_str("</g></svg>");
    out
}

/// Horizontal bar chart (e.g. feature importance). Bars use the magnitude;
/// the value label keeps the sign. Empty when no finite value.
pub fn hbar(title: &str, desc: &str, items: &[(String, f64)]) -> String {
    let items: Vec<&(String, f64)> = items.iter().filter(|(_, v)| v.is_finite()).collect();
    let max = items.iter().map(|(_, v)| v.abs()).fold(0.0_f64, f64::max);
    if items.is_empty() {
        return String::new();
    }
    let row = 24.0;
    let (w, label_w, value_w) = (520.0, 170.0, 64.0);
    let h = items.len() as f64 * row + 12.0;
    let xs = Scale::new((0.0, if max > 0.0 { max } else { 1.0 }), (label_w, w - value_w));
    let mut out = String::new();
    open(&mut out, w, h, title, desc);
    for (i, (name, v)) in items.iter().enumerate() {
        let y = 6.0 + i as f64 * row;
        let bw = (xs.map(v.abs()) - label_w).max(1.0);
        let _ = write!(
            out,
            "<g><title>{name}: {val}</title>\
             <text class=\"c-text c-bar-label\" x=\"{lx}\" y=\"{ty}\" text-anchor=\"end\">{short}</text>\
             <rect class=\"c-bar\" x=\"{x}\" y=\"{by}\" width=\"{bw}\" height=\"{bh}\" rx=\"2\"/>\
             <text class=\"c-text\" x=\"{vx}\" y=\"{ty}\">{val}</text></g>",
            name = escape(name),
            val = escape(&crate::format::metric(*v)),
            short = escape(&truncate(name, 24)),
            lx = num(label_w - 8.0),
            ty = num(y + row / 2.0 + 4.0),
            x = num(label_w),
            by = num(y + 4.0),
            bw = num(bw),
            bh = num(row - 8.0),
            vx = num(label_w + bw + 6.0),
        );
    }
    out.push_str("</svg>");
    out
}

/// Confusion matrix as a colored grid: rows = true class, columns = predicted.
/// Cell shade is the row share (per-class recall on the diagonal).
pub fn confusion(title: &str, desc: &str, classes: &[String], matrix: &[Vec<u64>]) -> String {
    let n = classes.len();
    if n == 0 || matrix.len() != n || matrix.iter().any(|r| r.len() != n) {
        return String::new();
    }
    let cell = if n <= 3 {
        88.0
    } else if n <= 6 {
        64.0
    } else {
        (480.0 / n as f64).max(22.0)
    };
    let label_w = 110.0;
    let top = 56.0;
    let w = label_w + cell * n as f64 + 12.0;
    let h = top + cell * n as f64 + 40.0;
    let small = cell < 50.0;
    let mut out = String::new();
    open(&mut out, w, h, title, desc);
    let grid_mid = label_w + cell * n as f64 / 2.0;
    let _ = write!(
        out,
        "<text class=\"c-label\" x=\"{}\" y=\"16\" text-anchor=\"middle\">Predicted class</text>",
        num(grid_mid)
    );
    for (j, c) in classes.iter().enumerate() {
        let _ = write!(
            out,
            "<text class=\"c-text\" x=\"{}\" y=\"{}\" text-anchor=\"middle\"><title>{}</title>{}</text>",
            num(label_w + cell * (j as f64 + 0.5)),
            num(top - 10.0),
            escape(c),
            escape(&truncate(c, if small { 4 } else { 12 }))
        );
    }
    for (i, row) in matrix.iter().enumerate() {
        let total: u64 = row.iter().sum();
        let y = top + cell * i as f64;
        let _ = write!(
            out,
            "<text class=\"c-text\" x=\"{}\" y=\"{}\" text-anchor=\"end\"><title>{}</title>{}</text>",
            num(label_w - 8.0),
            num(y + cell / 2.0 + 4.0),
            escape(&classes[i]),
            escape(&truncate(&classes[i], 14))
        );
        for (j, &v) in row.iter().enumerate() {
            let share = if total > 0 { v as f64 / total as f64 } else { 0.0 };
            let x = label_w + cell * j as f64;
            let kind = if i == j { "c-hit" } else { "c-miss" };
            let strong = if share > 0.55 { " c-strong" } else { "" };
            let _ = write!(
                out,
                "<g><title>True {t} → predicted {p}: {v} ({s})</title>\
                 <rect class=\"c-cell-bg\" x=\"{x}\" y=\"{y}\" width=\"{c}\" height=\"{c}\"/>\
                 <rect class=\"{kind}\" x=\"{x}\" y=\"{y}\" width=\"{c}\" height=\"{c}\" fill-opacity=\"{o:.2}\"/>\
                 <text class=\"c-count{strong}\" x=\"{cx}\" y=\"{cy}\" text-anchor=\"middle\">{v}</text>",
                t = escape(&classes[i]),
                p = escape(&classes[j]),
                v = crate::format::count(v),
                s = crate::format::percent(share),
                x = num(x),
                y = num(y),
                c = num(cell),
                o = 0.06 + 0.94 * share,
                cx = num(x + cell / 2.0),
                cy = num(y + cell / 2.0 + if small { 4.0 } else { 0.0 }),
            );
            if !small {
                let _ = write!(
                    out,
                    "<text class=\"c-share{strong}\" x=\"{}\" y=\"{}\" text-anchor=\"middle\">{}</text>",
                    num(x + cell / 2.0),
                    num(y + cell / 2.0 + 18.0),
                    crate::format::percent(share)
                );
            }
            out.push_str("</g>");
        }
    }
    let _ = write!(
        out,
        "<text class=\"c-label\" x=\"14\" y=\"{cy}\" text-anchor=\"middle\" transform=\"rotate(-90 14 {cy})\">True class</text>",
        cy = num(top + cell * n as f64 / 2.0)
    );
    let _ = write!(
        out,
        "<text class=\"c-text\" x=\"{}\" y=\"{}\" text-anchor=\"middle\">Shade: share of the row</text>",
        num(grid_mid),
        num(h - 14.0)
    );
    out.push_str("</svg>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_nice() {
        let unit = nice_ticks(0.0, 1.0, 5);
        assert_eq!(unit.len(), 6);
        assert!((unit[5] - 1.0).abs() < 1e-9);
        assert!(nice_ticks(f64::NAN, 1.0, 5).is_empty());
        assert!(nice_ticks(1.0, 1.0, 5).is_empty());
        let t = nice_ticks(-3.0, 117.0, 5);
        assert_eq!(t.first().copied(), Some(0.0));
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert_eq!(
            curve(&Curve {
                title: "t",
                desc: "d",
                x_label: "x",
                y_label: "y",
                points: &[(f64::NAN, 1.0)],
                reference: Reference::Diagonal,
                reference_label: "r",
                annotation: None,
            }),
            ""
        );
        assert_eq!(scatter("t", "d", &[], "x", "y"), "");
        let one = scatter("t", "d", &[(5.0, 5.0)], "x", "y");
        assert!(one.starts_with("<svg") && !one.contains("NaN"));
        assert_eq!(hbar("t", "d", &[("a".into(), f64::NAN)]), "");
        let zero = hbar("t", "d", &[("a".into(), 0.0)]);
        assert!(zero.contains("<rect"));
        assert_eq!(confusion("t", "d", &["a".into()], &[vec![1, 2]]), "");
        assert_eq!(num(f64::INFINITY), "0");
    }
}
