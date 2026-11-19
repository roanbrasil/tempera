// Small display helpers shared by the screens.

const LOWER_IS_BETTER = new Set(['rmse', 'mae', 'mape', 'mase', 'log_loss', 'perplexity']);

/** Whether a larger value of `metric` means a better model. */
export function higherIsBetter(metric: string | null | undefined): boolean {
	if (!metric) return true;
	const base = metric.split('/').pop() ?? metric;
	return !(LOWER_IS_BETTER.has(base) || base.endsWith('loss') || base.endsWith('logloss'));
}

const TASK_LABELS: Record<string, string> = {
	binary_classification: 'Binary classification',
	multiclass_classification: 'Multiclass classification',
	regression: 'Regression',
	time_series_forecast: 'Time-series forecast',
	clustering: 'Clustering',
	anomaly_detection: 'Anomaly detection',
	image_classification: 'Image classification',
	text_classification: 'Text classification',
	llm_fine_tune: 'LLM fine-tune'
};

const METRIC_LABELS: Record<string, string> = {
	roc_auc: 'ROC AUC',
	pr_auc: 'PR AUC',
	f1: 'F1',
	balanced_accuracy: 'Balanced accuracy',
	accuracy: 'Accuracy',
	log_loss: 'Log loss',
	rmse: 'RMSE',
	mae: 'MAE',
	mape: 'MAPE',
	mase: 'MASE',
	r2: 'R²',
	silhouette: 'Silhouette',
	perplexity: 'Perplexity'
};

export function taskLabel(task: string | null | undefined): string {
	if (!task) return '—';
	return TASK_LABELS[task] ?? task.replaceAll('_', ' ');
}

export function metricLabel(metric: string | null | undefined): string {
	if (!metric) return '—';
	return METRIC_LABELS[metric] ?? metric;
}

/** `0.87123` → `0.8712`; integers stay integers; tiny/huge values use exponents. */
export function fmtNum(v: number | null | undefined, digits = 4): string {
	if (v === null || v === undefined || Number.isNaN(v)) return '—';
	if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
	if (Number.isInteger(v) && Math.abs(v) < 1e9) return v.toLocaleString('en-US');
	const a = Math.abs(v);
	if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
	return v.toFixed(a >= 100 ? 1 : digits);
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
	if (v === null || v === undefined || Number.isNaN(v)) return '—';
	if (v > 0 && v < 0.001) return '<0.1%';
	return `${(v * 100).toFixed(digits)}%`;
}

export function fmtInt(v: number | null | undefined): string {
	if (v === null || v === undefined) return '—';
	return Math.round(v).toLocaleString('en-US');
}

/** Seconds → `42s`, `3m 05s`, `1h 12m`. */
export function fmtDuration(s: number | null | undefined): string {
	if (s === null || s === undefined || !Number.isFinite(s) || s < 0) return '—';
	if (s < 1) return '<1s';
	if (s < 60) return `${Math.round(s)}s`;
	const m = Math.floor(s / 60);
	if (m < 60) return `${m}m ${String(Math.round(s % 60)).padStart(2, '0')}s`;
	const h = Math.floor(m / 60);
	return `${h}h ${String(m % 60).padStart(2, '0')}m`;
}

export function fmtBytes(b: number | null | undefined): string {
	if (b === null || b === undefined) return '—';
	const units = ['B', 'KB', 'MB', 'GB', 'TB'];
	let i = 0;
	let v = b;
	while (v >= 1024 && i < units.length - 1) {
		v /= 1024;
		i++;
	}
	return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

/** ISO timestamp → `just now`, `5 min ago`, `3 h ago`, or a date. */
export function fmtRelative(iso: string | null | undefined, now = Date.now()): string {
	if (!iso) return '—';
	const t = Date.parse(iso);
	if (Number.isNaN(t)) return iso;
	const s = Math.round((now - t) / 1000);
	if (s < 45) return 'just now';
	if (s < 3600) return `${Math.round(s / 60)} min ago`;
	if (s < 86400) return `${Math.round(s / 3600)} h ago`;
	if (s < 7 * 86400) return `${Math.round(s / 86400)} d ago`;
	return new Date(t).toLocaleDateString('en-US', {
		year: 'numeric',
		month: 'short',
		day: 'numeric'
	});
}

export function fmtDateTime(iso: string | null | undefined): string {
	if (!iso) return '—';
	const t = Date.parse(iso);
	if (Number.isNaN(t)) return iso;
	return new Date(t).toLocaleString('en-US', {
		month: 'short',
		day: 'numeric',
		hour: '2-digit',
		minute: '2-digit'
	});
}

export function shortId(id: string, n = 10): string {
	return id.length > n ? id.slice(0, n) : id;
}
