// Dev mock: simulated training jobs that emit the live events of Contracts §9 with realistic
// timing (≈40 steps/s, several metrics per step) so the job screen can be developed offline.

import type { JobEvent } from '../src/lib/events';
import type {
	AuditReport,
	Evaluation,
	JobRow,
	Metric,
	RunRow,
	TaskInference
} from '../src/lib/types';

interface AlgoProfile {
	key: string;
	backend: string;
	steps: number;
	/** 0..1, how good the final model is. */
	quality: number;
	/** Validation loss starts rising after this fraction of the steps (overfitting), if set. */
	overfitAt?: number;
	warn?: string;
	fail?: string;
}

const CLASSIFICATION: AlgoProfile[] = [
	{ key: 'majority', backend: 'builtin', steps: 1, quality: 0 },
	{ key: 'logistic_regression', backend: 'native', steps: 120, quality: 0.62 },
	{ key: 'random_forest', backend: 'python', steps: 200, quality: 0.78 },
	{
		key: 'lightgbm',
		backend: 'python',
		steps: 400,
		quality: 0.88,
		warn: '[LightGBM] [Warning] No further splits with positive gain, best gain: -inf'
	},
	{ key: 'xgboost', backend: 'python', steps: 400, quality: 0.84, overfitAt: 0.55 },
	{ key: 'catboost', backend: 'python', steps: 560, quality: 0.93 }
];

const REGRESSION: AlgoProfile[] = [
	{ key: 'mean', backend: 'builtin', steps: 1, quality: 0 },
	{ key: 'ridge', backend: 'native', steps: 80, quality: 0.6 },
	{ key: 'random_forest', backend: 'python', steps: 200, quality: 0.75 },
	{
		key: 'lightgbm',
		backend: 'python',
		steps: 300,
		quality: 0.85,
		fail: 'worker exited with code 137 (killed: out of memory)'
	}
];

const CLUSTERING: AlgoProfile[] = [
	{ key: 'kmeans', backend: 'python', steps: 60, quality: 0.7 },
	{ key: 'pca_kmeans', backend: 'python', steps: 60, quality: 0.78 },
	{ key: 'dbscan', backend: 'python', steps: 1, quality: 0.5 }
];

export interface JobPlan {
	name: string;
	inference: TaskInference;
	audit: AuditReport;
	rows: number;
	candidates: 'auto' | string[];
	baseline: boolean;
	/** Force the job to fail on the last run (demo of a failed job). */
	failLast?: boolean;
}

type Step = JobEvent | { sleep: number };

function lowerIsBetter(m: Metric): boolean {
	return ['rmse', 'mae', 'mape', 'mase', 'log_loss', 'perplexity'].includes(m);
}

function rng(seed: number) {
	let s = seed >>> 0 || 1;
	return () => {
		s ^= s << 13;
		s ^= s >>> 17;
		s ^= s << 5;
		return ((s >>> 0) % 1_000_000) / 1_000_000;
	};
}

export class MockJob {
	readonly row: JobRow;
	readonly runs: RunRow[] = [];
	readonly history: JobEvent[] = [];
	private readonly subs = new Set<(ev: JobEvent) => void>();
	cancelled = false;
	private wake: (() => void) | null = null;

	constructor(
		readonly id: string,
		readonly plan: JobPlan,
		createdAt: Date
	) {
		this.row = {
			id,
			name: plan.name,
			status: 'created',
			task: plan.inference.task,
			metric: plan.inference.metric,
			dataset_id: plan.audit.dataset_id,
			winner_run_id: null,
			report_path: null,
			error: null,
			created_at: createdAt.toISOString().replace(/\.\d+Z$/, 'Z'),
			finished_at: null
		};
	}

	get finished(): boolean {
		return this.history.some((e) => e.type === 'job_finished');
	}

	subscribe(fn: (ev: JobEvent) => void): () => void {
		this.subs.add(fn);
		return () => this.subs.delete(fn);
	}

	cancel(): void {
		this.cancelled = true;
		this.wake?.();
	}

	private emit(ev: JobEvent): void {
		this.history.push(ev);
		this.track(ev);
		for (const fn of this.subs) fn(ev);
	}

	/** Mirrors the events into the stored rows (what GET /jobs/{id} returns). */
	private track(ev: JobEvent): void {
		switch (ev.type) {
			case 'stage':
				this.row.status = 'running';
				break;
			case 'run_started':
				this.runs.push({
					id: ev.run_id,
					job_id: this.id,
					algorithm: ev.algorithm,
					backend: ev.backend,
					is_baseline: ev.backend === 'builtin',
					status: 'running',
					valid_score: null,
					valid_metrics: {},
					model_dir: null,
					error: null,
					duration_s: null
				});
				break;
			case 'run_finished': {
				const r = this.runs.find((x) => x.id === ev.run_id);
				if (r) {
					r.status = ev.status;
					r.valid_score = ev.valid_score;
					r.valid_metrics = ev.valid_metrics ?? {};
					r.duration_s = ev.duration_s;
					r.error = ev.error;
					r.model_dir = ev.status === 'succeeded' ? `artifacts/${this.id}/${r.id}` : null;
				}
				break;
			}
			case 'winner':
				this.row.winner_run_id = ev.run_id;
				break;
			case 'job_finished':
				this.row.status = ev.status;
				this.row.error = ev.error;
				this.row.finished_at = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
				if (ev.report_url) this.row.report_path = `artifacts/${this.id}/report.html`;
				break;
		}
	}

	/** Runs the script; `live` = honour the sleeps, otherwise replay instantly (seeded jobs). */
	async run(live: boolean, speed = 1): Promise<void> {
		for (const step of this.script()) {
			if ('sleep' in step) {
				if (live && !this.cancelled)
					await new Promise<void>((res) => {
						const t = setTimeout(res, step.sleep / speed);
						this.wake = () => {
							clearTimeout(t);
							res();
						};
					});
				continue;
			}
			this.emit(step);
		}
	}

	private *script(): Generator<Step> {
		const p = this.plan;
		const r = rng(this.id.length * 7919 + p.rows);
		const metric = p.inference.metric;
		const lower = lowerIsBetter(metric);
		const task = p.inference.task;
		const classification = task.includes('classification');
		const lossName = classification ? 'binary_logloss' : task === 'regression' ? 'l2' : 'inertia';

		yield {
			type: 'stage',
			stage: 'ingest',
			message: `Reading ${p.rows.toLocaleString('en-US')} rows into Parquet`
		};
		yield {
			type: 'log',
			run_id: null,
			level: 'info',
			message: `dataset ${p.audit.dataset_id.slice(0, 12)} already ingested (content hash match)`
		};
		yield { sleep: 500 };
		yield { type: 'stage', stage: 'profile', message: 'Profiling columns' };
		yield { sleep: 450 };
		yield { type: 'stage', stage: 'audit', message: 'Running the data audit' };
		yield { sleep: 600 };
		yield { type: 'audit', inference: p.inference, audit: p.audit };
		for (const f of p.audit.findings.filter((f) => f.severity !== 'green'))
			yield {
				type: 'log',
				run_id: null,
				level: f.severity === 'red' ? 'error' : 'warn',
				message: `audit ${f.id}: ${f.title}${f.resolution.status === 'auto' ? ` (auto: ${f.resolution.action})` : ''}`
			};
		if (p.audit.blocked) {
			const red = p.audit.findings.filter(
				(f) => f.severity === 'red' && f.resolution.status === 'unresolved'
			);
			yield { type: 'blocked', findings: red };
			yield {
				type: 'job_finished',
				status: 'blocked',
				report_url: null,
				error: `blocked by the audit: ${red.map((f) => f.id).join(', ')}`
			};
			return;
		}

		const train = Math.round(p.rows * 0.64);
		const valid = Math.round(p.rows * 0.16);
		yield {
			type: 'stage',
			stage: 'split',
			message: `Stratified split · ${train} train / ${valid} valid / ${p.rows - train - valid} test`
		};
		yield { sleep: 400 };

		let algos = classification ? CLASSIFICATION : task === 'regression' ? REGRESSION : CLUSTERING;
		if (Array.isArray(p.candidates) && !(p.candidates.length === 1 && p.candidates[0] === 'all')) {
			const picked = p.candidates;
			const base = algos.filter((a) => a.backend === 'builtin' && p.baseline);
			const chosen = picked.map(
				(k) =>
					algos.find((a) => a.key === k) ?? {
						key: k,
						backend: 'python',
						steps: 150,
						quality: 0.6 + r() * 0.3
					}
			);
			algos = [...base, ...chosen.filter((a) => a.backend !== 'builtin')];
		} else if (!p.baseline) algos = algos.filter((a) => a.backend !== 'builtin');
		if (!p.failLast) algos = algos.map((a) => ({ ...a, fail: undefined }));

		yield {
			type: 'stage',
			stage: 'train',
			message: `Training ${algos.length} candidates (baseline first)`
		};

		const STEP_MS = 25;
		const durations: number[] = [];
		const results: {
			id: string;
			algo: AlgoProfile;
			score: number;
			metrics: Record<string, number>;
		}[] = [];
		let failed: string | null = null;

		for (let i = 0; i < algos.length; i++) {
			if (this.cancelled) break;
			const a = algos[i];
			const runId = `run-${this.id.slice(-4)}-${String(i + 1).padStart(2, '0')}`;
			yield {
				type: 'run_started',
				run_id: runId,
				algorithm: a.key,
				backend: a.backend,
				index: i + 1,
				total: algos.length
			};
			yield {
				type: 'log',
				run_id: runId,
				level: 'info',
				message: `starting ${a.key} on ${a.backend} backend`
			};
			if (a.backend === 'python')
				yield {
					type: 'log',
					run_id: runId,
					level: 'debug',
					message: 'python worker ready (uv env tabular, 0.4s)'
				};

			const finalScore = lower ? 1.0 - a.quality * 0.62 : 0.12 + a.quality * 0.6;
			const lossFloor = 0.55 - a.quality * 0.3;
			let t0 = 0;
			let ended = a.steps;
			const median = () => {
				if (durations.length === 0) return (a.steps * STEP_MS * 1.2) / 1000;
				const s = [...durations].sort((x, y) => x - y);
				return s[Math.floor(s.length / 2)];
			};
			for (let s = 1; s <= a.steps; s++) {
				if (this.cancelled) {
					ended = s;
					break;
				}
				if (a.fail && s > a.steps * 0.6) {
					ended = s;
					failed = a.fail;
					break;
				}
				const x = s / a.steps;
				const noise = () => (r() - 0.5) * 0.012;
				const trainLoss = lossFloor * 0.85 + 0.45 * Math.exp(-x * 4.5) + noise();
				let validLoss = lossFloor + 0.42 * Math.exp(-x * 4) + noise() * 1.5;
				if (a.overfitAt && x > a.overfitAt) validLoss += (x - a.overfitAt) ** 2 * 0.35;
				const progress = 1 - Math.exp(-x * 4);
				const score = lower
					? finalScore + (1 - progress) * 0.4 + noise()
					: 0.1 + (finalScore - 0.1) * progress + noise();
				if (a.steps > 1) {
					yield {
						type: 'metric',
						run_id: runId,
						name: `train/${lossName}`,
						step: s,
						value: trainLoss
					};
					yield {
						type: 'metric',
						run_id: runId,
						name: `valid/${lossName}`,
						step: s,
						value: validLoss
					};
					if (s % 2 === 0 || s === a.steps)
						yield { type: 'metric', run_id: runId, name: `valid/${metric}`, step: s, value: score };
				}
				if (s % 5 === 0 || s === a.steps) {
					const eta = ((a.steps - s) * STEP_MS) / 1000;
					yield {
						type: 'progress',
						run_id: runId,
						step: s,
						total_steps: a.steps,
						eta_s: eta,
						job_eta_s: eta + (algos.length - i - 1) * median()
					};
				}
				if (a.warn && s === Math.floor(a.steps * 0.7))
					yield { type: 'log', run_id: runId, level: 'warn', message: a.warn };
				if (a.overfitAt && s === Math.floor(a.steps * (a.overfitAt + 0.15)))
					yield {
						type: 'log',
						run_id: runId,
						level: 'warn',
						message: `overfitting: valid/${lossName} rising for 40 steps while train/${lossName} falls (best step ${Math.floor(a.steps * a.overfitAt)})`
					};
				if (a.backend === 'python' && s === Math.floor(a.steps / 3)) {
					yield {
						type: 'log',
						run_id: runId,
						level: 'warn',
						message: `/home/user/.cache/tempera/envs/tabular-3.12/lib/python3.12/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but ${a.key} was fitted with feature names (artifacts/${this.id}/${runId}/model/estimator.joblib)`
					};
					for (const line of [
						'Traceback (most recent call last):',
						'  File "/home/user/.cache/tempera/envs/tabular-3.12/lib/python3.12/site-packages/joblib/parallel.py", line 1863, in _get_sequential_output',
						'    res = func(*args, **kwargs)',
						'ResourceWarning: unclosed file <_io.BufferedReader name=3> (retried, harmless)'
					])
						yield { type: 'log', run_id: runId, level: 'raw', message: line };
				}
				if (s % 100 === 0)
					yield {
						type: 'log',
						run_id: runId,
						level: 'info',
						message: `[${s}] train/${lossName}=${trainLoss.toFixed(4)} valid/${lossName}=${validLoss.toFixed(4)}`
					};
				yield { sleep: a.steps === 1 ? 700 : STEP_MS };
				t0 += a.steps === 1 ? 0.7 : STEP_MS / 1000;
			}
			const duration = Math.round((t0 + 0.3 + r() * 0.4) * 100) / 100;
			if (this.cancelled) {
				yield { type: 'log', run_id: runId, level: 'warn', message: `cancelled at step ${ended}` };
				yield {
					type: 'run_finished',
					run_id: runId,
					status: 'cancelled',
					valid_score: null,
					valid_metrics: {},
					duration_s: duration,
					error: null
				};
				break;
			}
			if (failed) {
				yield { type: 'log', run_id: runId, level: 'error', message: failed };
				yield {
					type: 'run_finished',
					run_id: runId,
					status: 'failed',
					valid_score: null,
					valid_metrics: {},
					duration_s: duration,
					error: failed
				};
				break;
			}
			const metrics: Record<string, number> = classification
				? {
						[metric]: finalScore,
						roc_auc: Math.min(0.99, 0.5 + a.quality * 0.42),
						f1: 0.05 + a.quality * 0.55,
						log_loss: lossFloor + 0.02
					}
				: task === 'regression'
					? { rmse: finalScore, mae: finalScore * 0.78, r2: a.quality * 0.86 }
					: { silhouette: finalScore };
			if (a.backend !== 'builtin') {
				// Step-less scalars, as the real backends emit them (not curves).
				const feats = p.audit.plan.features.map((f) => f.name);
				const w = feats.map(() => r() ** 2);
				const sum = w.reduce((x, y) => x + y, 0) || 1;
				for (let k = 0; k < feats.length; k++)
					yield {
						type: 'metric',
						run_id: runId,
						name: `importance/${feats[k]}`,
						step: null,
						value: w[k] / sum
					};
				yield {
					type: 'metric',
					run_id: runId,
					name: 'eta/step_s',
					step: null,
					value: STEP_MS / 1000
				};
			}
			durations.push(duration);
			results.push({ id: runId, algo: a, score: finalScore, metrics });
			yield {
				type: 'log',
				run_id: runId,
				level: 'info',
				message: `${a.key} done: valid ${metric}=${finalScore.toFixed(4)} in ${duration.toFixed(1)}s`
			};
			yield {
				type: 'run_finished',
				run_id: runId,
				status: 'succeeded',
				valid_score: finalScore,
				valid_metrics: metrics,
				duration_s: duration,
				error: null
			};
			yield { sleep: 250 };
		}

		if (this.cancelled) {
			yield {
				type: 'job_finished',
				status: 'cancelled',
				report_url: null,
				error: 'cancelled by the user'
			};
			return;
		}
		if (failed) {
			yield { type: 'job_finished', status: 'failed', report_url: null, error: failed };
			return;
		}

		yield {
			type: 'stage',
			stage: 'evaluate',
			message: 'Scoring the best model on the test set (once)'
		};
		yield { sleep: 600 };
		const best = results.reduce((b, x) =>
			lower ? (x.score < b.score ? x : b) : x.score > b.score ? x : b
		);
		const testRows = p.rows - train - valid;
		const test: Evaluation = classification
			? {
					rows: testRows,
					metrics: Object.fromEntries(
						Object.entries(best.metrics).map(([k, v]) => [k, v - 0.01 + r() * 0.01])
					),
					classes: ['no', 'yes'],
					confusion: [
						[Math.round(testRows * 0.8), Math.round(testRows * 0.06)],
						[Math.round(testRows * 0.05), Math.round(testRows * 0.09)]
					]
				}
			: { rows: testRows, metrics: best.metrics };
		yield { type: 'winner', run_id: best.id, algorithm: best.algo.key, test };
		yield { type: 'stage', stage: 'report', message: 'Writing the HTML report' };
		yield { sleep: 400 };
		yield {
			type: 'job_finished',
			status: 'succeeded',
			report_url: `/api/v1/jobs/${this.id}/report`,
			error: null
		};
	}

	reportHtml(): string {
		const rows = this.runs
			.map(
				(r) =>
					`<tr><td>${r.algorithm}${r.is_baseline ? ' <em>(baseline)</em>' : ''}</td><td>${r.backend}</td><td>${r.status}</td><td>${r.valid_score?.toFixed(4) ?? '—'}</td><td>${r.duration_s?.toFixed(1) ?? '—'}s</td></tr>`
			)
			.join('');
		const winner = this.history.find((e) => e.type === 'winner');
		const test =
			winner && winner.type === 'winner'
				? Object.entries(winner.test.metrics)
						.map(([k, v]) => `<li><b>${k}</b>: ${v.toFixed(4)}</li>`)
						.join('')
				: '';
		return `<!doctype html><html><head><meta charset="utf-8"><title>${this.row.name} · report</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#2B2F36;background:#fff}h1{margin:0 0 .25rem}table{border-collapse:collapse;margin-top:1rem}td,th{padding:.4rem .8rem;border-bottom:1px solid #ddd;text-align:left}.band{height:4px;background:linear-gradient(90deg,#E9D9A6 0 25%,#B7793F 25% 50%,#6B4C9A 50% 75%,#2F5D8A 75%);margin:1rem 0}</style></head>
<body><h1>${this.row.name}</h1><div>${this.row.task} · ${this.row.metric} · mock report</div><div class="band"></div>
<h2>Winner: ${winner && winner.type === 'winner' ? winner.algorithm : '—'}</h2><ul>${test}</ul>
<h2>Leaderboard</h2><table><tr><th>Algorithm</th><th>Backend</th><th>Status</th><th>Valid score</th><th>Time</th></tr>${rows}</table>
<p style="color:#888;margin-top:2rem">Generated by the UI dev mock. The real report comes from tempera-report.</p></body></html>`;
	}
}
