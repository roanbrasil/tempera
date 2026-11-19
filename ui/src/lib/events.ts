// Live job events (docs/wiki/Contracts.md §9) and the reducer that folds them into a view state.
//
// The reducer is plain TypeScript (no Svelte) so it can be unit tested and run outside the UI.
// It mutates the state in place for speed (thousands of metric points); the Svelte layer
// (job-stream.svelte.ts) takes a shallow snapshot once per animation frame.

import type { AuditReport, Evaluation, Finding, RunRow, TaskInference } from './types';

export const STAGES = [
	'ingest',
	'profile',
	'audit',
	'split',
	'train',
	'evaluate',
	'report'
] as const;
export type StageName = (typeof STAGES)[number];

/** `raw` = unparsed backend stderr (e.g. Python tracebacks). */
export type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'raw';
export type JobFinalStatus = 'succeeded' | 'failed' | 'blocked' | 'cancelled';

export interface StageEvent {
	type: 'stage';
	stage: StageName;
	message: string;
}
export interface AuditEvent {
	type: 'audit';
	inference: TaskInference;
	audit: AuditReport;
}
export interface BlockedEvent {
	type: 'blocked';
	findings: Finding[];
}
export interface RunStartedEvent {
	type: 'run_started';
	run_id: string;
	algorithm: string;
	backend: string;
	index: number;
	total: number;
}
export interface MetricEvent {
	type: 'metric';
	run_id: string;
	name: string;
	step: number | null;
	value: number;
}
export interface ProgressEvent {
	type: 'progress';
	run_id: string;
	step: number;
	total_steps: number | null;
	eta_s: number | null;
	job_eta_s: number | null;
}
export interface LogEvent {
	type: 'log';
	run_id: string | null;
	level: LogLevel;
	message: string;
}
export interface RunFinishedEvent {
	type: 'run_finished';
	run_id: string;
	status: string;
	valid_score: number | null;
	valid_metrics: Record<string, number> | null;
	duration_s: number | null;
	error: string | null;
}
export interface WinnerEvent {
	type: 'winner';
	run_id: string;
	algorithm: string;
	test: Evaluation;
}
export interface JobFinishedEvent {
	type: 'job_finished';
	status: JobFinalStatus | (string & {});
	report_url: string | null;
	error: string | null;
}

/** Cloud spend (M6): on launch, while running, and once at the end. */
export interface CostEvent {
	type: 'cost';
	machine: string;
	usd_per_hour: number;
	estimated_usd: number;
	spent_usd: number;
	budget_usd: number | null;
	status:
		'launched' | 'running' | 'budget_alert' | 'budget_exceeded' | 'terminated' | (string & {});
	savings: string | null;
}

export type JobEvent =
	| CostEvent
	| StageEvent
	| AuditEvent
	| BlockedEvent
	| RunStartedEvent
	| MetricEvent
	| ProgressEvent
	| LogEvent
	| RunFinishedEvent
	| WinnerEvent
	| JobFinishedEvent;

// ------------------------------------------------------------------------ state

export interface LivePoint {
	step: number | null;
	value: number;
}

export interface RunState {
	run_id: string;
	algorithm: string;
	backend: string;
	/** 1-based position announced by the server (`index` of `run_started`). */
	index: number;
	status: string;
	is_baseline: boolean;
	step: number | null;
	total_steps: number | null;
	eta_s: number | null;
	/** Latest value of every metric streamed for this run. */
	live: Record<string, LivePoint>;
	valid_score: number | null;
	valid_metrics: Record<string, number>;
	duration_s: number | null;
	error: string | null;
}

/** One metric curve: parallel arrays, sorted by step. */
export interface Series {
	steps: number[];
	values: number[];
}

export interface LogLine {
	seq: number;
	run_id: string | null;
	level: LogLevel;
	message: string;
}

export interface JobState {
	/** Number of events applied. */
	events: number;
	stage: StageName | null;
	stages: Partial<Record<StageName, string>>;
	inference: TaskInference | null;
	audit: AuditReport | null;
	blocked: Finding[] | null;
	runs: RunState[];
	totalRuns: number | null;
	jobEtaS: number | null;
	/** Latest cloud spend, when the job runs on a rented machine. */
	cost: Omit<CostEvent, 'type'> | null;
	logs: LogLine[];
	logsDropped: number;
	winner: { run_id: string; algorithm: string; test: Evaluation } | null;
	finished: { status: string; report_url: string | null; error: string | null } | null;
	/** run_id → metric name → curve. */
	series: Record<string, Record<string, Series>>;
	/** Bumped whenever a curve changes (charts redraw on it). */
	metricRev: number;
}

export const LOG_CAP = 5000;

export function initialState(): JobState {
	return {
		events: 0,
		stage: null,
		stages: {},
		inference: null,
		audit: null,
		blocked: null,
		runs: [],
		totalRuns: null,
		jobEtaS: null,
		cost: null,
		logs: [],
		logsDropped: 0,
		winner: null,
		finished: null,
		series: {},
		metricRev: 0
	};
}

function findRun(s: JobState, id: string): RunState | undefined {
	// Runs are few (tens); a linear scan from the end hits the active run first.
	for (let i = s.runs.length - 1; i >= 0; i--) if (s.runs[i].run_id === id) return s.runs[i];
	return undefined;
}

function ensureRun(s: JobState, id: string): RunState {
	let r = findRun(s, id);
	if (!r) {
		r = newRun(id, '?', '?', s.runs.length + 1);
		s.runs.push(r);
	}
	return r;
}

function newRun(id: string, algorithm: string, backend: string, index: number): RunState {
	return {
		run_id: id,
		algorithm,
		backend,
		index,
		status: 'running',
		is_baseline: index === 1,
		step: null,
		total_steps: null,
		eta_s: null,
		live: {},
		valid_score: null,
		valid_metrics: {},
		duration_s: null,
		error: null
	};
}

/** Inserts or replaces the point at `step`, keeping the curve sorted. */
export function pushPoint(series: Series, step: number, value: number): void {
	const n = series.steps.length;
	if (n === 0 || step > series.steps[n - 1]) {
		series.steps.push(step);
		series.values.push(value);
		return;
	}
	// Binary search for the first step >= `step`.
	let lo = 0;
	let hi = n;
	while (lo < hi) {
		const mid = (lo + hi) >> 1;
		if (series.steps[mid] < step) lo = mid + 1;
		else hi = mid;
	}
	if (series.steps[lo] === step) series.values[lo] = value;
	else {
		series.steps.splice(lo, 0, step);
		series.values.splice(lo, 0, value);
	}
}

const TERMINAL_RUN = new Set(['succeeded', 'failed', 'cancelled', 'skipped']);

/** Applies one event to `s` (in place) and returns it. */
export function reduce(s: JobState, ev: JobEvent): JobState {
	s.events++;
	switch (ev.type) {
		case 'stage':
			s.stage = ev.stage;
			s.stages[ev.stage] = ev.message;
			break;
		case 'audit':
			s.inference = ev.inference;
			s.audit = ev.audit;
			break;
		case 'blocked':
			s.blocked = ev.findings;
			break;
		case 'run_started': {
			const existing = findRun(s, ev.run_id);
			if (existing) {
				existing.algorithm = ev.algorithm;
				existing.backend = ev.backend;
				existing.index = ev.index;
			} else {
				s.runs.push(newRun(ev.run_id, ev.algorithm, ev.backend, ev.index));
				s.runs.sort((a, b) => a.index - b.index);
			}
			s.totalRuns = ev.total;
			break;
		}
		case 'metric': {
			if (typeof ev.value !== 'number') break;
			const r = ensureRun(s, ev.run_id);
			const prev = r.live[ev.name];
			if (!prev || prev.step === null || ev.step === null || ev.step >= prev.step)
				r.live[ev.name] = { step: ev.step ?? null, value: ev.value };
			if (isCurveMetric(ev.name, ev.step)) {
				const byName = (s.series[ev.run_id] ??= {});
				const curve = (byName[ev.name] ??= { steps: [], values: [] });
				pushPoint(curve, ev.step as number, ev.value);
			}
			s.metricRev++;
			break;
		}
		case 'progress': {
			const r = ensureRun(s, ev.run_id);
			r.step = ev.step;
			r.total_steps = ev.total_steps ?? r.total_steps;
			r.eta_s = ev.eta_s;
			s.jobEtaS = ev.job_eta_s;
			break;
		}
		case 'log': {
			s.logs.push({
				seq: s.events,
				run_id: ev.run_id ?? null,
				level: ev.level,
				message: ev.message
			});
			if (s.logs.length > LOG_CAP + 500) {
				const drop = s.logs.length - LOG_CAP;
				s.logs.splice(0, drop);
				s.logsDropped += drop;
			}
			break;
		}
		case 'run_finished': {
			const r = ensureRun(s, ev.run_id);
			r.status = ev.status;
			r.valid_score = ev.valid_score;
			r.valid_metrics = ev.valid_metrics ?? {};
			r.duration_s = ev.duration_s;
			r.error = ev.error;
			r.eta_s = 0;
			if (ev.status === 'succeeded' && r.total_steps !== null) r.step = r.total_steps;
			break;
		}
		case 'cost':
			// The final event reports no hourly price; keep the last known one.
			s.cost = {
				machine: ev.machine,
				usd_per_hour: ev.usd_per_hour || s.cost?.usd_per_hour || 0,
				estimated_usd: ev.estimated_usd,
				spent_usd: ev.spent_usd,
				budget_usd: ev.budget_usd,
				status: ev.status,
				savings: ev.savings
			};
			break;
		case 'winner':
			s.winner = { run_id: ev.run_id, algorithm: ev.algorithm, test: ev.test };
			break;
		case 'job_finished':
			s.finished = { status: ev.status, report_url: ev.report_url, error: ev.error };
			s.jobEtaS = 0;
			for (const r of s.runs) {
				if (!TERMINAL_RUN.has(r.status))
					r.status = ev.status === 'cancelled' ? 'cancelled' : 'failed';
				r.eta_s = 0;
			}
			break;
		default:
			// Unknown event type (newer server): count it but otherwise ignore.
			break;
	}
	return s;
}

export function buildState(events: Iterable<JobEvent>): JobState {
	const s = initialState();
	for (const ev of events) reduce(s, ev);
	return s;
}

/** Parses one WebSocket text frame; returns null for anything that is not an event object. */
export function parseEvent(text: string): JobEvent | null {
	try {
		const v = JSON.parse(text) as unknown;
		if (v && typeof v === 'object' && typeof (v as { type?: unknown }).type === 'string')
			return v as JobEvent;
	} catch {
		// ignore malformed frames
	}
	return null;
}

// --------------------------------------------------------------- metric kinds

/** Metric families that are single values (scores, importances, diagnostics), never curves. */
export const SCALAR_FAMILIES = ['importance/', 'eval/', 'pca/', 'select/', 'dbscan/', 'eta/'];

/** Whether a metric point belongs on a training curve: it has a step and is not a scalar family. */
export function isCurveMetric(name: string, step: number | null | undefined): boolean {
	if (step === null || step === undefined) return false;
	return !SCALAR_FAMILIES.some((f) => name.startsWith(f));
}

export interface Importance {
	feature: string;
	value: number;
}

/** `importance/<feature>` values, largest magnitude first. */
export function importanceOf(values: Record<string, { value: number }>): Importance[] {
	return Object.entries(values)
		.filter(([n]) => n.startsWith('importance/'))
		.map(([n, v]) => ({ feature: n.slice('importance/'.length), value: v.value }))
		.filter((x) => Number.isFinite(x.value))
		.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

/** Splits stored points (`GET /runs/{id}/metrics`) into curves and latest scalar values. */
export function splitPoints(points: { name: string; step: number | null; value: number }[]): {
	curves: Record<string, Series>;
	scalars: Record<string, LivePoint>;
} {
	const curves: Record<string, Series> = {};
	const scalars: Record<string, LivePoint> = {};
	for (const p of points) {
		if (isCurveMetric(p.name, p.step)) {
			pushPoint((curves[p.name] ??= { steps: [], values: [] }), p.step as number, p.value);
		} else scalars[p.name] = { step: p.step, value: p.value };
	}
	return { curves, scalars };
}

// --------------------------------------------------------------------- replay

/**
 * Keeps a job state across WebSocket reconnects.
 *
 * On every (re)connect the server replays the job's whole history before streaming new events.
 * Folding the replay into the existing state would duplicate logs and curves, so a reconnect
 * starts a fresh `pending` state; the visible (`committed`) state stays on screen until the
 * replay has caught up with it, then the fresh state takes over. No flicker, no duplicates.
 */
export class ReplayingState {
	committed: JobState = initialState();
	private pending: JobState | null = null;

	/** Call when a connection opens (the first one included). */
	beginReplay(): void {
		this.pending = initialState();
	}

	get replaying(): boolean {
		return this.pending !== null;
	}

	apply(ev: JobEvent): void {
		const p = this.pending;
		if (!p) {
			reduce(this.committed, ev);
			return;
		}
		if (ev.type === 'job_finished' && p.events + 1 < this.committed.events) {
			// The server lost the history (e.g. it restarted and only reports the final status):
			// keep what we already know and just record the outcome.
			reduce(this.committed, ev);
			this.pending = null;
			return;
		}
		reduce(p, ev);
		if (p.events >= this.committed.events) {
			this.committed = p;
			this.pending = null;
		}
	}

	get state(): JobState {
		return this.committed;
	}
}

// -------------------------------------------------------------------- helpers

/** Runs known from the event stream, completed with the rows stored on the server (`GET /jobs/{id}`). */
export function mergeRuns(live: RunState[], rows: RunRow[]): RunState[] {
	const byId = new Map(rows.map((r) => [r.id, r]));
	const out = live.map((r) => {
		const row = byId.get(r.run_id);
		if (!row) return r;
		byId.delete(r.run_id);
		return {
			...r,
			is_baseline: row.is_baseline,
			algorithm: r.algorithm === '?' ? row.algorithm : r.algorithm,
			backend: r.backend === '?' ? row.backend : r.backend
		};
	});
	let next = out.length;
	for (const row of rows) {
		if (!byId.has(row.id)) continue;
		next++;
		out.push({
			...newRun(row.id, row.algorithm, row.backend, next),
			is_baseline: row.is_baseline,
			status: row.status,
			valid_score: row.valid_score,
			valid_metrics: row.valid_metrics ?? {},
			duration_s: row.duration_s,
			error: row.error
		});
	}
	return out;
}

/** The validation metric to show live for a run: the job metric if streamed, else any `valid/*`. */
export function liveValidMetric(
	run: RunState,
	metric: string | null | undefined
): { name: string; value: number } | null {
	const names = Object.keys(run.live);
	const valid = names.filter((n) => n.startsWith('valid/') || n.startsWith('eval/'));
	const pick =
		(metric ? valid.find((n) => n.endsWith('/' + metric)) : undefined) ??
		valid[valid.length - 1] ??
		null;
	return pick ? { name: pick, value: run.live[pick].value } : null;
}

export function runProgress(run: RunState): number | null {
	if (TERMINAL_RUN.has(run.status)) return 1;
	if (run.step === null || !run.total_steps) return null;
	return Math.min(1, run.step / run.total_steps);
}

export function isTerminalRun(status: string): boolean {
	return TERMINAL_RUN.has(status);
}
