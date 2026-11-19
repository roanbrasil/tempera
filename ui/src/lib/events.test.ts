import { describe, expect, it } from 'vitest';
import {
	buildState,
	importanceOf,
	initialState,
	LOG_CAP,
	liveValidMetric,
	mergeRuns,
	parseEvent,
	pushPoint,
	reduce,
	ReplayingState,
	runProgress,
	splitPoints,
	type JobEvent
} from './events';
import { alignSeries } from './chart-data';
import type { RunRow } from './types';

const R1 = 'run-1';
const R2 = 'run-2';

/** A short but complete job: 2 runs, metrics, logs, winner. */
function jobEvents(): JobEvent[] {
	return [
		{ type: 'stage', stage: 'ingest', message: 'reading' },
		{ type: 'stage', stage: 'audit', message: 'auditing' },
		{
			type: 'audit',
			inference: { task: 'binary_classification', metric: 'pr_auc', reasons: ['2 classes'] },
			audit: {
				dataset_id: 'ds',
				policy: 'block_on_red',
				findings: [],
				plan: { features: [], datetime: [], dropped: {}, dedupe: false },
				blocked: false
			}
		},
		{ type: 'stage', stage: 'train', message: 'training' },
		{
			type: 'run_started',
			run_id: R1,
			algorithm: 'majority',
			backend: 'builtin',
			index: 1,
			total: 2
		},
		{
			type: 'run_finished',
			run_id: R1,
			status: 'succeeded',
			valid_score: 0.12,
			valid_metrics: { pr_auc: 0.12 },
			duration_s: 0.1,
			error: null
		},
		{
			type: 'run_started',
			run_id: R2,
			algorithm: 'lightgbm',
			backend: 'python',
			index: 2,
			total: 2
		},
		{ type: 'metric', run_id: R2, name: 'train/binary_logloss', step: 1, value: 0.6 },
		{ type: 'metric', run_id: R2, name: 'valid/binary_logloss', step: 1, value: 0.65 },
		{ type: 'metric', run_id: R2, name: 'valid/pr_auc', step: 1, value: 0.3 },
		{ type: 'progress', run_id: R2, step: 1, total_steps: 4, eta_s: 3, job_eta_s: 3 },
		{ type: 'metric', run_id: R2, name: 'train/binary_logloss', step: 2, value: 0.5 },
		{ type: 'metric', run_id: R2, name: 'valid/binary_logloss', step: 2, value: 0.55 },
		{ type: 'metric', run_id: R2, name: 'valid/pr_auc', step: 2, value: 0.5 },
		{ type: 'progress', run_id: R2, step: 2, total_steps: 4, eta_s: 2, job_eta_s: 2 },
		{ type: 'log', run_id: R2, level: 'warn', message: 'no further splits' },
		{ type: 'metric', run_id: R2, name: 'valid/pr_auc', step: 4, value: 0.62 },
		{
			type: 'run_finished',
			run_id: R2,
			status: 'succeeded',
			valid_score: 0.62,
			valid_metrics: { pr_auc: 0.62 },
			duration_s: 4.2,
			error: null
		},
		{ type: 'stage', stage: 'evaluate', message: 'testing' },
		{
			type: 'winner',
			run_id: R2,
			algorithm: 'lightgbm',
			test: { rows: 100, metrics: { pr_auc: 0.6 } }
		},
		{ type: 'job_finished', status: 'succeeded', report_url: '/api/v1/jobs/j/report', error: null }
	];
}

describe('reduce', () => {
	it('builds the full job state from an event list', () => {
		const s = buildState(jobEvents());
		expect(s.events).toBe(jobEvents().length);
		expect(s.stage).toBe('evaluate');
		expect(s.stages.ingest).toBe('reading');
		expect(s.inference?.metric).toBe('pr_auc');
		expect(s.runs.map((r) => r.algorithm)).toEqual(['majority', 'lightgbm']);
		expect(s.totalRuns).toBe(2);
		const lgbm = s.runs[1];
		expect(lgbm.status).toBe('succeeded');
		expect(lgbm.valid_score).toBe(0.62);
		expect(lgbm.step).toBe(4); // completed runs show the full progress
		expect(runProgress(lgbm)).toBe(1);
		expect(s.series[R2]['valid/pr_auc']).toEqual({ steps: [1, 2, 4], values: [0.3, 0.5, 0.62] });
		expect(s.series[R2]['train/binary_logloss'].values).toEqual([0.6, 0.5]);
		expect(s.logs).toHaveLength(1);
		expect(s.logs[0]).toMatchObject({ level: 'warn', run_id: R2 });
		expect(s.winner?.algorithm).toBe('lightgbm');
		expect(s.finished?.status).toBe('succeeded');
		expect(s.jobEtaS).toBe(0);
	});

	it('tracks live progress, ETA and the latest validation metric of a running run', () => {
		const events = jobEvents().slice(0, 15);
		const s = buildState(events);
		const run = s.runs[1];
		expect(run.status).toBe('running');
		expect(run.step).toBe(2);
		expect(run.total_steps).toBe(4);
		expect(run.eta_s).toBe(2);
		expect(runProgress(run)).toBe(0.5);
		expect(s.jobEtaS).toBe(2);
		expect(liveValidMetric(run, 'pr_auc')).toEqual({ name: 'valid/pr_auc', value: 0.5 });
		expect(s.finished).toBeNull();
	});

	it('keeps runs ordered by index even if they start out of order', () => {
		const s = buildState([
			{ type: 'run_started', run_id: 'b', algorithm: 'b', backend: 'python', index: 2, total: 2 },
			{ type: 'run_started', run_id: 'a', algorithm: 'a', backend: 'python', index: 1, total: 2 }
		]);
		expect(s.runs.map((r) => r.run_id)).toEqual(['a', 'b']);
	});

	it('marks unfinished runs as cancelled when the job is cancelled', () => {
		const s = buildState([
			{ type: 'run_started', run_id: 'a', algorithm: 'a', backend: 'python', index: 1, total: 3 },
			{ type: 'progress', run_id: 'a', step: 3, total_steps: 10, eta_s: 5, job_eta_s: 20 },
			{
				type: 'job_finished',
				status: 'cancelled',
				report_url: null,
				error: 'cancelled by the user'
			}
		]);
		expect(s.runs[0].status).toBe('cancelled');
		expect(s.jobEtaS).toBe(0);
	});

	it('keeps step-less and scalar-family metrics out of the curves', () => {
		const s = initialState();
		reduce(s, { type: 'metric', run_id: 'r', name: 'eval/silhouette', step: null, value: 0.4 });
		reduce(s, { type: 'metric', run_id: 'r', name: 'importance/tenure', step: null, value: 0.3 });
		reduce(s, { type: 'metric', run_id: 'r', name: 'importance/plan', step: null, value: -0.5 });
		reduce(s, { type: 'metric', run_id: 'r', name: 'pca/explained', step: 1, value: 0.9 });
		expect(s.series.r).toBeUndefined();
		expect(s.runs[0].live['eval/silhouette'].value).toBe(0.4);
		expect(importanceOf(s.runs[0].live)).toEqual([
			{ feature: 'plan', value: -0.5 },
			{ feature: 'tenure', value: 0.3 }
		]);
		expect(liveValidMetric(s.runs[0], 'silhouette')).toEqual({
			name: 'eval/silhouette',
			value: 0.4
		});
	});

	it('handles out-of-order and repeated steps', () => {
		const s = initialState();
		reduce(s, { type: 'metric', run_id: 'r', name: 'm', step: 5, value: 5 });
		reduce(s, { type: 'metric', run_id: 'r', name: 'm', step: 2, value: 2 });
		reduce(s, { type: 'metric', run_id: 'r', name: 'm', step: 5, value: 55 });
		expect(s.series.r.m).toEqual({ steps: [2, 5], values: [2, 55] });
		// The run was never announced: it is created on the fly.
		expect(s.runs).toHaveLength(1);
	});

	it('splitPoints separates stored curves from scalars', () => {
		const { curves, scalars } = splitPoints([
			{ name: 'train/loss', step: 1, value: 1 },
			{ name: 'train/loss', step: 2, value: 0.5 },
			{ name: 'importance/a', step: null, value: 0.2 },
			{ name: 'eta/step_s', step: 3, value: 0.01 }
		]);
		expect(Object.keys(curves)).toEqual(['train/loss']);
		expect(Object.keys(scalars).sort()).toEqual(['eta/step_s', 'importance/a']);
	});

	it('caps the log buffer', () => {
		const s = initialState();
		for (let i = 0; i < LOG_CAP + 600; i++)
			reduce(s, { type: 'log', run_id: null, level: 'info', message: `line ${i}` });
		expect(s.logs.length).toBeLessThanOrEqual(LOG_CAP + 500);
		expect(s.logsDropped + s.logs.length).toBe(LOG_CAP + 600);
		expect(s.logs[s.logs.length - 1].message).toBe(`line ${LOG_CAP + 599}`);
	});

	it('ignores unknown event types', () => {
		const s = buildState([{ type: 'gpu_usage', util: 0.5 } as unknown as JobEvent]);
		expect(s.events).toBe(1);
		expect(s.runs).toHaveLength(0);
	});
});

describe('ReplayingState (reconnect)', () => {
	it('rebuilds the same state when the server replays the history on reconnect', () => {
		const all = jobEvents();
		const rs = new ReplayingState();

		// First connection: receives the first 12 events, then drops.
		rs.beginReplay();
		for (const ev of all.slice(0, 12)) rs.apply(ev);
		const before = rs.state;
		expect(before.series[R2]['valid/pr_auc'].steps).toEqual([1]);

		// Reconnect: the server replays everything from the start, then streams the rest.
		rs.beginReplay();
		for (const ev of all.slice(0, 6)) {
			rs.apply(ev);
			// The old state stays visible until the replay catches up (no flicker to empty).
			expect(rs.state).toBe(before);
		}
		for (const ev of all.slice(6)) rs.apply(ev);

		expect(rs.replaying).toBe(false);
		expect(rs.state).toEqual(buildState(all));
		// No duplicated logs or points.
		expect(rs.state.logs).toHaveLength(1);
		expect(rs.state.series[R2]['valid/pr_auc'].steps).toEqual([1, 2, 4]);
	});

	it('a page reload (fresh state + full replay) yields the complete state', () => {
		const rs = new ReplayingState();
		rs.beginReplay();
		for (const ev of jobEvents()) rs.apply(ev);
		expect(rs.state).toEqual(buildState(jobEvents()));
	});

	it('keeps the known state when the server lost the history (restart) and only sends the outcome', () => {
		const rs = new ReplayingState();
		rs.beginReplay();
		for (const ev of jobEvents().slice(0, 15)) rs.apply(ev);
		rs.beginReplay();
		rs.apply({
			type: 'job_finished',
			status: 'failed',
			report_url: null,
			error: 'server restarted'
		});
		expect(rs.state.runs).toHaveLength(2);
		expect(rs.state.finished?.status).toBe('failed');
		expect(rs.replaying).toBe(false);
	});
});

describe('helpers', () => {
	it('parseEvent accepts event objects only', () => {
		expect(parseEvent('{"type":"log","run_id":null,"level":"info","message":"x"}')?.type).toBe(
			'log'
		);
		expect(parseEvent('not json')).toBeNull();
		expect(parseEvent('[1,2]')).toBeNull();
		expect(parseEvent('{"no":"type"}')).toBeNull();
	});

	it('mergeRuns completes streamed runs with stored rows', () => {
		const live = buildState(jobEvents().slice(0, 7)).runs;
		const rows: RunRow[] = [
			{
				id: R1,
				job_id: 'j',
				algorithm: 'majority',
				backend: 'builtin',
				is_baseline: true,
				status: 'succeeded',
				valid_score: 0.12,
				valid_metrics: {},
				model_dir: null,
				error: null,
				duration_s: 0.1
			},
			{
				id: 'old',
				job_id: 'j',
				algorithm: 'ridge',
				backend: 'native',
				is_baseline: false,
				status: 'failed',
				valid_score: null,
				valid_metrics: {},
				model_dir: null,
				error: 'boom',
				duration_s: 1
			}
		];
		const merged = mergeRuns(live, rows);
		expect(merged.map((r) => r.run_id)).toEqual([R1, R2, 'old']);
		expect(merged[0].is_baseline).toBe(true);
		expect(merged[2]).toMatchObject({ status: 'failed', error: 'boom', index: 3 });
	});

	it('pushPoint keeps curves sorted', () => {
		const c = { steps: [] as number[], values: [] as number[] };
		for (const s of [3, 1, 2, 5, 4]) pushPoint(c, s, s * 10);
		expect(c.steps).toEqual([1, 2, 3, 4, 5]);
		expect(c.values).toEqual([10, 20, 30, 40, 50]);
	});

	it('alignSeries merges curves with different steps', () => {
		const a = { steps: [1, 2, 3], values: [1, 2, 3] };
		const b = { steps: [2, 4], values: [20, 40] };
		expect(alignSeries([a, b])).toEqual([
			[1, 2, 3, 4],
			[1, 2, 3, null],
			[null, 20, null, 40]
		]);
		const shared = alignSeries([a, { steps: [1, 2, 3], values: [9, 9, 9] }]);
		expect(shared[0]).toBe(a.steps);
	});
});
