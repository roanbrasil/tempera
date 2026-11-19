<script lang="ts">
	import MetricChart, { type Curve } from '#lib/components/MetricChart.svelte';
	import { api } from '#lib/api.js';
	import FeatureImportance from './FeatureImportance.svelte';
	import {
		importanceOf,
		isTerminalRun,
		splitPoints,
		type LivePoint,
		type RunState,
		type Series
	} from '#lib/events.js';

	let {
		runs,
		series,
		metricRev,
		selected,
		metric
	}: {
		runs: RunState[];
		series: Record<string, Record<string, Series>>;
		metricRev: number;
		selected: string | null;
		metric: string | null;
	} = $props();

	type Mode = 'run' | 'compare';
	let mode: Mode = $state('run');
	let compareMetric: string | null = $state(null);

	// Metrics of finished runs not seen on the stream (e.g. after a server restart) are loaded on demand.
	let fetched: Record<string, Record<string, Series>> = $state.raw({});
	let fetchedScalars: Record<string, Record<string, LivePoint>> = $state.raw({});
	let fetchedRev = $state(0);
	const loading: Record<string, boolean> = {};

	function curvesOf(id: string): Record<string, Series> | undefined {
		const s = series[id];
		return s && Object.keys(s).length > 0 ? s : fetched[id];
	}

	$effect(() => {
		const id = selected;
		if (!id || loading[id] || fetched[id]) return;
		const run = runs.find((r) => r.run_id === id);
		if (!run || !isTerminalRun(run.status)) return;
		if (Object.keys(run.live).length > 0) return; // streamed: everything is already here
		loading[id] = true;
		api
			.runMetrics(id)
			.then((points) => {
				const { curves, scalars } = splitPoints(points);
				fetched = { ...fetched, [id]: curves };
				fetchedScalars = { ...fetchedScalars, [id]: scalars };
				fetchedRev++;
			})
			.catch(() => {
				fetched = { ...fetched, [id]: {} };
			});
	});

	function order(name: string): number {
		if (name.startsWith('train/')) return 0;
		if (name.startsWith('valid/')) return 1;
		return 2;
	}

	const selectedRun = $derived(runs.find((r) => r.run_id === selected) ?? null);

	const allMetricNames = $derived.by(() => {
		void metricRev;
		void fetchedRev;
		const names: Record<string, true> = {};
		for (const r of runs) for (const n of Object.keys(curvesOf(r.run_id) ?? {})) names[n] = true;
		return Object.keys(names).sort((a, b) => order(a) - order(b) || a.localeCompare(b));
	});

	const defaultCompare = $derived(
		allMetricNames.find((n) => metric && n.endsWith('/' + metric)) ??
			allMetricNames.find((n) => n.startsWith('valid/')) ??
			allMetricNames[0] ??
			null
	);
	const cmp = $derived(
		compareMetric && allMetricNames.includes(compareMetric) ? compareMetric : defaultCompare
	);

	const curves: Curve[] = $derived.by(() => {
		void metricRev;
		void fetchedRev;
		if (mode === 'compare') {
			if (!cmp) return [];
			return runs
				.filter((r) => curvesOf(r.run_id)?.[cmp])
				.map((r) => ({ label: `${r.index}. ${r.algorithm}`, series: curvesOf(r.run_id)![cmp] }));
		}
		if (!selected) return [];
		const byName = curvesOf(selected) ?? {};
		return Object.keys(byName)
			.sort((a, b) => order(a) - order(b) || a.localeCompare(b))
			.map((n) => ({ label: n, series: byName[n] }));
	});

	const points = $derived(curves.reduce((n, c) => n + c.series.steps.length, 0));

	const importance = $derived.by(() => {
		void fetchedRev;
		if (!selectedRun) return [];
		const fromStream = importanceOf(selectedRun.live);
		return fromStream.length ? fromStream : importanceOf(fetchedScalars[selectedRun.run_id] ?? {});
	});

	/** The selected run is done (or reports only scalars) and has no step-wise metric. */
	const noCurve = $derived(
		mode === 'run' &&
			selectedRun !== null &&
			curves.length === 0 &&
			(isTerminalRun(selectedRun.status) || Object.keys(selectedRun.live).length > 0) &&
			(!isTerminalRun(selectedRun.status) ||
				Object.keys(selectedRun.live).length > 0 ||
				fetched[selectedRun.run_id] !== undefined)
	);
</script>

<div class="panel">
	<div class="bar">
		<div class="seg" role="tablist" aria-label="Chart mode">
			<button role="tab" aria-selected={mode === 'run'} onclick={() => (mode = 'run')}>
				This run
			</button>
			<button role="tab" aria-selected={mode === 'compare'} onclick={() => (mode = 'compare')}>
				Compare runs
			</button>
		</div>
		{#if mode === 'run'}
			<span class="what">
				{#if selectedRun}<span class="mono">
						{selectedRun.index}. {selectedRun.algorithm}
					</span>{:else}<span class="muted">Select a run</span>{/if}
			</span>
		{:else}
			<select
				aria-label="Metric to compare"
				value={cmp ?? ''}
				onchange={(e) => (compareMetric = e.currentTarget.value)}
			>
				{#each allMetricNames as n (n)}<option value={n}>{n}</option>{/each}
			</select>
		{/if}
		<span class="spacer"></span>
		<span class="faint small num">{points.toLocaleString('en-US')} points</span>
	</div>
	{#if noCurve}
		<div class="nocurve muted small">
			This algorithm reports no training curve; see its score above.
		</div>
	{:else}
		<MetricChart {curves} rev={metricRev + fetchedRev} height={300} />
	{/if}
	{#if mode === 'run' && importance.length > 0}
		<FeatureImportance items={importance} />
	{/if}
</div>

<style>
	.nocurve {
		display: grid;
		place-items: center;
		text-align: center;
		min-height: 120px;
		padding: 1rem;
		border: 1px dashed var(--border-strong);
		border-radius: var(--radius-sm);
	}
	.panel {
		display: grid;
		gap: 0.6rem;
		min-width: 0;
	}
	.bar {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
	}
	.seg {
		display: inline-flex;
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}
	.seg button {
		font: inherit;
		font-size: 0.85rem;
		padding: 0.3rem 0.75rem;
		border: none;
		background: var(--surface);
		color: var(--muted);
		cursor: pointer;
	}
	.seg button + button {
		border-left: 1px solid var(--border-strong);
	}
	.seg button[aria-selected='true'] {
		background: var(--accent-soft);
		color: var(--text);
		font-weight: 600;
	}
	.what {
		font-weight: 600;
		min-width: 0;
		overflow-wrap: anywhere;
	}
	select {
		min-height: 32px;
		padding: 0.2rem 0.5rem;
		font-size: 0.85rem;
	}
</style>
