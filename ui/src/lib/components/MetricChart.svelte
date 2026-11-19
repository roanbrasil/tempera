<script lang="ts" module>
	import type { Series } from '#lib/events.js';

	export interface Curve {
		label: string;
		series: Series;
	}
</script>

<script lang="ts">
	// Live line chart (uPlot). Redraws when `rev` changes; recreated only when the set of curves
	// changes, so streaming thousands of points stays cheap.
	import uPlot from 'uplot';
	import 'uplot/dist/uPlot.min.css';
	import { onMount, untrack } from 'svelte';
	import { alignSeries } from '#lib/chart-data.js';
	import { fmtNum } from '#lib/format.js';

	let {
		curves,
		rev,
		height = 280,
		xLabel = 'step'
	}: { curves: Curve[]; rev: number; height?: number; xLabel?: string } = $props();

	let host: HTMLDivElement;
	let plot: uPlot | null = null;
	let builtFor = '';
	let width = $state(0);
	let dark = $state(false);

	const key = $derived(curves.map((c) => c.label).join('\u0000'));
	const empty = $derived(curves.every((c) => c.series.steps.length === 0));

	function palette(): string[] {
		const cs = getComputedStyle(host);
		return Array.from({ length: 8 }, (_, i) => cs.getPropertyValue(`--series-${i + 1}`).trim());
	}

	const ready = $derived(width > 0);

	function build() {
		plot?.destroy();
		plot = null;
		const w = untrack(() => width);
		if (!host || w === 0) return;
		const cs = getComputedStyle(host);
		const grid = cs.getPropertyValue('--border').trim();
		const axis = cs.getPropertyValue('--muted').trim();
		const colors = palette();
		const axisCommon = {
			stroke: axis,
			grid: { stroke: grid, width: 1 },
			ticks: { stroke: grid, width: 1 },
			font: '11px system-ui, sans-serif'
		};
		const opts: uPlot.Options = {
			width: w,
			height,
			pxAlign: false,
			cursor: { drag: { x: true, y: false }, points: { size: 6 } },
			legend: { live: true },
			scales: { x: { time: false } },
			axes: [
				{ ...axisCommon, label: xLabel, labelSize: 18, labelFont: '11px system-ui, sans-serif' },
				{ ...axisCommon, size: 56, values: (_u, ticks) => ticks.map((t) => fmtNum(t, 3)) }
			],
			series: [
				{ label: xLabel, value: (_u, v) => (v === null ? '—' : String(v)) },
				...curves.map((c, i) => ({
					label: c.label,
					stroke: colors[i % colors.length],
					width: 1.75,
					spanGaps: true,
					points: { show: false },
					value: (_u: uPlot, v: number | null) => (v === null ? '—' : fmtNum(v))
				}))
			]
		};
		plot = new uPlot(opts, alignSeries(curves.map((c) => c.series)) as uPlot.AlignedData, host);
		builtFor = key + (dark ? ':dark' : ':light');
	}

	// Rebuild when the curve set, size class or theme changes; otherwise only push new data.
	$effect(() => {
		void rev;
		if (!ready) return;
		const want = key + (dark ? ':dark' : ':light');
		if (!plot || builtFor !== want) {
			build();
			return;
		}
		plot.setData(alignSeries(curves.map((c) => c.series)) as uPlot.AlignedData);
	});

	$effect(() => {
		if (plot && width > 0) plot.setSize({ width, height });
	});

	onMount(() => {
		const mq = window.matchMedia('(prefers-color-scheme: dark)');
		dark = mq.matches;
		const onScheme = () => (dark = mq.matches);
		mq.addEventListener('change', onScheme);
		const ro = new ResizeObserver((entries) => {
			const w = Math.floor(entries[0].contentRect.width);
			if (w !== width) width = w;
		});
		ro.observe(host);
		return () => {
			ro.disconnect();
			mq.removeEventListener('change', onScheme);
			plot?.destroy();
			plot = null;
		};
	});
</script>

<div class="chart" style:min-height="{height}px">
	<div class="host" bind:this={host}></div>
	{#if empty}
		<div class="empty muted small">Waiting for metrics…</div>
	{/if}
</div>

<style>
	.chart {
		position: relative;
		width: 100%;
		min-width: 0;
	}
	.host {
		width: 100%;
	}
	.empty {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		pointer-events: none;
	}
	.chart :global(.uplot) {
		font-family: var(--font);
		color: var(--text);
	}
	.chart :global(.u-legend) {
		font-size: 0.8rem;
		text-align: left;
		margin-top: 0.25rem;
	}
	.chart :global(.u-legend .u-series) {
		padding: 0.1rem 0.4rem;
	}
	.chart :global(.u-legend th) {
		font-weight: 500;
		color: var(--muted);
	}
	.chart :global(.u-legend td) {
		font-variant-numeric: tabular-nums;
	}
	.chart :global(.u-select) {
		background: color-mix(in srgb, var(--accent) 18%, transparent);
	}
	.chart :global(.u-cursor-x),
	.chart :global(.u-cursor-y) {
		border-color: var(--faint) !important;
	}
</style>
