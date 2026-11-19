<script lang="ts">
	// Tiny inline bar chart (profile histograms and top values).
	let {
		bins,
		width = 120,
		height = 28,
		label = 'Distribution'
	}: { bins: number[]; width?: number; height?: number; label?: string } = $props();

	const max = $derived(Math.max(1, ...bins));
	const bw = $derived(bins.length > 0 ? width / bins.length : width);
</script>

<svg {width} {height} viewBox="0 0 {width} {height}" role="img" aria-label={label}>
	{#each bins as b, i (i)}
		{@const h = b === 0 ? 0 : Math.max(1.5, (b / max) * (height - 2))}
		<rect x={i * bw + 0.5} y={height - h} width={Math.max(1, bw - 1.5)} height={h} rx="1" />
	{/each}
	<line x1="0" x2={width} y1={height - 0.5} y2={height - 0.5} />
</svg>

<style>
	svg {
		display: block;
		max-width: 100%;
	}
	rect {
		fill: var(--series-1);
		opacity: 0.85;
	}
	line {
		stroke: var(--border-strong);
		stroke-width: 1;
	}
</style>
