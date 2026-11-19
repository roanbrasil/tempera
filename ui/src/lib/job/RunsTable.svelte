<script lang="ts">
	import { isTerminalRun, liveValidMetric, runProgress, type RunState } from '#lib/events.js';
	import StatusBadge from '#lib/components/StatusBadge.svelte';
	import { fmtDuration, fmtNum, higherIsBetter, metricLabel } from '#lib/format.js';

	let {
		runs,
		selected,
		onselect,
		metric,
		winnerId,
		totalRuns
	}: {
		runs: RunState[];
		selected: string | null;
		onselect: (id: string) => void;
		metric: string | null;
		winnerId: string | null;
		totalRuns: number | null;
	} = $props();

	const higher = $derived(higherIsBetter(metric));
	const baseline = $derived(runs.find((r) => r.is_baseline && r.valid_score !== null) ?? null);
	const best = $derived.by(() => {
		let b: RunState | null = null;
		for (const r of runs) {
			if (r.valid_score === null || r.status !== 'succeeded') continue;
			if (!b || (higher ? r.valid_score > b.valid_score! : r.valid_score < b.valid_score!)) b = r;
		}
		return b?.run_id ?? null;
	});
	const pending = $derived(Math.max(0, (totalRuns ?? 0) - runs.length));

	function delta(r: RunState): { text: string; good: boolean } | null {
		if (!baseline || r.run_id === baseline.run_id || r.valid_score === null) return null;
		const d = r.valid_score - baseline.valid_score!;
		const good = higher ? d > 0 : d < 0;
		return { text: (d >= 0 ? '+' : '−') + fmtNum(Math.abs(d)), good };
	}
</script>

<div class="runs">
	<table class="data">
		<thead>
			<tr>
				<th class="idx">#</th>
				<th>Algorithm</th>
				<th>Progress</th>
				<th class="right">Live valid</th>
				<th class="right">Score <span class="faint">({metricLabel(metric)})</span></th>
				<th class="right">vs baseline</th>
				<th class="right">Time</th>
				<th>Status</th>
			</tr>
		</thead>
		<tbody>
			{#each runs as r (r.run_id)}
				{@const p = runProgress(r)}
				{@const live = liveValidMetric(r, metric)}
				{@const d = delta(r)}
				<tr class:sel={selected === r.run_id} class:winner={winnerId === r.run_id}>
					<td class="idx num faint">{r.index}</td>
					<td class="algo">
						<button
							class="pick"
							aria-pressed={selected === r.run_id}
							title="Show the curves of this run"
							onclick={() => onselect(r.run_id)}
						>
							<span class="mono">{r.algorithm}</span>
						</button>
						<span class="tags">
							<span class="badge">{r.backend}</span>
							{#if r.is_baseline}<span class="badge info">baseline</span>{/if}
							{#if winnerId === r.run_id}<span class="badge accent">
									★ winner
								</span>{:else if !winnerId && best === r.run_id}<span class="badge ok">
									best so far
								</span>{/if}
						</span>
					</td>
					<td class="prog" data-label="Progress">
						<div
							class="progress"
							class:indeterminate={p === null && !isTerminalRun(r.status)}
							role="progressbar"
							aria-valuemin="0"
							aria-valuemax="100"
							aria-valuenow={p === null ? undefined : Math.round(p * 100)}
						>
							<span style:width="{(p ?? 0) * 100}%" class:failed={r.status === 'failed'}></span>
						</div>
						<span class="faint small num">
							{#if isTerminalRun(r.status)}
								{r.step !== null && r.step > 1 ? `${r.step} steps` : ''}
							{:else if r.total_steps}
								{r.step ?? 0}/{r.total_steps} · ETA {fmtDuration(r.eta_s)}
							{:else}
								starting…
							{/if}
						</span>
					</td>
					<td class="right num" data-label="Live valid" title={live?.name ?? ''}>
						{live ? fmtNum(live.value) : '—'}
					</td>
					<td class="right num score" data-label="Score">{fmtNum(r.valid_score)}</td>
					<td class="right num" data-label="vs baseline">
						{#if d}<span class:good={d.good} class:bad={!d.good}>{d.text}</span>{:else}<span
								class="faint"
							>
								—
							</span>{/if}
					</td>
					<td class="right num" data-label="Time">{fmtDuration(r.duration_s)}</td>
					<td data-label="Status">
						<StatusBadge status={r.status} />
						{#if r.error}<div class="err small" title={r.error}>{r.error}</div>{/if}
					</td>
				</tr>
			{/each}
			{#each { length: pending }, i (i)}
				<tr class="pending">
					<td class="idx num faint">{runs.length + i + 1}</td>
					<td colspan="7" class="faint small">waiting…</td>
				</tr>
			{/each}
		</tbody>
	</table>
	{#if runs.length === 0 && pending === 0}
		<p class="muted small none">Runs appear here as soon as training starts.</p>
	{/if}
</div>

<style>
	.runs {
		overflow-x: auto;
	}
	tbody tr:not(.pending) {
		transition: background 0.12s;
	}
	tbody tr:not(.pending):hover {
		background: var(--surface-2);
	}
	tr.sel {
		background: var(--accent-soft) !important;
		box-shadow: inset 3px 0 0 var(--accent);
	}
	.idx {
		width: 2rem;
	}
	.algo {
		min-width: 11rem;
	}
	.pick {
		all: unset;
		cursor: pointer;
		font-weight: 600;
	}
	.pick:focus-visible {
		outline: 2px solid var(--focus);
		outline-offset: 2px;
		border-radius: 3px;
	}
	.tags {
		display: flex;
		gap: 0.25rem;
		flex-wrap: wrap;
		margin-top: 0.2rem;
	}
	.tags .badge {
		font-size: 0.68rem;
		padding: 0 0.45rem;
	}
	.prog {
		min-width: 10rem;
	}
	.prog .progress {
		margin-bottom: 0.25rem;
	}
	.progress span.failed {
		background: var(--danger);
	}
	.score {
		font-weight: 650;
	}
	.good {
		color: var(--ok);
		font-weight: 600;
	}
	.bad {
		color: var(--danger);
	}
	.err {
		color: var(--danger);
		max-width: 16rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.none {
		padding: 0.5rem 0.75rem;
		margin: 0;
	}

	@media (max-width: 760px) {
		.runs {
			overflow: visible;
		}
		thead {
			display: none;
		}
		table,
		tbody,
		tr,
		td {
			display: block;
			width: 100%;
		}
		tbody tr:not(.pending) {
			display: grid;
			grid-template-columns: repeat(4, minmax(0, 1fr));
			gap: 0.45rem 0.75rem;
			padding: 0.75rem 0.6rem;
			border-bottom: 1px solid var(--border);
		}
		tr.pending {
			display: flex;
			gap: 0.75rem;
			padding: 0.5rem 0.6rem;
		}
		td {
			border: none !important;
			padding: 0 !important;
			text-align: left !important;
			min-width: 0 !important;
		}
		td.idx {
			display: none;
		}
		tr.pending td.idx {
			display: block;
			width: auto;
		}
		td.algo {
			grid-column: 1 / 4;
		}
		td.prog {
			grid-column: 1 / -1;
			order: 3;
		}
		td[data-label='Status'] {
			grid-column: 4;
			grid-row: 1;
			justify-self: end;
			width: auto;
		}
		td[data-label]:not(.prog):not([data-label='Status'])::before {
			content: attr(data-label);
			display: block;
			font-size: 0.68rem;
			color: var(--faint);
			text-transform: uppercase;
			letter-spacing: 0.04em;
		}
		td[data-label]:not(.prog):not([data-label='Status']) {
			order: 4;
		}
	}
</style>
