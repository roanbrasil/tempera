<script lang="ts">
	import type { ColumnProfile, Profile } from '#lib/types.js';
	import Histogram from '#lib/components/Histogram.svelte';
	import { fmtInt, fmtNum, fmtPct } from '#lib/format.js';

	let { profile, highlight = null }: { profile: Profile; highlight?: string | null } = $props();

	let showSample = $state(false);

	const KIND_TONE: Record<string, string> = {
		numeric: 'info',
		boolean: 'ok',
		categorical: 'accent',
		datetime: 'info',
		text: '',
		high_cardinality: 'warn',
		constant: 'warn'
	};

	function topShare(c: ColumnProfile): { value: string; share: number }[] {
		const nonNull = Math.max(1, profile.rows - c.null_count);
		return (c.top_values ?? [])
			.slice(0, 4)
			.map((t) => ({ value: t.value, share: t.count / nonNull }));
	}

	function shortDate(s: string | undefined): string {
		return s ? s.slice(0, 10) : '?';
	}

	const sampleCols = $derived(profile.columns.map((c) => c.name));
	const sampleRows = $derived(profile.sample.slice(0, 8));
</script>

<div class="card prof">
	<div class="row title">
		<h3>Columns</h3>
		<span class="spacer"></span>
		<button class="btn small ghost" onclick={() => (showSample = !showSample)}>
			{showSample ? 'Hide sample rows' : 'Show sample rows'}
		</button>
	</div>

	{#if showSample}
		<div class="table-wrap sample">
			<table class="data">
				<thead>
					<tr>
						{#each sampleCols as c (c)}<th>{c}</th>{/each}
					</tr>
				</thead>
				<tbody>
					{#each sampleRows as r, i (i)}
						<tr>
							{#each sampleCols as c (c)}
								<td class="mono nowrap">
									{r[c] === null || r[c] === undefined ? '∅' : String(r[c])}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}

	<table class="data cols">
		<thead>
			<tr>
				<th>Column</th>
				<th>Kind</th>
				<th>Nulls</th>
				<th class="right">Distinct</th>
				<th>Distribution</th>
			</tr>
		</thead>
		<tbody>
			{#each profile.columns as c (c.name)}
				<tr class:hl={highlight === c.name}>
					<td class="name">
						<strong>{c.name}</strong>
						<span class="faint mono small">{c.dtype}</span>
					</td>
					<td data-label="Kind">
						<span class="badge {KIND_TONE[c.kind] ?? ''}">{c.kind.replace('_', ' ')}</span>
					</td>
					<td data-label="Nulls" class="nulls">
						<span class="num" class:warnc={c.null_fraction > 0.2}>{fmtPct(c.null_fraction)}</span>
						<span class="bar" aria-hidden="true">
							<span style:width="{Math.max(c.null_fraction > 0 ? 3 : 0, c.null_fraction * 100)}%"
							></span>
						</span>
					</td>
					<td data-label="Distinct" class="right num">{fmtInt(c.n_unique)}</td>
					<td data-label="Distribution" class="dist">
						{#if c.numeric}
							<div class="numeric">
								<Histogram bins={c.numeric.histogram} label="Histogram of {c.name}" />
								<span class="faint small num">
									{fmtNum(c.numeric.min, 2)} … {fmtNum(c.numeric.max, 2)} · μ {fmtNum(
										c.numeric.mean,
										2
									)}
								</span>
							</div>
						{:else if c.kind === 'datetime'}
							<span class="small">{shortDate(c.min_time)} → {shortDate(c.max_time)}</span>
						{:else if c.top_values && c.top_values.length > 0 && c.kind !== 'high_cardinality'}
							<div class="tops">
								{#each topShare(c) as t (t.value)}
									<div class="top">
										<span class="tv mono" title={t.value}>{t.value}</span>
										<span class="tb"><span style:width="{t.share * 100}%"></span></span>
										<span class="faint small num">{fmtPct(t.share, 0)}</span>
									</div>
								{/each}
							</div>
						{:else if c.kind === 'high_cardinality'}
							<span class="small muted">≈ one value per row (identifier?)</span>
						{:else if c.mean_length}
							<span class="small muted">text · mean length {fmtNum(c.mean_length, 0)}</span>
						{:else}
							<span class="faint">—</span>
						{/if}
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.prof {
		padding: 0.9rem 0.9rem 0.4rem;
	}
	.title {
		padding: 0 0.3rem 0.4rem;
	}
	.title h3 {
		margin: 0;
	}
	.sample {
		margin-bottom: 1rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
	}
	.sample td {
		font-size: 0.8rem;
	}
	.name {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.name strong {
		overflow-wrap: anywhere;
	}
	tr.hl {
		background: var(--accent-soft);
	}
	.nulls {
		min-width: 6.5rem;
	}
	.nulls .bar {
		display: block;
		height: 4px;
		margin-top: 3px;
		background: var(--surface-3);
		border-radius: 2px;
		max-width: 5rem;
		overflow: hidden;
	}
	.nulls .bar span {
		display: block;
		height: 100%;
		background: var(--warn);
	}
	.warnc {
		color: var(--warn);
		font-weight: 600;
	}
	.numeric {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
	}
	.tops {
		display: grid;
		gap: 2px;
		min-width: 12rem;
		max-width: 22rem;
	}
	.top {
		display: grid;
		grid-template-columns: minmax(3rem, 7rem) 1fr 2.6rem;
		align-items: center;
		gap: 0.5rem;
	}
	.tv {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 0.8rem;
	}
	.tb {
		height: 6px;
		background: var(--surface-3);
		border-radius: 3px;
		overflow: hidden;
	}
	.tb span {
		display: block;
		height: 100%;
		background: var(--series-2);
	}

	@media (max-width: 720px) {
		.prof {
			padding: 0.75rem 0.5rem 0.25rem;
		}
		.cols thead {
			display: none;
		}
		.cols,
		.cols tbody,
		.cols tr,
		.cols td {
			display: block;
			width: 100%;
		}
		.cols tr {
			display: grid;
			grid-template-columns: 1fr auto;
			gap: 0.3rem 0.75rem;
			padding: 0.7rem 0.4rem;
			border-bottom: 1px solid var(--border);
		}
		.cols tr:last-child {
			border-bottom: none;
		}
		.cols td {
			padding: 0;
			border: none;
		}
		.cols td.right {
			text-align: left;
		}
		.cols td.dist {
			grid-column: 1 / -1;
		}
		.cols td.nulls,
		.cols td[data-label='Distinct'] {
			display: flex;
			gap: 0.4rem;
			align-items: center;
		}
		.cols td.nulls::before,
		.cols td[data-label='Distinct']::before {
			content: attr(data-label);
			font-size: 0.72rem;
			color: var(--faint);
			text-transform: uppercase;
			letter-spacing: 0.04em;
		}
		.nulls .bar {
			display: none;
		}
		.tops {
			min-width: 0;
		}
	}
</style>
