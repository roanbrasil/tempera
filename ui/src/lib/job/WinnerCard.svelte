<script lang="ts">
	import type { Evaluation } from '#lib/types.js';
	import type { RunState } from '#lib/events.js';
	import { fmtDuration, fmtInt, fmtNum, metricLabel } from '#lib/format.js';

	let {
		algorithm,
		test,
		metric,
		run,
		reportHref
	}: {
		algorithm: string;
		test: Evaluation;
		metric: string | null;
		run: RunState | null;
		reportHref: string;
	} = $props();

	const primary = $derived(metric && test.metrics[metric] !== undefined ? metric : null);
	const others = $derived(Object.entries(test.metrics).filter(([k]) => k !== primary));
	const confusion = $derived(
		test.confusion && test.classes && test.classes.length <= 6 ? test.confusion : null
	);
	const cmax = $derived(Math.max(1, ...(confusion ?? []).flat()));
</script>

<div class="card winner">
	<div class="top">
		<div class="trophy" aria-hidden="true">★</div>
		<div class="who">
			<div class="k">Winner</div>
			<div class="algo mono">{algorithm}</div>
			{#if run}
				<div class="faint small">
					{run.backend} · validation {fmtNum(run.valid_score)} · trained in {fmtDuration(
						run.duration_s
					)}
				</div>
			{/if}
		</div>
		{#if primary}
			<div class="primary">
				<div class="k">Test {metricLabel(primary)}</div>
				<div class="big num">{fmtNum(test.metrics[primary])}</div>
			</div>
		{/if}
	</div>

	<div class="body">
		<dl class="metrics">
			{#each others as [k, v] (k)}
				<div>
					<dt>{metricLabel(k)}</dt>
					<dd class="num">{fmtNum(v)}</dd>
				</div>
			{/each}
			<div>
				<dt>Test rows</dt>
				<dd class="num">{fmtInt(test.rows)}</dd>
			</div>
		</dl>

		{#if confusion && test.classes}
			<div class="cm">
				<div class="k">Confusion matrix <span class="faint">(rows = truth)</span></div>
				<table style:--n={test.classes.length}>
					<thead>
						<tr>
							<th></th>
							{#each test.classes as c (c)}<th class="mono">{c}</th>{/each}
						</tr>
					</thead>
					<tbody>
						{#each confusion as row, i (i)}
							<tr>
								<th class="mono">{test.classes[i]}</th>
								{#each row as v, k (k)}
									<td class="num" class:diag={i === k} style:--a={(v / cmax).toFixed(3)}>
										{fmtInt(v)}
									</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>

	<p class="small muted test-note">
		Scored once on the held-out test set. The full report has the curves, feature importance and
		calibration.
	</p>
	<div class="row">
		<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- server endpoint, not an app route -->
		<a class="btn primary" href={reportHref} target="_blank" rel="noopener">Open the report ↗</a>
	</div>
</div>

<style>
	.winner {
		border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--border));
		background:
			linear-gradient(135deg, color-mix(in srgb, var(--straw) 22%, transparent), transparent 55%),
			var(--surface);
		display: grid;
		gap: 1rem;
	}
	.top {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex-wrap: wrap;
	}
	.trophy {
		width: 3rem;
		height: 3rem;
		border-radius: 50%;
		display: grid;
		place-items: center;
		font-size: 1.5rem;
		color: #fff;
		background: linear-gradient(135deg, var(--bronze), var(--purple));
		flex: none;
	}
	.who {
		flex: 1;
		min-width: 10rem;
	}
	.k {
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--faint);
	}
	.algo {
		font-size: 1.35rem;
		font-weight: 700;
		overflow-wrap: anywhere;
	}
	.primary {
		text-align: right;
	}
	.big {
		font-size: 2rem;
		font-weight: 700;
		line-height: 1.1;
		color: var(--accent-strong);
	}
	.body {
		display: flex;
		gap: 1.5rem;
		flex-wrap: wrap;
		align-items: flex-start;
	}
	.metrics {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(8.5rem, 1fr));
		gap: 0.6rem 1rem;
		margin: 0;
		flex: 1;
		min-width: min(100%, 16rem);
	}
	dt {
		font-size: 0.78rem;
		color: var(--muted);
	}
	dd {
		margin: 0;
		font-weight: 650;
	}
	.cm table {
		border-collapse: collapse;
		margin-top: 0.3rem;
		font-size: 0.82rem;
	}
	.cm th {
		font-weight: 500;
		color: var(--muted);
		padding: 0.2rem 0.45rem;
		text-align: right;
		max-width: 6rem;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.cm td {
		width: 3.6rem;
		height: 2.4rem;
		text-align: center;
		border: 1px solid var(--surface);
		background: color-mix(in srgb, var(--danger) calc(var(--a) * 55%), var(--surface-2));
	}
	.cm td.diag {
		background: color-mix(in srgb, var(--ok) calc(var(--a) * 60%), var(--surface-2));
		font-weight: 650;
	}
	.test-note {
		margin: 0;
	}
	@media (max-width: 520px) {
		.primary {
			text-align: left;
			width: 100%;
		}
	}
</style>
