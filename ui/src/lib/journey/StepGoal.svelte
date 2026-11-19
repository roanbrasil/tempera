<script lang="ts">
	import type { Journey } from './journey.svelte';
	import Ticks from '#lib/components/Ticks.svelte';
	import { getIn } from '#lib/spec-yaml.js';
	import { fmtPct, metricLabel, taskLabel } from '#lib/format.js';
	import type { ColumnProfile } from '#lib/types.js';

	let { j }: { j: Journey } = $props();

	const NO_TARGET = '__none__';

	const columns = $derived(j.profile?.columns ?? []);

	/** Columns that look like a label: booleans, small categoricals, then the last column. */
	const suggested = $derived.by(() => {
		const score = (c: ColumnProfile, i: number) => {
			let s = 0;
			if (c.kind === 'boolean') s += 3;
			if (c.kind === 'categorical' && c.n_unique <= 20) s += 2;
			if (/target|label|churn|class|outcome|default|fraud|price|y$/i.test(c.name)) s += 3;
			if (i === columns.length - 1) s += 1;
			if (c.kind === 'high_cardinality' || c.kind === 'constant' || c.kind === 'datetime') s -= 5;
			return s;
		};
		return columns
			.map((c, i) => ({ c, s: score(c, i) }))
			.filter((x) => x.s > 0)
			.sort((a, b) => b.s - a.s)
			.slice(0, 3)
			.map((x) => x.c.name);
	});

	const selectValue = $derived(
		j.target === undefined ? '' : j.target === null ? NO_TARGET : j.target
	);

	function choose(v: string) {
		if (!v) return;
		j.chooseTarget(v === NO_TARGET ? null : v);
	}

	const timeColumn = $derived((getIn(j.yaml, ['data', 'time_column']) as string) ?? '');
	const groupColumn = $derived((getIn(j.yaml, ['data', 'group_column']) as string) ?? '');
	const datetimeCols = $derived(columns.filter((c) => c.kind === 'datetime'));
	const otherCols = $derived(columns.filter((c) => c.name !== j.target));

	async function setOptional(key: 'time_column' | 'group_column', v: string) {
		if (j.set(['data', key], v || null)) await j.runAudit();
	}

	function rename(v: string) {
		j.name = v;
		if (j.yaml) j.set(['name'], v || 'job');
	}
</script>

<section class="stack">
	<div>
		<h2>2 · Goal</h2>
		<p class="muted">What do you want to predict? Pick the column with the answer you want.</p>
	</div>

	<div class="card stack">
		<div class="grid2">
			<label class="field">
				Target column
				<select
					value={selectValue}
					onchange={(e) => choose(e.currentTarget.value)}
					disabled={j.drafting}
				>
					<option value="" disabled>Choose a column…</option>
					{#each columns as c (c.name)}
						<option value={c.name}>{c.name} · {c.kind.replace('_', ' ')}</option>
					{/each}
					<option value={NO_TARGET}>No target — find groups (clustering)</option>
				</select>
			</label>
			<label class="field">
				Job name
				<input
					type="text"
					value={j.name}
					onchange={(e) => rename(e.currentTarget.value.trim())}
					placeholder="customer-churn"
					spellcheck="false"
				/>
			</label>
		</div>

		{#if suggested.length > 0}
			<div class="row small">
				<span class="muted">Suggested:</span>
				{#each suggested as s (s)}
					<button
						class="chip"
						class:on={j.target === s}
						onclick={() => choose(s)}
						disabled={j.drafting}
					>
						{s}
					</button>
				{/each}
				<button
					class="chip"
					class:on={j.target === null}
					onclick={() => choose(NO_TARGET)}
					disabled={j.drafting}
				>
					no target
				</button>
			</div>
		{/if}
	</div>

	{#if j.goalError}
		<div class="alert danger" role="alert">{j.goalError}</div>
	{/if}

	{#if j.drafting || (j.auditing && !j.inference)}
		<div class="card row">
			<span class="spinner"></span> Drafting the spec and inferring the task…
		</div>
	{:else if j.inference}
		<div class="card infer">
			<div class="pair">
				<div>
					<div class="k">Task</div>
					<div class="big">{taskLabel(j.inference.task)}</div>
				</div>
				<div>
					<div class="k">Primary metric</div>
					<div class="big">{metricLabel(j.inference.metric)}</div>
				</div>
				{#if j.inference.minority_share !== undefined}
					<div>
						<div class="k">Rarest class</div>
						<div class="big num">{fmtPct(j.inference.minority_share)}</div>
					</div>
				{/if}
			</div>
			<div>
				<div class="k">Why</div>
				<ul class="reasons">
					{#each j.inference.reasons as r, i (i)}<li><Ticks text={r} /></li>{/each}
				</ul>
			</div>
		</div>

		<details class="card opt">
			<summary>
				<strong>Optional</strong> <span class="muted">· time and group columns</span>
			</summary>
			<div class="grid2">
				<label class="field">
					Time column
					<span class="hint">
						Set it if you will predict future rows: the split becomes temporal.
					</span>
					<select
						value={timeColumn}
						onchange={(e) => setOptional('time_column', e.currentTarget.value)}
					>
						<option value="">None</option>
						{#each datetimeCols as c (c.name)}<option value={c.name}>{c.name}</option>{/each}
					</select>
				</label>
				<label class="field">
					Group column
					<span class="hint">
						Rows of the same entity (e.g. a customer) never split across train and test.
					</span>
					<select
						value={groupColumn}
						onchange={(e) => setOptional('group_column', e.currentTarget.value)}
					>
						<option value="">None</option>
						{#each otherCols as c (c.name)}<option value={c.name}>{c.name}</option>{/each}
					</select>
				</label>
			</div>
		</details>
	{/if}
</section>

<style>
	.grid2 {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr));
		gap: 1rem;
	}
	.chip {
		font: inherit;
		font-size: 0.82rem;
		padding: 0.15rem 0.65rem;
		border-radius: 999px;
		border: 1px solid var(--border-strong);
		background: var(--surface);
		color: var(--text);
		cursor: pointer;
		font-family: var(--mono);
	}
	.chip:hover:not(:disabled) {
		border-color: var(--accent);
	}
	.chip.on {
		background: var(--accent);
		color: var(--accent-contrast);
		border-color: var(--accent);
	}
	.infer {
		display: grid;
		gap: 1rem;
		border-left: 4px solid var(--accent);
	}
	.pair {
		display: flex;
		gap: 1rem 2.5rem;
		flex-wrap: wrap;
	}
	.k {
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--faint);
	}
	.big {
		font-size: 1.15rem;
		font-weight: 650;
	}
	.reasons {
		margin: 0.25rem 0 0;
		padding-left: 1.2rem;
	}
	.reasons li {
		margin-bottom: 0.2rem;
	}
	.opt summary {
		cursor: pointer;
	}
	.opt[open] summary {
		margin-bottom: 1rem;
	}
</style>
