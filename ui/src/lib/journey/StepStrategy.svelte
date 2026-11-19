<script lang="ts">
	import type { Journey } from './journey.svelte';
	import { getIn } from '#lib/spec-yaml.js';
	import { taskLabel } from '#lib/format.js';

	let { j }: { j: Journey } = $props();

	type Mode = 'auto' | 'pick' | 'all';

	const task = $derived(j.inference?.task ?? null);
	const candidatesRaw = $derived(getIn(j.yaml, ['strategy', 'candidates']));
	const picked = $derived(
		Array.isArray(candidatesRaw) ? candidatesRaw.map(String).filter((c) => c !== 'all') : []
	);
	const mode: Mode = $derived(
		Array.isArray(candidatesRaw)
			? candidatesRaw.length === 1 && candidatesRaw[0] === 'all'
				? 'all'
				: 'pick'
			: 'auto'
	);
	// While the user is in "pick" mode with nothing checked yet, the YAML still says `auto`.
	let pickMode = $state(false);
	const effectiveMode: Mode = $derived(pickMode && mode === 'auto' ? 'pick' : mode);

	/** Algorithms supporting the task, one entry per key with all its backends. */
	const algos = $derived.by(() => {
		const map: Record<string, string[]> = {};
		for (const a of j.algorithms ?? []) {
			if (task && !a.tasks.includes(task)) continue;
			(map[a.key] ??= []).push(a.backend);
		}
		return Object.entries(map)
			.map(([key, backends]) => ({ key, backends }))
			.sort((a, b) => a.key.localeCompare(b.key));
	});

	const baseline = $derived(getIn(j.yaml, ['strategy', 'baseline']) !== false);
	const splitStrategy = $derived((getIn(j.yaml, ['split', 'strategy']) as string) ?? 'auto');
	const testSize = $derived(Number(getIn(j.yaml, ['split', 'test_size']) ?? 0.2));
	const seed = $derived(Number(getIn(j.yaml, ['split', 'seed']) ?? 42));
	const maxTime = $derived((getIn(j.yaml, ['budget', 'max_time']) as string) ?? '');
	const maxCost = $derived(getIn(j.yaml, ['budget', 'max_cost', 'amount']) as number | undefined);

	function setMode(m: Mode) {
		pickMode = m === 'pick';
		if (m === 'auto') j.set(['strategy', 'candidates'], null);
		else if (m === 'all') j.set(['strategy', 'candidates'], ['all']);
		else if (picked.length === 0) j.set(['strategy', 'candidates'], null);
	}

	function toggleAlgo(key: string, on: boolean) {
		const next = on ? [...picked, key] : picked.filter((k) => k !== key);
		j.set(['strategy', 'candidates'], next.length ? next : null);
	}

	function setCost(v: string) {
		const n = Number(v);
		j.set(['budget', 'max_cost'], v && n > 0 ? { amount: n, currency: 'USD' } : null);
	}

	const TIMES = ['', '10m', '30m', '1h', '2h', '6h', '12h', '1d'];
	const SPLITS = [
		['auto', 'Auto — chosen from the audit'],
		['stratified', 'Stratified random'],
		['random', 'Random'],
		['group', 'By group'],
		['temporal', 'Temporal']
	] as const;
</script>

<section class="stack">
	<div>
		<h2>4 · Strategy</h2>
		<p class="muted">
			The cheapest baseline runs first and sets the bar; every candidate has to beat it. Defaults
			are sensible — change only what you need.
		</p>
	</div>

	<div class="card stack">
		<h3>Algorithms <span class="faint small">for {taskLabel(task)}</span></h3>
		<div class="modes" role="radiogroup" aria-label="Candidate algorithms">
			<label class="mode" class:on={effectiveMode === 'auto'}>
				<input
					type="radio"
					name="cmode"
					checked={effectiveMode === 'auto'}
					onchange={() => setMode('auto')}
				/>
				<span>
					<strong>Auto</strong><br /><span class="muted small">Baseline + recommended ladder</span>
				</span>
			</label>
			<label class="mode" class:on={effectiveMode === 'pick'}>
				<input
					type="radio"
					name="cmode"
					checked={effectiveMode === 'pick'}
					onchange={() => setMode('pick')}
				/>
				<span>
					<strong>Pick</strong><br /><span class="muted small">Choose the algorithms</span>
				</span>
			</label>
			<label class="mode" class:on={effectiveMode === 'all'}>
				<input
					type="radio"
					name="cmode"
					checked={effectiveMode === 'all'}
					onchange={() => setMode('all')}
				/>
				<span>
					<strong>All</strong><br /><span class="muted small">Every algorithm for this task</span>
				</span>
			</label>
		</div>

		{#if effectiveMode === 'pick'}
			{#if j.algorithms === null}
				<div class="row muted"><span class="spinner"></span> Loading the catalog…</div>
			{:else if algos.length === 0}
				<div class="alert warn">No algorithm in the catalog supports this task yet.</div>
			{:else}
				<div class="algos">
					{#each algos as a (a.key)}
						<label class="algo" class:on={picked.includes(a.key)}>
							<input
								type="checkbox"
								checked={picked.includes(a.key)}
								onchange={(e) => toggleAlgo(a.key, e.currentTarget.checked)}
							/>
							<span class="mono">{a.key}</span>
							<span class="backends">
								{#each a.backends as b (b)}<span class="badge">{b}</span>{/each}
							</span>
						</label>
					{/each}
				</div>
				{#if picked.length === 0}
					<p class="small muted">Nothing picked yet: the spec stays on <code>auto</code>.</p>
				{/if}
			{/if}
		{/if}

		<label class="toggle">
			<input
				type="checkbox"
				checked={baseline}
				onchange={(e) => j.set(['strategy', 'baseline'], e.currentTarget.checked ? null : false)}
			/>
			Run the trivial baseline first (recommended)
		</label>
	</div>

	<div class="card">
		<h3>Split</h3>
		<div class="grid3">
			<label class="field">
				Strategy
				<select
					value={splitStrategy}
					onchange={(e) =>
						j.set(
							['split', 'strategy'],
							e.currentTarget.value === 'auto' ? null : e.currentTarget.value
						)}
				>
					{#each SPLITS as [v, label] (v)}<option value={v}>{label}</option>{/each}
				</select>
			</label>
			<label class="field">
				Test size · <span class="num">{Math.round(testSize * 100)}%</span>
				<input
					type="range"
					min="0.05"
					max="0.5"
					step="0.05"
					value={testSize}
					onchange={(e) => j.set(['split', 'test_size'], Number(e.currentTarget.value))}
				/>
				<span class="hint">Held out and touched only once, for the final report.</span>
			</label>
			<label class="field">
				Seed
				<input
					type="number"
					min="0"
					step="1"
					value={seed}
					onchange={(e) =>
						j.set(['split', 'seed'], Math.max(0, Math.floor(Number(e.currentTarget.value) || 0)))}
				/>
				<span class="hint">Same seed + same data hash = same result.</span>
			</label>
		</div>
	</div>

	<div class="card">
		<h3>Budget</h3>
		<div class="grid3">
			<label class="field">
				Maximum time
				<select
					value={maxTime}
					onchange={(e) => j.set(['budget', 'max_time'], e.currentTarget.value || null)}
				>
					{#each TIMES as t (t)}<option value={t}>{t || 'No limit'}</option>{/each}
					{#if maxTime && !TIMES.includes(maxTime)}<option value={maxTime}>{maxTime}</option>{/if}
				</select>
			</label>
			<label class="field">
				Maximum cost (USD)
				<input
					type="number"
					min="0"
					step="0.5"
					placeholder="No limit"
					value={maxCost ?? ''}
					onchange={(e) => setCost(e.currentTarget.value)}
				/>
				<span class="hint">Local runs cost US$ 0.</span>
			</label>
		</div>
	</div>
</section>

<style>
	h3 {
		margin-bottom: 0.75rem;
	}
	.card.stack h3 {
		margin-bottom: 0;
	}
	.modes {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr));
		gap: 0.6rem;
	}
	.mode,
	.algo {
		display: flex;
		gap: 0.6rem;
		align-items: flex-start;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 0.65rem 0.75rem;
		cursor: pointer;
		background: var(--surface);
		transition:
			border-color 0.15s,
			background 0.15s;
	}
	.mode input,
	.algo input {
		margin-top: 0.2rem;
	}
	.mode.on,
	.algo.on {
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	.algos {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 15rem), 1fr));
		gap: 0.5rem;
	}
	.algo {
		align-items: center;
		padding: 0.45rem 0.65rem;
	}
	.algo input {
		margin-top: 0;
	}
	.algo .mono {
		flex: 1;
		overflow-wrap: anywhere;
	}
	.backends {
		display: flex;
		gap: 0.25rem;
	}
	.backends .badge {
		font-size: 0.68rem;
		padding: 0 0.4rem;
	}
	.grid3 {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 13rem), 1fr));
		gap: 1rem 1.25rem;
	}
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		cursor: pointer;
	}
</style>
