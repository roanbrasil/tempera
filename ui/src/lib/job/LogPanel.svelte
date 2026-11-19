<script lang="ts">
	import { tick } from 'svelte';
	import type { LogLine, RunState } from '#lib/events.js';

	let {
		logs,
		runs,
		dropped = 0
	}: { logs: LogLine[]; runs: RunState[]; dropped?: number } = $props();

	const SHOW = 800;

	let level: 'all' | 'warn' | 'error' = $state('all');
	let run: string = $state('');
	/** Unparsed backend stderr (`raw`) is noisy: hidden unless asked for. */
	let raw = $state(false);
	let expanded: Record<number, boolean> = $state({});
	let box: HTMLDivElement;
	let stick = $state(true);

	const names = $derived(
		Object.fromEntries(runs.map((r) => [r.run_id, `${r.index}.${r.algorithm}`]))
	);
	const filtered = $derived(
		logs.filter(
			(l) =>
				(level === 'all' ||
					(level === 'warn' && (l.level === 'warn' || l.level === 'error')) ||
					(level === 'error' && l.level === 'error')) &&
				(raw || l.level !== 'raw') &&
				(!run || l.run_id === run)
		)
	);
	const shown = $derived(filtered.length > SHOW ? filtered.slice(-SHOW) : filtered);
	const counts = $derived({
		warn: logs.filter((l) => l.level === 'warn').length,
		error: logs.filter((l) => l.level === 'error').length,
		raw: logs.filter((l) => l.level === 'raw').length
	});

	$effect(() => {
		void shown;
		if (!stick || !box) return;
		tick().then(() => {
			if (box) box.scrollTop = box.scrollHeight;
		});
	});

	function onScroll() {
		stick = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
	}

	/** Long lines (paths, tracebacks) are clamped to a few rows; click to expand. */
	function isLong(m: string): boolean {
		return m.length > 180 || m.split('\n').length > 3;
	}

	function toggle(seq: number) {
		expanded = { ...expanded, [seq]: !expanded[seq] };
	}

	function jump() {
		stick = true;
		box.scrollTop = box.scrollHeight;
	}
</script>

<div class="logs">
	<div class="bar">
		<div class="seg" role="radiogroup" aria-label="Log level">
			<button role="radio" aria-checked={level === 'all'} onclick={() => (level = 'all')}>
				All
			</button>
			<button role="radio" aria-checked={level === 'warn'} onclick={() => (level = 'warn')}>
				Warnings {#if counts.warn + counts.error > 0}<span class="n warn">
						{counts.warn + counts.error}
					</span>{/if}
			</button>
			<button role="radio" aria-checked={level === 'error'} onclick={() => (level = 'error')}>
				Errors {#if counts.error > 0}<span class="n err">{counts.error}</span>{/if}
			</button>
		</div>
		<select aria-label="Filter by run" bind:value={run}>
			<option value="">All runs</option>
			{#each runs as r (r.run_id)}<option value={r.run_id}>{r.index}. {r.algorithm}</option>{/each}
		</select>
		{#if counts.raw > 0}
			<label class="rawtoggle small">
				<input type="checkbox" bind:checked={raw} />
				Raw output <span class="faint">({counts.raw})</span>
			</label>
		{/if}
		<span class="spacer"></span>
		{#if !stick}<button class="btn small" onclick={jump}>↓ Follow</button>{/if}
	</div>
	<div class="box mono" bind:this={box} onscroll={onScroll} role="log" aria-live="off">
		{#if dropped > 0 || filtered.length > shown.length}
			<div class="more faint">… earlier lines hidden</div>
		{/if}
		{#each shown as l (l.seq)}
			<div class="line {l.level}">
				<span class="lvl">{l.level}</span>
				{#if l.run_id}<span class="src">{names[l.run_id] ?? l.run_id.slice(0, 8)}</span>{/if}
				{#if isLong(l.message)}
					<button
						class="msg clamp"
						class:open={expanded[l.seq]}
						title={expanded[l.seq] ? 'Collapse' : 'Show the whole line'}
						onclick={() => toggle(l.seq)}
					>
						{l.message}
					</button>
				{:else}
					<span class="msg">{l.message}</span>
				{/if}
			</div>
		{:else}
			<div class="faint empty">
				No log lines{level !== 'all' || run ? ' match the filter' : ' yet'}.
			</div>
		{/each}
	</div>
</div>

<style>
	.logs {
		display: grid;
		gap: 0.6rem;
		min-width: 0;
	}
	.bar {
		display: flex;
		align-items: center;
		gap: 0.6rem;
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
		font-size: 0.82rem;
		padding: 0.25rem 0.65rem;
		border: none;
		background: var(--surface);
		color: var(--muted);
		cursor: pointer;
		display: inline-flex;
		gap: 0.3rem;
		align-items: center;
	}
	.seg button + button {
		border-left: 1px solid var(--border-strong);
	}
	.seg button[aria-checked='true'] {
		background: var(--accent-soft);
		color: var(--text);
		font-weight: 600;
	}
	.n {
		font-size: 0.7rem;
		border-radius: 999px;
		padding: 0 0.4rem;
		font-weight: 700;
	}
	.n.warn {
		background: var(--warn-soft);
		color: var(--warn);
	}
	.n.err {
		background: var(--danger-soft);
		color: var(--danger);
	}
	select {
		min-height: 30px;
		padding: 0.15rem 0.5rem;
		font-size: 0.82rem;
	}
	.box {
		height: 300px;
		overflow: auto;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 0.5rem 0;
		font-size: 0.78rem;
		line-height: 1.55;
	}
	.line {
		display: flex;
		gap: 0.6rem;
		padding: 0 0.75rem;
		min-width: 0;
	}
	.line.warn {
		background: color-mix(in srgb, var(--warn) 12%, transparent);
	}
	.line.error {
		background: color-mix(in srgb, var(--danger) 14%, transparent);
	}
	.lvl {
		flex: none;
		width: 2.8rem;
		text-transform: uppercase;
		font-size: 0.68rem;
		font-weight: 700;
		color: var(--faint);
		padding-top: 0.1rem;
	}
	.warn .lvl {
		color: var(--warn);
	}
	.error .lvl {
		color: var(--danger);
	}
	.src {
		flex: none;
		color: var(--link);
		max-width: 9rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.msg {
		flex: 1;
		min-width: 0;
		white-space: pre-wrap;
		overflow-wrap: break-word;
		word-break: break-word;
	}
	button.msg {
		all: unset;
		flex: 1;
		min-width: 0;
		cursor: zoom-in;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	button.msg:focus-visible {
		outline: 2px solid var(--focus);
	}
	.clamp:not(.open) {
		display: -webkit-box;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		overflow: hidden;
	}
	.clamp.open {
		cursor: zoom-out;
	}
	.line.raw {
		color: var(--muted);
	}
	.rawtoggle {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		cursor: pointer;
		color: var(--muted);
	}
	.more,
	.empty {
		padding: 0 0.75rem;
	}
	@media (max-width: 560px) {
		.line {
			flex-wrap: wrap;
			gap: 0 0.5rem;
			padding-bottom: 0.2rem;
		}
		.msg {
			flex-basis: 100%;
		}
		.src {
			max-width: 10rem;
		}
		.lvl {
			width: 2.3rem;
		}
	}
</style>
