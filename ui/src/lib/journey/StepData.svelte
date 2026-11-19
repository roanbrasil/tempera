<script lang="ts">
	import type { Journey } from './journey.svelte';
	import ProfileTable from './ProfileTable.svelte';
	import { fmtInt, shortId } from '#lib/format.js';

	let { j }: { j: Journey } = $props();

	let over = $state(false);
	let input: HTMLInputElement;

	const ACCEPT = '.csv,.tsv,.parquet,.jsonl,.ndjson,.txt';

	function pick(files: FileList | null | undefined) {
		const f = files?.[0];
		if (f) j.upload(f);
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		over = false;
		pick(e.dataTransfer?.files);
	}

	const kinds = $derived.by(() => {
		const out: Record<string, number> = {};
		for (const c of j.profile?.columns ?? []) out[c.kind] = (out[c.kind] ?? 0) + 1;
		return Object.entries(out).sort((a, b) => b[1] - a[1]);
	});
</script>

<section class="stack">
	<div>
		<h2>1 · Your data</h2>
		<p class="muted">
			CSV, TSV, Parquet or JSON Lines. Têmpera converts it to Parquet, versions it by content hash
			and profiles every column.
		</p>
	</div>

	<div
		class="drop"
		class:over
		class:compact={j.profile !== null}
		role="button"
		tabindex="0"
		aria-label="Upload a dataset: drop a file here or press Enter to browse"
		ondragenter={(e) => {
			e.preventDefault();
			over = true;
		}}
		ondragover={(e) => {
			e.preventDefault();
			over = true;
		}}
		ondragleave={() => (over = false)}
		ondrop={onDrop}
		onclick={() => input.click()}
		onkeydown={(e) => {
			if (e.key === 'Enter' || e.key === ' ') {
				e.preventDefault();
				input.click();
			}
		}}
	>
		<input
			bind:this={input}
			type="file"
			accept={ACCEPT}
			class="sr-only"
			tabindex="-1"
			onchange={(e) => pick(e.currentTarget.files)}
		/>
		{#if j.uploading}
			<span class="spinner"></span>
			<div><strong>Uploading and profiling…</strong></div>
		{:else}
			<svg width="34" height="34" viewBox="0 0 24 24" fill="none" aria-hidden="true">
				<path
					d="M12 16V4m0 0-4 4m4-4 4 4M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"
					stroke="currentColor"
					stroke-width="1.8"
					stroke-linecap="round"
					stroke-linejoin="round"
				/>
			</svg>
			<div>
				<strong>{j.profile ? 'Replace the dataset' : 'Drop a file here'}</strong>
				<span class="muted"> or <span class="link">browse</span></span>
			</div>
			<div class="faint small">.csv · .tsv · .parquet · .jsonl</div>
		{/if}
	</div>

	{#if j.uploadError}
		<div class="alert danger" role="alert"><strong>Upload failed.</strong> {j.uploadError}</div>
	{/if}

	{#if j.dataset && j.profile}
		<div class="card summary">
			<div class="kv">
				<span class="k">File</span>
				<span class="v mono" title={j.dataset.source}>{j.dataset.source.split(/[\\/]/).pop()}</span>
			</div>
			<div class="kv">
				<span class="k">Rows</span><span class="v num">{fmtInt(j.profile.rows)}</span>
			</div>
			<div class="kv">
				<span class="k">Columns</span><span class="v num">{j.profile.columns.length}</span>
			</div>
			<div class="kv">
				<span class="k">Format</span><span class="v">{j.dataset.format.toUpperCase()}</span>
			</div>
			<div class="kv">
				<span class="k">Version</span>
				<span class="v mono" title={j.dataset.id}>{shortId(j.dataset.id, 12)}</span>
			</div>
			<div class="kinds">
				{#each kinds as [k, n] (k)}
					<span class="badge">{n} {k.replace('_', ' ')}</span>
				{/each}
			</div>
		</div>
		{#if j.profile.sampled_rows}
			<p class="small muted">
				Quantiles, histograms and top values come from a uniform sample of {fmtInt(
					j.profile.sampled_rows
				)} rows; counts are exact.
			</p>
		{/if}
		<ProfileTable profile={j.profile} />
	{/if}
</section>

<style>
	.drop {
		border: 2px dashed var(--border-strong);
		border-radius: var(--radius);
		background: var(--surface);
		padding: 2.5rem 1rem;
		display: grid;
		justify-items: center;
		gap: 0.4rem;
		text-align: center;
		cursor: pointer;
		color: var(--muted);
		transition:
			border-color 0.15s,
			background 0.15s;
	}
	.drop.compact {
		padding: 1.1rem 1rem;
		grid-auto-flow: column;
		justify-content: center;
		align-items: center;
		gap: 0.75rem;
	}
	.drop.compact .faint {
		display: none;
	}
	.drop:hover,
	.drop.over {
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	.drop strong {
		color: var(--text);
	}
	.link {
		color: var(--link);
		text-decoration: underline;
	}
	.summary {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75rem 2rem;
		align-items: center;
	}
	.kv {
		display: grid;
		min-width: 0;
	}
	.k {
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--faint);
	}
	.v {
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 16rem;
	}
	.kinds {
		display: flex;
		gap: 0.35rem;
		flex-wrap: wrap;
		margin-left: auto;
	}
	@media (max-width: 640px) {
		.kinds {
			margin-left: 0;
		}
		.summary {
			gap: 0.6rem 1.25rem;
		}
	}
</style>
