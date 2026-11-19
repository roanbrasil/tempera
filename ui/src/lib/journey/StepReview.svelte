<script lang="ts">
	import type { Journey } from './journey.svelte';
	import { syntaxError } from '#lib/spec-yaml.js';

	let { j }: { j: Journey } = $props();

	let gutter: HTMLDivElement;

	const lines = $derived(j.yaml.split('\n').length);
	const syntax = $derived(syntaxError(j.yaml));

	// Validate on the server shortly after the user stops typing.
	$effect(() => {
		void j.yaml;
		if (syntax) return;
		const t = setTimeout(() => j.validate(), 350);
		return () => clearTimeout(t);
	});

	function onKey(e: KeyboardEvent & { currentTarget: HTMLTextAreaElement }) {
		if (e.key === 'Tab' && !e.shiftKey) {
			// YAML forbids tabs: insert two spaces.
			e.preventDefault();
			const el = e.currentTarget;
			const { selectionStart: s, selectionEnd: end } = el;
			j.yaml = j.yaml.slice(0, s) + '  ' + j.yaml.slice(end);
			requestAnimationFrame(() => el.setSelectionRange(s + 2, s + 2));
		}
	}

	function syncScroll(e: Event & { currentTarget: HTMLTextAreaElement }) {
		if (gutter) gutter.scrollTop = e.currentTarget.scrollTop;
	}
</script>

<section class="stack">
	<div>
		<h2>6 · Review the spec</h2>
		<p class="muted">
			This is the complete <code>tempera.yaml</code> for the job — commit it to git to reproduce the run.
			You can edit it directly.
		</p>
	</div>

	<div class="card editor-card">
		<div class="bar">
			<span class="mono small">tempera.yaml</span>
			<span class="spacer"></span>
			{#if syntax}
				<span class="badge danger">syntax error</span>
			{:else if j.validating}
				<span class="badge"><span class="spinner"></span> validating</span>
			{:else if j.validation?.valid}
				<span class="badge ok"><span class="dot"></span>valid</span>
			{:else if j.validation}
				<span class="badge danger">
					{j.validation.issues.length} issue{j.validation.issues.length === 1 ? '' : 's'}
				</span>
			{/if}
		</div>
		<div class="editor">
			<div class="gutter mono" bind:this={gutter} aria-hidden="true">
				{#each { length: lines }, i (i)}<div>{i + 1}</div>{/each}
			</div>
			<textarea
				class="mono"
				bind:value={j.yaml}
				spellcheck="false"
				autocapitalize="off"
				autocomplete="off"
				wrap="off"
				aria-label="Job spec (YAML)"
				onkeydown={onKey}
				onscroll={syncScroll}
				rows={Math.min(32, Math.max(14, lines + 1))}></textarea>
		</div>
	</div>

	{#if syntax}
		<div class="alert danger" role="alert"><strong>YAML syntax:</strong> {syntax}</div>
	{:else if j.validation && !j.validation.valid}
		<div class="alert danger" role="alert">
			<strong>The spec is not valid.</strong>
			<ul class="issues">
				{#each j.validation.issues as is, i (i)}
					<li>
						{#if is.path}<code>{is.path}</code>
						{/if}{is.message}
					</li>
				{/each}
			</ul>
		</div>
	{/if}

	{#if j.auditStale && !syntax}
		<div class="alert info">
			The spec changed since the last audit; it is audited again when you press <strong>
				Train
			</strong>.
		</div>
	{/if}
</section>

<style>
	.editor-card {
		padding: 0;
		overflow: hidden;
	}
	.bar {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.9rem;
		border-bottom: 1px solid var(--border);
		background: var(--surface-2);
	}
	.bar .spinner {
		width: 0.7rem;
		height: 0.7rem;
	}
	.editor {
		display: flex;
		min-width: 0;
		background: var(--surface);
	}
	.gutter {
		flex: none;
		width: 2.6rem;
		padding: 0.75rem 0.5rem 0.75rem 0;
		text-align: right;
		color: var(--faint);
		background: var(--surface-2);
		border-right: 1px solid var(--border);
		overflow: hidden;
		user-select: none;
		font-size: 0.82rem;
		line-height: 1.55;
	}
	textarea {
		flex: 1;
		min-width: 0;
		border: none;
		border-radius: 0;
		padding: 0.75rem 0.9rem;
		resize: vertical;
		font-size: 0.82rem;
		line-height: 1.55;
		background: transparent;
		tab-size: 2;
		white-space: pre;
		overflow: auto;
	}
	textarea:focus {
		outline: none;
		box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--focus) 50%, transparent);
	}
	.issues {
		margin: 0.4rem 0 0;
		padding-left: 1.2rem;
	}
</style>
