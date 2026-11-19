<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { Journey, STEPS, type StepKey } from '#lib/journey/journey.svelte.js';
	import StepData from '#lib/journey/StepData.svelte';
	import StepGoal from '#lib/journey/StepGoal.svelte';
	import StepAudit from '#lib/journey/StepAudit.svelte';
	import StepStrategy from '#lib/journey/StepStrategy.svelte';
	import StepMachine from '#lib/journey/StepMachine.svelte';
	import StepReview from '#lib/journey/StepReview.svelte';

	const j = new Journey();

	onMount(() => {
		j.loadCatalog();
	});

	const idx = $derived(STEPS.findIndex((s) => s.key === j.step));
	const next: StepKey | null = $derived(idx < STEPS.length - 1 ? STEPS[idx + 1].key : null);
	const prev: StepKey | null = $derived(idx > 0 ? STEPS[idx - 1].key : null);

	const nextBlockedReason = $derived.by(() => {
		switch (j.step) {
			case 'data':
				return j.profile ? null : 'Upload a dataset first.';
			case 'goal':
				if (j.target === undefined) return 'Choose what to predict.';
				return j.audit ? null : j.drafting || j.auditing ? 'Working…' : 'Waiting for the audit.';
			case 'audit':
				if (j.auditing) return 'Re-running the audit…';
				return j.blocked ? 'Resolve the red findings to continue.' : null;
			default:
				return null;
		}
	});

	function done(k: StepKey): boolean {
		const i = STEPS.findIndex((s) => s.key === k);
		return i < idx && j.reachable(k);
	}

	async function train() {
		const id = await j.train();
		if (id) await goto(resolve('/jobs/[id]', { id }));
	}

	const trainDisabled = $derived(
		j.submitting || j.blocked || j.auditing || (j.validation !== null && !j.validation.valid)
	);
</script>

<svelte:head><title>New job · têmpera</title></svelte:head>

<div class="page">
	<div class="title">
		<h1>New job</h1>
		<p class="muted">Seven steps from a file to an audited, reproducible model.</p>
	</div>

	<ol class="stepper" aria-label="Steps">
		{#each STEPS as s, i (s.key)}
			<li class:current={s.key === j.step} class:done={done(s.key)}>
				<button
					onclick={() => j.go(s.key)}
					disabled={!j.reachable(s.key)}
					aria-current={s.key === j.step ? 'step' : undefined}
				>
					<span class="n">
						{#if done(s.key)}
							<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
								<path
									d="M2.5 6.5 5 9l4.5-6"
									fill="none"
									stroke="currentColor"
									stroke-width="2"
									stroke-linecap="round"
								/>
							</svg>
						{:else}{i + 1}{/if}
					</span>
					<span class="lbl">
						<span class="name">{s.label}</span><span class="hint">{s.hint}</span>
					</span>
				</button>
			</li>
		{/each}
	</ol>

	<div class="body">
		{#if j.step === 'data'}
			<StepData {j} />
		{:else if j.step === 'goal'}
			<StepGoal {j} />
		{:else if j.step === 'audit'}
			<StepAudit {j} />
		{:else if j.step === 'strategy'}
			<StepStrategy {j} />
		{:else if j.step === 'machine'}
			<StepMachine {j} />
		{:else}
			<StepReview {j} />
		{/if}
	</div>

	<footer class="nav">
		{#if prev}
			<button class="btn" onclick={() => j.go(prev)}>← Back</button>
		{/if}
		<span class="spacer"></span>
		{#if j.step === 'review'}
			{#if j.submitError}<span class="err small" role="alert">{j.submitError}</span>{/if}
			<button class="btn primary train" onclick={train} disabled={trainDisabled}>
				{#if j.submitting}<span class="spinner"></span> Starting…{:else}
					<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
						<path d="M3 2v10l9-5z" fill="currentColor" />
					</svg> Train{/if}
			</button>
		{:else if next}
			{#if nextBlockedReason}<span class="muted small why">{nextBlockedReason}</span>{/if}
			<button class="btn primary" onclick={() => j.go(next)} disabled={nextBlockedReason !== null}>
				Next: {STEPS[idx + 1].label} →
			</button>
		{/if}
	</footer>
</div>

<style>
	.page {
		display: grid;
		gap: 1.25rem;
		min-width: 0;
	}
	.title h1 {
		margin-bottom: 0.15rem;
	}
	.title p {
		margin: 0;
	}

	.stepper {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(6, minmax(0, 1fr));
		gap: 0.4rem;
		counter-reset: step;
	}
	.stepper li {
		position: relative;
		min-width: 0;
	}
	.stepper button {
		width: 100%;
		display: flex;
		align-items: center;
		gap: 0.55rem;
		padding: 0.55rem 0.6rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--surface);
		color: var(--text);
		font: inherit;
		text-align: left;
		cursor: pointer;
		min-width: 0;
	}
	.stepper button:disabled {
		cursor: default;
		color: var(--faint);
		background: transparent;
	}
	.stepper li.current button {
		border-color: var(--accent);
		box-shadow: 0 0 0 3px var(--accent-soft);
	}
	.n {
		flex: none;
		width: 1.6rem;
		height: 1.6rem;
		border-radius: 50%;
		display: grid;
		place-items: center;
		font-size: 0.8rem;
		font-weight: 700;
		background: var(--surface-3);
		color: var(--muted);
	}
	li.current .n {
		background: var(--accent);
		color: var(--accent-contrast);
	}
	li.done .n {
		background: var(--ok-soft);
		color: var(--ok);
	}
	.lbl {
		display: grid;
		min-width: 0;
	}
	.name {
		font-weight: 600;
		font-size: 0.9rem;
	}
	.hint {
		font-size: 0.75rem;
		color: var(--muted);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.body {
		min-width: 0;
	}

	.nav {
		position: sticky;
		bottom: 0;
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.8rem 0;
		background: linear-gradient(to top, var(--bg) 70%, transparent);
		flex-wrap: wrap;
	}
	.err {
		color: var(--danger);
	}
	.train {
		min-width: 8rem;
	}

	@media (max-width: 900px) {
		.hint {
			display: none;
		}
	}
	@media (max-width: 640px) {
		.stepper {
			gap: 0.25rem;
		}
		.stepper button {
			flex-direction: column;
			gap: 0.2rem;
			padding: 0.4rem 0.1rem;
			text-align: center;
			border-color: transparent;
			background: transparent;
		}
		.stepper li.current button {
			box-shadow: none;
			border-color: transparent;
		}
		.name {
			font-size: 0.68rem;
		}
		.lbl {
			justify-items: center;
		}
		.why {
			width: 100%;
			order: -1;
			text-align: right;
		}
	}
</style>
