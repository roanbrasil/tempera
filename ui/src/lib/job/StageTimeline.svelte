<script lang="ts">
	import { STAGES, type StageName } from '#lib/events.js';

	let {
		current,
		messages,
		finished
	}: {
		current: StageName | null;
		messages: Partial<Record<StageName, string>>;
		finished: string | null;
	} = $props();

	const LABELS: Record<StageName, string> = {
		ingest: 'Ingest',
		profile: 'Profile',
		audit: 'Audit',
		split: 'Split',
		train: 'Train',
		evaluate: 'Evaluate',
		report: 'Report'
	};

	const curIdx = $derived(current ? STAGES.indexOf(current) : -1);

	function state(i: number): 'done' | 'active' | 'failed' | 'todo' {
		if (finished === 'succeeded') return i <= curIdx || curIdx === -1 ? 'done' : 'todo';
		if (i < curIdx) return 'done';
		if (i === curIdx) return finished ? 'failed' : 'active';
		return 'todo';
	}
</script>

<ol class="timeline" aria-label="Pipeline stages">
	{#each STAGES as s, i (s)}
		{@const st = state(i)}
		<li class={st} title={messages[s] ?? ''}>
			<span class="node" aria-hidden="true">
				{#if st === 'done'}
					<svg width="10" height="10" viewBox="0 0 12 12">
						<path
							d="M2.5 6.5 5 9l4.5-6"
							fill="none"
							stroke="currentColor"
							stroke-width="2.2"
							stroke-linecap="round"
						/>
					</svg>
				{:else if st === 'failed'}
					<svg width="10" height="10" viewBox="0 0 12 12">
						<path
							d="M3 3l6 6M9 3 3 9"
							stroke="currentColor"
							stroke-width="2.2"
							stroke-linecap="round"
						/>
					</svg>
				{/if}
			</span>
			<span class="label">{LABELS[s]}</span>
			<span class="sr-only">{st}</span>
		</li>
	{/each}
</ol>
{#if current && messages[current]}
	<p class="msg small muted">{messages[current]}</p>
{/if}

<style>
	.timeline {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(7, minmax(0, 1fr));
	}
	li {
		position: relative;
		display: grid;
		justify-items: center;
		gap: 0.35rem;
		min-width: 0;
	}
	/* connector line to the previous node */
	li + li::before {
		content: '';
		position: absolute;
		top: 9px;
		right: 50%;
		width: 100%;
		height: 2px;
		background: var(--surface-3);
		z-index: 0;
	}
	li.done + li::before,
	li.done + li.active::before {
		background: var(--accent);
	}
	.node {
		position: relative;
		z-index: 1;
		width: 20px;
		height: 20px;
		border-radius: 50%;
		display: grid;
		place-items: center;
		background: var(--surface);
		border: 2px solid var(--border-strong);
		color: var(--accent-contrast);
	}
	li.done .node {
		background: var(--accent);
		border-color: var(--accent);
	}
	li.active .node {
		border-color: var(--accent);
		box-shadow: 0 0 0 4px var(--accent-soft);
	}
	li.active .node::after {
		content: '';
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--accent);
		animation: pulse 1.4s ease-in-out infinite;
	}
	li.failed .node {
		background: var(--danger);
		border-color: var(--danger);
		color: #fff;
	}
	.label {
		font-size: 0.8rem;
		color: var(--muted);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 100%;
	}
	li.active .label,
	li.failed .label {
		color: var(--text);
		font-weight: 600;
	}
	.msg {
		margin: 0.75rem 0 0;
		text-align: center;
		overflow-wrap: anywhere;
	}
	@media (max-width: 520px) {
		.label {
			font-size: 0.66rem;
		}
	}
</style>
