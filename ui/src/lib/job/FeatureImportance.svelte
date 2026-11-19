<script lang="ts">
	import type { Importance } from '#lib/events.js';
	import { fmtNum } from '#lib/format.js';

	let { items, limit = 12 }: { items: Importance[]; limit?: number } = $props();

	let all = $state(false);
	const shown = $derived(all ? items : items.slice(0, limit));
	const max = $derived(Math.max(1e-12, ...items.map((i) => Math.abs(i.value))));
</script>

<div class="fi">
	<div class="head">
		<h3>Feature importance</h3>
		{#if items.length > limit}
			<button class="btn small ghost" onclick={() => (all = !all)}>
				{all ? 'Top ' + limit : `All ${items.length}`}
			</button>
		{/if}
	</div>
	<ol>
		{#each shown as it (it.feature)}
			<li>
				<span class="name mono" title={it.feature}>{it.feature}</span>
				<span class="bar" aria-hidden="true">
					<span class:neg={it.value < 0} style:width="{(Math.abs(it.value) / max) * 100}%"></span>
				</span>
				<span class="val num small">{fmtNum(it.value, 3)}</span>
			</li>
		{/each}
	</ol>
</div>

<style>
	.fi {
		border-top: 1px solid var(--border);
		padding-top: 0.75rem;
		margin-top: 0.25rem;
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}
	h3 {
		margin: 0;
		font-size: 0.95rem;
	}
	ol {
		list-style: none;
		margin: 0.5rem 0 0;
		padding: 0;
		display: grid;
		gap: 0.3rem;
	}
	li {
		display: grid;
		grid-template-columns: minmax(5rem, 11rem) 1fr 3.5rem;
		align-items: center;
		gap: 0.6rem;
		font-size: 0.85rem;
	}
	.name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.bar {
		height: 8px;
		background: var(--surface-3);
		border-radius: 4px;
		overflow: hidden;
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--series-2);
		border-radius: 4px;
	}
	.bar span.neg {
		background: var(--series-1);
	}
	.val {
		text-align: right;
		color: var(--muted);
	}
</style>
