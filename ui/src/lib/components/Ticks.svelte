<script lang="ts">
	// Renders server text where `backticks` mark identifiers (column names, YAML keys) as code.
	let { text }: { text: string } = $props();
	const parts = $derived(text.split('`'));
</script>

{#each parts as p, i (i)}{#if i % 2 === 1}<code>{p}</code>{:else}{p}{/if}{/each}
