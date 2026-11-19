<script lang="ts">
	import type { Severity } from '#lib/types.js';

	let { severity, size = 12 }: { severity: Severity; size?: number } = $props();
	const label = $derived(
		severity === 'red' ? 'Blocking' : severity === 'yellow' ? 'Warning' : 'OK'
	);
</script>

<span
	class="light {severity}"
	style:width="{size}px"
	style:height="{size}px"
	role="img"
	aria-label={label}
	title={label}
></span>

<style>
	.light {
		display: inline-block;
		flex: none;
		border-radius: 50%;
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 12%);
	}
	.red {
		background: var(--danger);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--danger) 22%, transparent);
	}
	.yellow {
		background: var(--warn);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--warn) 22%, transparent);
	}
	.green {
		background: var(--ok);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent);
	}
</style>
