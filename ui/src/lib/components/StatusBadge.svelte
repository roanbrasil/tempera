<script lang="ts">
	let { status }: { status: string | null | undefined } = $props();

	const tone = $derived.by(() => {
		switch (status) {
			case 'succeeded':
				return 'ok';
			case 'failed':
			case 'blocked':
				return 'danger';
			case 'cancelled':
			case 'skipped':
				return 'neutral';
			case 'running':
				return 'info';
			case 'created':
			case 'queued':
				return 'accent';
			default:
				return 'neutral';
		}
	});
	const live = $derived(status === 'running' || status === 'created' || status === 'queued');
</script>

<span class="badge {tone}"><span class="dot" class:pulse={live}></span>{status ?? 'unknown'}</span>
