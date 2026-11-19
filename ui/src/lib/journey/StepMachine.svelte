<script lang="ts">
	import type { Journey } from './journey.svelte';
	import { listAt } from '#lib/spec-yaml.js';
	import { fmtBytes } from '#lib/format.js';
	import type { Machine } from '#lib/types.js';

	let { j }: { j: Journey } = $props();

	const allow = $derived(listAt(j.yaml, ['compute', 'allow']));
	const selected = $derived(allow[0] ?? j.machineId);

	/** Rough in-memory size of the dataset as Arrow (8 bytes/cell × 3 for training copies). */
	const needBytes = $derived(
		j.profile ? j.profile.rows * Math.max(1, j.profile.columns.length) * 8 * 3 : 0
	);

	function fits(m: Machine): boolean {
		return needBytes < m.hardware.ram_bytes * 0.5;
	}

	function choose(id: string) {
		j.machineId = id;
		j.set(['compute', 'allow'], [id]);
	}

	function kindLabel(kind: string): string {
		return kind === 'local' ? 'This computer' : kind === 'ssh' ? 'LAN / SSH' : kind;
	}
</script>

<section class="stack">
	<div>
		<h2>5 · Machine</h2>
		<p class="muted">
			Where the job runs. Remote machines and cloud GPUs arrive in a later milestone; for now jobs
			run on the machine serving Têmpera.
		</p>
	</div>

	{#if j.machines === null}
		<div class="card row muted"><span class="spinner"></span> Detecting hardware…</div>
	{:else if j.machines.length === 0}
		<div class="alert warn">No machine available.</div>
	{:else}
		<div class="machines" role="radiogroup" aria-label="Machine">
			{#each j.machines as m (m.id)}
				<label class="machine card" class:on={selected === m.id}>
					<div class="mh">
						<input
							type="radio"
							name="machine"
							checked={selected === m.id}
							onchange={() => choose(m.id)}
						/>
						<div class="mt">
							<strong>{kindLabel(m.kind)}</strong>
							<span class="faint mono small">{m.id}</span>
						</div>
						<span class="badge" class:ok={m.status === 'ready'}>
							<span class="dot"></span>{m.status}
						</span>
					</div>
					<dl>
						<dt>System</dt>
						<dd>{m.hardware.os} · {m.hardware.arch}</dd>
						<dt>CPU</dt>
						<dd>{m.hardware.cpu_cores} cores</dd>
						<dt>Memory</dt>
						<dd>{fmtBytes(m.hardware.ram_bytes)}</dd>
						<dt>GPU</dt>
						<dd>
							{#if m.hardware.gpus.length === 0}
								<span class="muted">none</span>
							{:else}
								{#each m.hardware.gpus as g, i (i)}
									<div>{g.name} <span class="muted">· {fmtBytes(g.vram_bytes)}</span></div>
								{/each}
							{/if}
						</dd>
						<dt>Fits in memory</dt>
						<dd>
							{#if fits(m)}<span class="badge ok">yes</span>{:else}<span class="badge warn">
									tight
								</span>{/if}
							<span class="muted small">≈ {fmtBytes(needBytes)} needed</span>
						</dd>
						<dt>Cost</dt>
						<dd>{m.kind === 'local' ? 'US$ 0' : '—'}</dd>
					</dl>
				</label>
			{/each}
		</div>
	{/if}
</section>

<style>
	.machines {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 20rem), 1fr));
		gap: 1rem;
	}
	.machine {
		cursor: pointer;
		border-width: 1px;
		transition: border-color 0.15s;
	}
	.machine.on {
		border-color: var(--accent);
		box-shadow:
			0 0 0 3px var(--accent-soft),
			var(--shadow);
	}
	.mh {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		margin-bottom: 0.75rem;
	}
	.mt {
		display: grid;
		flex: 1;
		min-width: 0;
	}
	dl {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 0.35rem 1rem;
		margin: 0;
		font-size: 0.92rem;
	}
	dt {
		color: var(--muted);
	}
	dd {
		margin: 0;
		min-width: 0;
		overflow-wrap: anywhere;
	}
</style>
