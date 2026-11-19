<script lang="ts">
	import type { Journey } from './journey.svelte';
	import SeverityDot from '#lib/components/SeverityDot.svelte';
	import Ticks from '#lib/components/Ticks.svelte';
	import { listAt, DROP_PATH, ACK_PATH } from '#lib/spec-yaml.js';
	import type { Finding, Severity } from '#lib/types.js';

	let { j }: { j: Journey } = $props();

	const ORDER: Record<Severity, number> = { red: 0, yellow: 1, green: 2 };

	const findings = $derived(
		[...(j.audit?.findings ?? [])].sort((a, b) => ORDER[a.severity] - ORDER[b.severity])
	);
	const counts = $derived({
		red: findings.filter((f) => f.severity === 'red').length,
		yellow: findings.filter((f) => f.severity === 'yellow').length,
		green: findings.filter((f) => f.severity === 'green').length
	});
	const unresolvedRed = $derived(
		findings.filter((f) => f.severity === 'red' && f.resolution.status === 'unresolved')
	);
	const dropped = $derived(listAt(j.yaml, DROP_PATH));
	const acknowledged = $derived(listAt(j.yaml, ACK_PATH));
	const plan = $derived(j.audit?.plan);
	const autoDropped = $derived(
		Object.entries(plan?.dropped ?? {}).filter(([c]) => !dropped.includes(c))
	);

	function resolutionText(f: Finding): string {
		switch (f.resolution.status) {
			case 'unresolved':
				return f.severity === 'green' ? 'ok' : 'needs a decision';
			case 'auto':
				return `auto: ${f.resolution.action}`;
			case 'acknowledged':
				return 'acknowledged';
		}
	}
</script>

<section class="stack">
	<div>
		<h2>3 · Audit</h2>
		<p class="muted">
			Têmpera checks the data for leakage, duplicates, imbalance and split pitfalls before any
			training. Every decision you make here is written to the spec.
		</p>
	</div>

	{#if j.auditError}
		<div class="alert danger" role="alert">
			<strong>Audit failed.</strong>
			{j.auditError}
			<button class="btn small" onclick={() => j.runAudit()}>Retry</button>
		</div>
	{/if}

	{#if j.audit}
		<div class="lights card">
			<div class="light"><SeverityDot severity="red" /> <b>{counts.red}</b> blocking</div>
			<div class="light"><SeverityDot severity="yellow" /> <b>{counts.yellow}</b> warnings</div>
			<div class="light"><SeverityDot severity="green" /> <b>{counts.green}</b> ok</div>
			<span class="spacer"></span>
			{#if j.auditing}
				<span class="row muted small"><span class="spinner"></span> Re-running the audit…</span>
			{:else}
				<span class="badge">policy: {j.audit.policy.replaceAll('_', ' ')}</span>
			{/if}
		</div>

		{#if j.blocked}
			<div class="alert danger" role="alert">
				<strong>Training is blocked.</strong>
				{unresolvedRed.length} red finding{unresolvedRed.length === 1 ? '' : 's'} need{unresolvedRed.length ===
				1
					? 's'
					: ''} a decision: drop the column, or acknowledge it if you are sure it is fine.
			</div>
		{:else if counts.red + counts.yellow === 0}
			<div class="alert ok">All checks passed. Nothing needs your attention.</div>
		{:else}
			<div class="alert ok">No blocking issues. You can continue.</div>
		{/if}

		<ul class="findings" aria-busy={j.auditing}>
			{#each findings as f (f.id)}
				{@const isDropped = f.column ? dropped.includes(f.column) : false}
				{@const isAck = acknowledged.includes(f.id)}
				<li class="finding card {f.severity}">
					<div class="fhead">
						<SeverityDot severity={f.severity} />
						<div class="ftitle">
							<strong>{f.title}</strong>
							<span class="meta faint small">
								<span class="mono">{f.id}</span>
								{#if f.column}· column <code>{f.column}</code>{/if}
							</span>
						</div>
						<span
							class="badge"
							class:danger={f.resolution.status === 'unresolved' && f.severity === 'red'}
							class:warn={f.resolution.status === 'unresolved' && f.severity === 'yellow'}
							class:ok={f.resolution.status !== 'unresolved' || f.severity === 'green'}
						>
							{resolutionText(f)}
						</span>
					</div>
					<p class="evidence"><Ticks text={f.evidence} /></p>
					<p class="suggestion">
						<span class="faint">Suggestion ·</span>
						<Ticks text={f.suggestion} />
					</p>
					{#if f.severity !== 'green' && (f.resolution.status !== 'auto' || isDropped)}
						<div class="actions">
							{#if f.column}
								<label class="toggle">
									<input
										type="checkbox"
										checked={isDropped}
										disabled={j.auditing}
										onchange={(e) => j.setDrop(f.column!, e.currentTarget.checked)}
									/>
									Drop column <code>{f.column}</code>
								</label>
							{/if}
							<label class="toggle">
								<input
									type="checkbox"
									checked={isAck}
									disabled={j.auditing}
									onchange={(e) => j.setAcknowledged(f.id, e.currentTarget.checked)}
								/>
								Acknowledge (I reviewed it, keep as is)
							</label>
						</div>
					{/if}
				</li>
			{/each}
		</ul>

		{#if dropped.length > 0 || autoDropped.length > 0}
			<div class="card">
				<h3>Columns not used</h3>
				<ul class="dropped">
					{#each dropped as c (c)}
						<li>
							<code>{c}</code> <span class="muted small">dropped by you</span>
							<button
								class="btn small ghost"
								disabled={j.auditing}
								onclick={() => j.setDrop(c, false)}
							>
								Undo
							</button>
						</li>
					{/each}
					{#each autoDropped as [c, why] (c)}
						<li><code>{c}</code> <span class="muted small">{why}</span></li>
					{/each}
				</ul>
				{#if plan}
					<p class="small muted">
						{plan.features.length} feature{plan.features.length === 1 ? '' : 's'} will be used{plan
							.datetime.length
							? `, plus ${plan.datetime.length} date column(s) expanded into year/month/day-of-week/hour`
							: ''}{plan.dedupe ? '; exact duplicate rows are removed before the split' : ''}.
					</p>
				{/if}
			</div>
		{/if}
	{:else if j.auditing}
		<div class="card row"><span class="spinner"></span> Auditing the data…</div>
	{/if}
</section>

<style>
	.lights {
		display: flex;
		gap: 0.6rem 1.5rem;
		align-items: center;
		flex-wrap: wrap;
		padding: 0.8rem 1.1rem;
	}
	.light {
		display: inline-flex;
		align-items: center;
		gap: 0.55rem;
	}
	.findings {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 0.75rem;
		transition: opacity 0.15s;
	}
	.findings[aria-busy='true'] {
		opacity: 0.65;
	}
	.finding {
		border-left: 4px solid var(--ok);
	}
	.finding.red {
		border-left-color: var(--danger);
	}
	.finding.yellow {
		border-left-color: var(--warn);
	}
	.fhead {
		display: flex;
		gap: 0.75rem;
		align-items: flex-start;
	}
	.fhead :global(.light),
	.fhead > :global(span:first-child) {
		margin-top: 0.35rem;
	}
	.ftitle {
		display: grid;
		flex: 1;
		min-width: 0;
	}
	.meta {
		overflow-wrap: anywhere;
	}
	.evidence {
		margin: 0.6rem 0 0.3rem;
	}
	.suggestion {
		margin: 0;
		font-size: 0.92rem;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem 1.5rem;
		margin-top: 0.75rem;
		padding-top: 0.7rem;
		border-top: 1px dashed var(--border);
	}
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 0.45rem;
		cursor: pointer;
		font-size: 0.92rem;
	}
	.dropped {
		list-style: none;
		padding: 0;
		margin: 0 0 0.5rem;
		display: grid;
		gap: 0.3rem;
	}
	.dropped li {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	@media (max-width: 560px) {
		.fhead {
			flex-wrap: wrap;
		}
		.ftitle {
			flex-basis: calc(100% - 2rem);
		}
		.fhead .badge {
			order: 3;
			margin-left: 1.6rem;
		}
	}
</style>
