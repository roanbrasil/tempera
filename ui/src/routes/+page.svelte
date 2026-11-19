<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import { api, ApiError } from '#lib/api.js';
	import type { JobRow } from '#lib/types.js';
	import StatusBadge from '#lib/components/StatusBadge.svelte';
	import { fmtDateTime, fmtRelative, metricLabel, taskLabel } from '#lib/format.js';

	let jobs = $state<JobRow[] | null>(null);
	let error = $state<string | null>(null);
	let now = $state(Date.now());

	async function load() {
		try {
			jobs = await api.jobs();
			error = null;
		} catch (e) {
			error = e instanceof ApiError ? e.message : String(e);
		}
		now = Date.now();
	}

	const active = $derived(
		(jobs ?? []).filter((j) => j.status === 'running' || j.status === 'created').length
	);

	onMount(() => {
		load();
		// Refresh faster while something is running.
		let t: ReturnType<typeof setTimeout>;
		const tick = async () => {
			await load();
			t = setTimeout(tick, active > 0 ? 3000 : 10_000);
		};
		t = setTimeout(tick, 3000);
		return () => clearTimeout(t);
	});
</script>

<svelte:head><title>Jobs · têmpera</title></svelte:head>

<div class="head">
	<div>
		<h1>Jobs</h1>
		<p class="muted">
			Every training job, newest first.
			{#if active > 0}<span class="badge info">
					<span class="dot pulse"></span>{active} running
				</span>{/if}
		</p>
	</div>
	<a class="btn primary" href={resolve('/new')}>
		<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
			<path d="M8 3v10M3 8h10" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
		</svg>
		New job
	</a>
</div>

{#if error}
	<div class="alert danger" role="alert">
		<strong>Could not load jobs.</strong>
		{error}
		<button class="btn small" onclick={load}>Retry</button>
	</div>
{/if}

{#if jobs === null && !error}
	<div class="card skeleton" aria-busy="true">
		{#each [0, 1, 2] as i (i)}<div class="sk-row"></div>{/each}
	</div>
{:else if jobs && jobs.length === 0}
	<div class="card empty">
		<img src="/mark.png" alt="" width="72" height="72" />
		<h2>No jobs yet</h2>
		<p class="muted">
			Upload a dataset, pick what to predict, and Têmpera audits the data, trains a baseline and
			candidates, and reports the winner.
		</p>
		<a class="btn primary" href={resolve('/new')}>Start your first job</a>
	</div>
{:else if jobs}
	<div class="card list">
		<table class="data jobs">
			<thead>
				<tr>
					<th>Name</th>
					<th>Status</th>
					<th>Task</th>
					<th>Metric</th>
					<th>Created</th>
				</tr>
			</thead>
			<tbody>
				{#each jobs as job (job.id)}
					<tr>
						<td class="name">
							<a href={resolve('/jobs/[id]', { id: job.id })}>{job.name}</a>
							<div class="faint mono id">{job.id}</div>
							{#if job.error}<div class="err small">{job.error}</div>{/if}
						</td>
						<td data-label="Status"><StatusBadge status={job.status} /></td>
						<td data-label="Task">{taskLabel(job.task)}</td>
						<td data-label="Metric">{metricLabel(job.metric)}</td>
						<td data-label="Created" title={fmtDateTime(job.created_at)}>
							{fmtRelative(job.created_at, now)}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}

<style>
	.head {
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
		gap: 1rem;
		margin-bottom: 1.25rem;
		flex-wrap: wrap;
	}
	.head p {
		margin: 0;
		display: flex;
		gap: 0.6rem;
		align-items: center;
		flex-wrap: wrap;
	}
	.list {
		padding: 0.25rem 0.5rem;
	}
	.name a {
		font-weight: 600;
		font-size: 0.98rem;
	}
	.id {
		font-size: 0.75rem;
		overflow-wrap: anywhere;
	}
	.err {
		color: var(--danger);
		margin-top: 0.15rem;
	}
	.jobs tbody tr {
		transition: background 0.12s;
	}
	.jobs tbody tr:hover {
		background: var(--surface-2);
	}
	.empty {
		text-align: center;
		padding: 3rem 1.5rem;
		display: grid;
		justify-items: center;
		gap: 0.25rem;
	}
	.empty p {
		max-width: 34rem;
	}
	.skeleton {
		display: grid;
		gap: 0.8rem;
	}
	.sk-row {
		height: 2.4rem;
		border-radius: var(--radius-sm);
		background: linear-gradient(90deg, var(--surface-2), var(--surface-3), var(--surface-2));
		background-size: 200% 100%;
		animation: sk 1.4s linear infinite;
	}
	@keyframes sk {
		to {
			background-position: -200% 0;
		}
	}

	/* Phone: each job becomes a stacked card. */
	@media (max-width: 640px) {
		.list {
			padding: 0;
			background: transparent;
			border: none;
			box-shadow: none;
		}
		.jobs thead {
			display: none;
		}
		.jobs,
		.jobs tbody,
		.jobs tr,
		.jobs td {
			display: block;
			width: 100%;
		}
		.jobs tr {
			background: var(--surface);
			border: 1px solid var(--border);
			border-radius: var(--radius);
			box-shadow: var(--shadow);
			padding: 0.75rem 0.9rem;
			margin-bottom: 0.75rem;
			display: grid;
			grid-template-columns: 1fr 1fr;
			gap: 0.35rem 1rem;
		}
		.jobs td {
			border: none;
			padding: 0;
		}
		.jobs td.name {
			grid-column: 1 / -1;
			margin-bottom: 0.25rem;
		}
		.jobs td[data-label]::before {
			content: attr(data-label);
			display: block;
			font-size: 0.7rem;
			text-transform: uppercase;
			letter-spacing: 0.05em;
			color: var(--faint);
		}
	}
</style>
