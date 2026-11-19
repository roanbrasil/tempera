<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { api, ApiError, reportUrl } from '#lib/api.js';
	import { JobStream } from '#lib/job-stream.svelte.js';
	import { isTerminalRun, mergeRuns } from '#lib/events.js';
	import type { JobDetail } from '#lib/types.js';
	import StatusBadge from '#lib/components/StatusBadge.svelte';
	import SeverityDot from '#lib/components/SeverityDot.svelte';
	import Ticks from '#lib/components/Ticks.svelte';
	import StageTimeline from '#lib/job/StageTimeline.svelte';
	import RunsTable from '#lib/job/RunsTable.svelte';
	import ChartPanel from '#lib/job/ChartPanel.svelte';
	import LogPanel from '#lib/job/LogPanel.svelte';
	import WinnerCard from '#lib/job/WinnerCard.svelte';
	import { fmtDateTime, fmtDuration, metricLabel, taskLabel } from '#lib/format.js';

	const id = $derived(page.params.id ?? '');

	let detail = $state.raw<JobDetail | null>(null);
	let loadError: string | null = $state(null);
	let stream = $state.raw<JobStream | null>(null);
	let picked: string | null = $state(null);
	let cancelArmed = $state(false);
	let cancelling = $state(false);
	let actionError: string | null = $state(null);
	let showReport = $state(false);

	async function loadDetail(jobId: string) {
		try {
			const d = await api.job(jobId);
			if (jobId === id) {
				detail = d;
				loadError = null;
			}
		} catch (e) {
			if (jobId === id) loadError = e instanceof ApiError ? e.message : String(e);
		}
	}

	// One stream (and one detail fetch) per job id.
	$effect(() => {
		const jobId = id;
		detail = null;
		picked = null;
		showReport = false;
		const s = new JobStream(jobId);
		stream = s;
		s.start();
		loadDetail(jobId);
		return () => s.stop();
	});

	const view = $derived(stream?.view ?? null);
	const finished = $derived(view?.finished ?? null);

	// Refresh the stored rows when the job ends (winner, report path, final statuses).
	$effect(() => {
		if (finished) loadDetail(id);
	});

	const job = $derived(detail?.job ?? null);
	const status = $derived(
		finished?.status ?? (view && view.events > 0 ? 'running' : (job?.status ?? 'running'))
	);
	const running = $derived(!finished && (status === 'running' || status === 'created'));
	const runs = $derived(mergeRuns(view?.runs ?? [], detail?.runs ?? []));
	const task = $derived(view?.inference?.task ?? job?.task ?? null);
	const metric = $derived(view?.inference?.metric ?? job?.metric ?? null);
	const winnerId = $derived(view?.winner?.run_id ?? job?.winner_run_id ?? null);
	const activeRun = $derived(
		[...runs].reverse().find((r) => !isTerminalRun(r.status))?.run_id ?? null
	);
	// Follow the active run until the user picks one; afterwards show the winner.
	// Default: the most recent run that has curves, then the winner, then the active run.
	const lastWithCurves = $derived(
		[...runs].reverse().find((r) => Object.keys(view?.series[r.run_id] ?? {}).length > 0)?.run_id ??
			null
	);
	const selected = $derived(
		picked ?? lastWithCurves ?? winnerId ?? activeRun ?? runs[runs.length - 1]?.run_id ?? null
	);
	const doneRuns = $derived(runs.filter((r) => isTerminalRun(r.status)).length);
	const total = $derived(Math.max(view?.totalRuns ?? 0, runs.length));
	const report = $derived(finished?.report_url ?? (job?.report_path ? reportUrl(id) : null));
	const winnerRun = $derived(runs.find((r) => r.run_id === winnerId) ?? null);
	const blockedFindings = $derived(
		view?.blocked ?? view?.audit?.findings.filter((f) => f.severity === 'red') ?? []
	);

	async function cancel() {
		if (!cancelArmed) {
			cancelArmed = true;
			setTimeout(() => (cancelArmed = false), 4000);
			return;
		}
		cancelArmed = false;
		cancelling = true;
		actionError = null;
		try {
			await api.cancelJob(id);
		} catch (e) {
			actionError = e instanceof ApiError ? e.message : String(e);
		} finally {
			cancelling = false;
		}
	}
</script>

<svelte:head><title>{job?.name ?? id} · têmpera</title></svelte:head>

<div class="page">
	<nav class="crumbs small"><a href={resolve('/')}>Jobs</a> <span class="faint">/</span></nav>

	<header class="head">
		<div class="title">
			<h1>{job?.name ?? id}</h1>
			<div class="meta">
				<StatusBadge {status} />
				<span>{taskLabel(task)}</span>
				<span class="faint">·</span>
				<span>{metricLabel(metric)}</span>
				{#if job}<span class="faint">·</span><span class="muted">
						{fmtDateTime(job.created_at)}
					</span>{/if}
				{#if stream}
					<span class="conn {stream.connection}" title="Live event stream">
						<span class="led"></span>
						{#if stream.connection === 'open'}live{:else if stream.connection === 'connecting'}connecting…{:else if stream.connection === 'reconnecting'}reconnecting{stream.retryIn
								? ` in ${stream.retryIn}s`
								: '…'}{:else}ended{/if}
					</span>
					{#if stream.connection === 'reconnecting'}
						<button class="btn small ghost" onclick={() => stream?.retryNow()}>Retry now</button>
					{/if}
				{/if}
			</div>
		</div>
		<div class="actions">
			{#if report && finished}
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- server endpoint, not an app route -->
				<a class="btn" href={report} target="_blank" rel="noopener">Report ↗</a>
			{/if}
			{#if running}
				<button class="btn danger" onclick={cancel} disabled={cancelling}>
					{#if cancelling}<span class="spinner"></span>{/if}
					{cancelArmed ? 'Click again to cancel' : 'Cancel job'}
				</button>
			{/if}
		</div>
	</header>

	{#if loadError && !view?.events}
		<div class="alert danger" role="alert">{loadError}</div>
	{/if}
	{#if actionError}
		<div class="alert danger" role="alert">Cancel failed: {actionError}</div>
	{/if}

	<section class="card">
		<StageTimeline
			current={view?.stage ?? null}
			messages={view?.stages ?? {}}
			finished={finished?.status ?? null}
		/>
		{#if total > 0}
			<div class="overall">
				<div class="progress">
					<span style:width="{(doneRuns / total) * 100}%"></span>
				</div>
				<div class="row small">
					<span><strong>{doneRuns}</strong> of {total} runs done</span>
					<span class="spacer"></span>
					{#if running && view?.jobEtaS}
						<span>Job ETA <strong class="num">{fmtDuration(view.jobEtaS)}</strong></span>
					{/if}
				</div>
			</div>
		{/if}
	</section>

	{#if finished && finished.status !== 'succeeded'}
		<div
			class="alert"
			class:danger={finished.status === 'failed' || finished.status === 'blocked'}
			class:warn={finished.status === 'cancelled'}
			role="status"
		>
			<strong>
				{finished.status === 'blocked'
					? 'The audit blocked this job.'
					: finished.status === 'cancelled'
						? 'The job was cancelled.'
						: 'The job failed.'}
			</strong>
			{#if finished.error}<span class="mono small"> {finished.error}</span>{/if}
			{#if finished.status === 'blocked' && blockedFindings.length > 0}
				<ul class="blocked">
					{#each blockedFindings as f (f.id)}
						<li>
							<SeverityDot severity={f.severity} size={9} />
							<span>
								<strong>{f.title}</strong> — <Ticks text={f.evidence} />
								<span class="muted"><Ticks text={f.suggestion} /></span>
							</span>
						</li>
					{/each}
				</ul>
				<a class="btn small" href={resolve('/new')}>Start again with the guided flow</a>
			{/if}
		</div>
	{/if}

	{#if view?.winner}
		<WinnerCard
			algorithm={view.winner.algorithm}
			test={view.winner.test}
			{metric}
			run={winnerRun}
			reportHref={report ?? reportUrl(id)}
		/>
	{/if}

	<section class="card flush">
		<div class="sec-head">
			<h2>Runs</h2>
			<span class="muted small">Baseline first; every candidate must beat it on validation.</span>
		</div>
		<RunsTable
			{runs}
			{selected}
			onselect={(rid) => (picked = rid)}
			{metric}
			{winnerId}
			totalRuns={view?.totalRuns ?? null}
		/>
	</section>

	<div class="two">
		<section class="card">
			<h2>Curves</h2>
			<ChartPanel
				{runs}
				series={view?.series ?? {}}
				metricRev={view?.metricRev ?? 0}
				{selected}
				{metric}
			/>
		</section>
		<section class="card">
			<h2>Logs</h2>
			<LogPanel logs={view?.logs ?? []} {runs} dropped={view?.logsDropped ?? 0} />
		</section>
	</div>

	{#if report && finished?.status === 'succeeded'}
		<section class="card">
			<div class="row">
				<h2 class="nomb">Report</h2>
				<span class="spacer"></span>
				<button class="btn small" onclick={() => (showReport = !showReport)}>
					{showReport ? 'Hide' : 'Show here'}
				</button>
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- server endpoint, not an app route -->
				<a class="btn small" href={report} target="_blank" rel="noopener">Open in a new tab ↗</a>
			</div>
			{#if showReport}
				<iframe class="report" src={report} title="Job report"></iframe>
			{/if}
		</section>
	{/if}
</div>

<style>
	.page {
		display: grid;
		gap: 1.1rem;
		min-width: 0;
	}
	.crumbs {
		margin-bottom: -0.6rem;
	}
	.head {
		display: flex;
		gap: 1rem;
		align-items: flex-start;
		justify-content: space-between;
		flex-wrap: wrap;
	}
	.title {
		min-width: 0;
		flex: 1;
	}
	.title h1 {
		overflow-wrap: anywhere;
		margin-bottom: 0.35rem;
	}
	.meta {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		flex-wrap: wrap;
		font-size: 0.92rem;
	}
	.actions {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	.conn {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		font-size: 0.78rem;
		color: var(--muted);
		margin-left: 0.4rem;
	}
	.conn .led {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--faint);
	}
	.conn.open .led {
		background: var(--ok);
		animation: pulse 1.6s ease-in-out infinite;
	}
	.conn.reconnecting .led,
	.conn.connecting .led {
		background: var(--warn);
	}
	.overall {
		margin-top: 1.1rem;
		display: grid;
		gap: 0.4rem;
	}
	.flush {
		padding: 0.9rem 0.4rem 0.4rem;
	}
	.sec-head {
		display: flex;
		align-items: baseline;
		gap: 0.75rem;
		flex-wrap: wrap;
		padding: 0 0.8rem 0.4rem;
	}
	.sec-head h2,
	.nomb {
		margin: 0;
	}
	.two {
		display: grid;
		grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
		gap: 1.1rem;
	}
	.blocked {
		list-style: none;
		padding: 0;
		margin: 0.6rem 0;
		display: grid;
		gap: 0.4rem;
	}
	.blocked li {
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
	}
	.report {
		width: 100%;
		height: 75vh;
		margin-top: 0.75rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: #fff;
	}
	@media (max-width: 1000px) {
		.two {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@media (max-width: 760px) {
		.flush {
			padding: 0.8rem 0.2rem 0.2rem;
		}
	}
</style>
