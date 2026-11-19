<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';
	import { api } from '#lib/api.js';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();

	let version = $state<string | null>(null);
	let online = $state<boolean | null>(null);

	async function ping() {
		try {
			const h = await api.health();
			online = h.ok;
			version = h.version;
		} catch {
			online = false;
		}
	}

	onMount(() => {
		ping();
		const t = setInterval(ping, 15_000);
		return () => clearInterval(t);
	});

	const path = $derived(page.url.pathname);
	const onJobs = $derived(path === '/' || path.startsWith('/jobs'));
	const onNew = $derived(path.startsWith('/new'));
</script>

<div class="shell">
	<header class="top">
		<div class="inner">
			<a class="brand" href={resolve('/')} aria-label="têmpera home">
				<img src="/mark.png" alt="" width="34" height="34" />
				<span class="word">têmpera</span>
			</a>
			<nav aria-label="Main">
				<a href={resolve('/')} class:active={onJobs} aria-current={onJobs ? 'page' : undefined}>
					Jobs
				</a>
				<a href={resolve('/new')} class:active={onNew} aria-current={onNew ? 'page' : undefined}>
					New job
				</a>
			</nav>
			<span class="spacer"></span>
			<span
				class="server"
				class:off={online === false}
				title={online === false
					? 'The Têmpera server is not reachable'
					: version
						? `têmpera ${version}`
						: 'Checking server…'}
			>
				<span class="led"></span>
				<span class="label">
					{online === false ? 'offline' : version ? `v${version}` : 'connecting'}
				</span>
			</span>
		</div>
		<div class="band" aria-hidden="true"></div>
	</header>

	<main>
		{@render children()}
	</main>
</div>

<style>
	.shell {
		min-height: 100dvh;
		display: flex;
		flex-direction: column;
	}

	.top {
		position: sticky;
		top: 0;
		z-index: 20;
		background: color-mix(in srgb, var(--surface) 92%, transparent);
		backdrop-filter: saturate(1.4) blur(10px);
		border-bottom: 1px solid var(--border);
	}

	.inner {
		max-width: 1240px;
		margin: 0 auto;
		padding: 0 1rem;
		height: 56px;
		display: flex;
		align-items: center;
		gap: 1.25rem;
	}

	.band {
		height: 3px;
		background: var(--band);
		opacity: 0.9;
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		color: var(--text);
		text-decoration: none;
	}
	.brand img {
		display: block;
	}
	.word {
		font-size: 1.3rem;
		font-weight: 700;
		letter-spacing: -0.02em;
	}

	nav {
		display: flex;
		gap: 0.25rem;
	}
	nav a {
		color: var(--muted);
		font-weight: 550;
		padding: 0.4rem 0.7rem;
		border-radius: var(--radius-sm);
		text-decoration: none;
	}
	nav a:hover {
		color: var(--text);
		background: var(--surface-2);
	}
	nav a.active {
		color: var(--text);
		background: var(--accent-soft);
	}

	.server {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.8rem;
		color: var(--muted);
	}
	.led {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--ok);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 20%, transparent);
	}
	.server.off .led {
		background: var(--danger);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--danger) 20%, transparent);
	}

	main {
		flex: 1;
		width: 100%;
		max-width: 1240px;
		margin: 0 auto;
		padding: 1.5rem 1rem 3rem;
		min-width: 0;
	}

	@media (max-width: 560px) {
		.inner {
			gap: 0.6rem;
		}
		.word {
			font-size: 1.1rem;
		}
		.brand img {
			width: 28px;
			height: 28px;
		}
		nav a {
			padding: 0.35rem 0.5rem;
		}
		.server .label {
			display: none;
		}
		main {
			padding-top: 1rem;
		}
	}
</style>
