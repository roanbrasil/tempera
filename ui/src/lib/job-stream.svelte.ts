// WebSocket connection to `/api/v1/jobs/{id}/events` with reconnect/backoff, exposing the
// reduced job state as Svelte state. Updates are batched to one per animation frame so a
// 10 Hz × 10 curves stream does not re-render on every message.

import { eventsUrl } from './api';
import { initialState, parseEvent, ReplayingState, type JobState } from './events';

export type ConnectionState = 'connecting' | 'open' | 'reconnecting' | 'closed';

const MAX_BACKOFF_MS = 15_000;

export class JobStream {
	/** Snapshot of the reduced state; replaced (not mutated) on every flush. */
	view: JobState = $state.raw(initialState());
	connection: ConnectionState = $state('connecting');
	/** Seconds until the next reconnect attempt (while `reconnecting`). */
	retryIn: number = $state(0);

	private readonly rs = new ReplayingState();
	private ws: WebSocket | null = null;
	private attempts = 0;
	private retryTimer: ReturnType<typeof setTimeout> | null = null;
	private countdown: ReturnType<typeof setInterval> | null = null;
	private frame: number | null = null;
	private disposed = false;
	private readonly jobId: string;

	constructor(jobId: string) {
		this.jobId = jobId;
	}

	start(): void {
		this.disposed = false;
		this.connect();
		document.addEventListener('visibilitychange', this.onVisible);
	}

	stop(): void {
		this.disposed = true;
		document.removeEventListener('visibilitychange', this.onVisible);
		this.clearTimers();
		if (this.frame !== null) cancelAnimationFrame(this.frame);
		this.frame = null;
		if (this.ws) {
			this.ws.onclose = null;
			this.ws.close();
			this.ws = null;
		}
	}

	/** Reconnect now (e.g. from a "retry" button). */
	retryNow(): void {
		if (this.connection === 'open' || this.disposed) return;
		this.clearTimers();
		this.connect();
	}

	private onVisible = (): void => {
		if (document.visibilityState === 'visible') this.flush();
	};

	private clearTimers(): void {
		if (this.retryTimer) clearTimeout(this.retryTimer);
		if (this.countdown) clearInterval(this.countdown);
		this.retryTimer = null;
		this.countdown = null;
	}

	private connect(): void {
		this.connection = this.attempts === 0 ? 'connecting' : 'reconnecting';
		let ws: WebSocket;
		try {
			ws = new WebSocket(eventsUrl(this.jobId));
		} catch {
			this.scheduleReconnect();
			return;
		}
		this.ws = ws;
		ws.onopen = () => {
			this.attempts = 0;
			this.connection = 'open';
			this.rs.beginReplay();
		};
		ws.onmessage = (m: MessageEvent) => {
			if (typeof m.data !== 'string') return;
			const ev = parseEvent(m.data);
			if (!ev) return;
			this.rs.apply(ev);
			this.schedule();
		};
		ws.onclose = () => {
			this.ws = null;
			if (this.disposed) return;
			this.flush();
			// The server closes the socket after `job_finished`: nothing more will come.
			if (this.rs.state.finished && !this.rs.replaying) {
				this.connection = 'closed';
				return;
			}
			this.scheduleReconnect();
		};
	}

	private scheduleReconnect(): void {
		this.connection = 'reconnecting';
		const base = Math.min(MAX_BACKOFF_MS, 500 * 2 ** this.attempts);
		const delay = base / 2 + Math.random() * (base / 2);
		this.attempts++;
		this.retryIn = Math.ceil(delay / 1000);
		this.clearTimers();
		this.countdown = setInterval(() => (this.retryIn = Math.max(0, this.retryIn - 1)), 1000);
		this.retryTimer = setTimeout(() => {
			this.clearTimers();
			if (!this.disposed) this.connect();
		}, delay);
	}

	private schedule(): void {
		if (this.frame !== null) return;
		if (document.visibilityState === 'hidden') {
			// rAF does not run in background tabs; flush on visibility change instead.
			return;
		}
		this.frame = requestAnimationFrame(() => {
			this.frame = null;
			this.flush();
		});
	}

	private flush(): void {
		const s = this.rs.state;
		// Shallow copies so that keyed blocks and props see new identities.
		this.view = {
			...s,
			stages: { ...s.stages },
			runs: s.runs.map((r) => ({ ...r, live: { ...r.live } })),
			logs: s.logs.slice()
		};
	}
}
