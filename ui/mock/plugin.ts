// Vite plugin that serves a simulated Têmpera API (Contracts §9) for `npm run dev:mock`.

import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Duplex } from 'node:stream';
import type { Plugin } from 'vite';
import { WebSocketServer } from 'ws';
import {
	churnCsv,
	draftYaml,
	ingest,
	readSpec,
	runAudit,
	validate,
	type StoredDataset
} from './data.ts';
import { MockJob, type JobPlan } from './sim.ts';

const SPEED = Number(process.env.MOCK_SPEED ?? 1) || 1;

const ALGORITHMS = (() => {
	const cls = ['binary_classification', 'multiclass_classification'];
	const out: { key: string; backend: string; tasks: string[] }[] = [
		{ key: 'majority', backend: 'builtin', tasks: cls },
		{ key: 'mean', backend: 'builtin', tasks: ['regression'] },
		{ key: 'linear_regression', backend: 'native', tasks: ['regression'] },
		{ key: 'ridge', backend: 'native', tasks: ['regression'] },
		{ key: 'logistic_regression', backend: 'native', tasks: cls },
		{ key: 'decision_tree', backend: 'native', tasks: cls }
	];
	const py: [string, boolean, boolean][] = [
		['logistic_regression', true, false],
		['linear_regression', false, true],
		['ridge', false, true],
		['lasso', false, true],
		['elastic_net', false, true],
		['sgd', true, true],
		['random_forest', true, true],
		['extra_trees', true, true],
		['hist_gradient_boosting', true, true],
		['adaboost', true, true],
		['lightgbm', true, true],
		['xgboost', true, true],
		['catboost', true, true],
		['knn', true, true],
		['svm', true, true],
		['naive_bayes', true, false],
		['mlp', true, true]
	];
	for (const [key, c, r] of py)
		out.push({ key, backend: 'python', tasks: [...(c ? cls : []), ...(r ? ['regression'] : [])] });
	for (const key of ['kmeans', 'dbscan', 'pca_kmeans'])
		out.push({ key, backend: 'python', tasks: ['clustering'] });
	out.push({ key: 'isolation_forest', backend: 'python', tasks: ['anomaly_detection'] });
	return out;
})();

const MACHINES = [
	{
		id: 'local',
		kind: 'local',
		status: 'ready',
		hardware: {
			os: 'linux',
			arch: 'x86_64',
			cpu_cores: 16,
			ram_bytes: 32 * 1024 ** 3,
			python_available: true,
			gpus: [{ name: 'NVIDIA GeForce RTX 3060', vram_bytes: 12 * 1024 ** 3 }]
		}
	}
];

function jobId(name: string, at: Date): string {
	const stamp = at
		.toISOString()
		.replace(/[-:]/g, '')
		.replace('T', '-')
		.replace(/\.\d+Z$/, '');
	return `${name}-${stamp}-${Math.floor(Math.random() * 0x10000)
		.toString(16)
		.padStart(4, '0')}`;
}

class MockState {
	datasets = new Map<string, StoredDataset>();
	jobs = new Map<string, MockJob>();

	constructor() {
		const churn = ingest('churn.csv', Buffer.from(churnCsv()));
		this.datasets.set(churn.dataset.id, churn);
		const now = Date.now();

		// A finished job, a blocked job, a failed job and one that is training right now.
		const fixed = this.datasets.get(churn.dataset.id)!;
		const spec = (drop: string[], target = 'churned') =>
			readSpec(
				`version: 1\nname: x\ndata:\n  source: uploads/churn.csv\n  target: ${target}\n  drop: [${drop.join(', ')}]\n`
			);

		this.seed('customer-churn', fixed, spec(['refund_amount']), new Date(now - 2 * 86400e3), false);
		this.seed('churn-first-try', fixed, spec([]), new Date(now - 26 * 3600e3), false);
		const houses = this.housesDataset();
		this.seed(
			'house-prices',
			houses,
			readSpec('version: 1\nname: x\ndata:\n  source: uploads/houses.csv\n  target: price\n'),
			new Date(now - 5 * 3600e3),
			false,
			true
		);
		this.seed('customer-churn-v2', fixed, spec(['refund_amount']), new Date(now - 20e3), true);
	}

	private housesDataset(): StoredDataset {
		const lines = ['house_id,area_m2,rooms,neighborhood,year_built,price'];
		let s = 11;
		const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
		for (let i = 0; i < 800; i++) {
			const area = Math.round(40 + rnd() * 200);
			const rooms = 1 + Math.floor(area / 45);
			const hood = ['centro', 'jardins', 'moema', 'pinheiros'][Math.floor(rnd() * 4)];
			const year = 1960 + Math.floor(rnd() * 64);
			const price = Math.round(area * (3000 + rnd() * 9000) + rnd() * 600000);
			lines.push(`H${i},${area},${rooms},${hood},${year},${price}`);
		}
		const ds = ingest('houses.csv', Buffer.from(lines.join('\n')));
		this.datasets.set(ds.dataset.id, ds);
		return ds;
	}

	private seed(
		name: string,
		ds: StoredDataset,
		spec: ReturnType<typeof readSpec>,
		at: Date,
		live: boolean,
		failLast = false
	): void {
		const { inference, audit } = runAudit(ds, spec);
		const plan: JobPlan = {
			name,
			inference,
			audit,
			rows: ds.profile.rows,
			candidates: 'auto',
			baseline: true,
			failLast
		};
		const job = new MockJob(jobId(name, at), plan, at);
		this.jobs.set(job.id, job);
		void job.run(live, SPEED);
	}

	findDataset(source: string): StoredDataset | undefined {
		// Newest upload wins (the same file name may be uploaded several times).
		for (const d of [...this.datasets.values()].reverse())
			if (d.dataset.source === source) return d;
		return undefined;
	}

	createJob(yaml: string): MockJob {
		const spec = readSpec(yaml);
		const ds = this.findDataset(spec.source) ?? [...this.datasets.values()][0];
		const { inference, audit } = runAudit(ds, spec);
		const plan: JobPlan = {
			name: spec.name,
			inference,
			audit,
			rows: ds.profile.rows,
			candidates: spec.candidates,
			baseline: spec.baseline
		};
		const now = new Date();
		const job = new MockJob(jobId(spec.name, now), plan, now);
		this.jobs.set(job.id, job);
		void job.run(true, SPEED);
		return job;
	}
}

// ---------------------------------------------------------------- HTTP helpers

function send(res: ServerResponse, status: number, body: unknown, type = 'application/json'): void {
	res.statusCode = status;
	res.setHeader('content-type', type);
	res.end(type === 'application/json' ? JSON.stringify(body) : String(body));
}

function fail(res: ServerResponse, status: number, error: string, issues?: unknown[]): void {
	send(res, status, issues ? { error, issues } : { error });
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
	const chunks: Buffer[] = [];
	for await (const c of req) chunks.push(c as Buffer);
	return Buffer.concat(chunks);
}

async function readJson<T>(req: IncomingMessage): Promise<T> {
	const b = await readBody(req);
	return JSON.parse(b.toString('utf8') || '{}') as T;
}

/** Minimal multipart/form-data parser: returns the `file` field. */
function multipartFile(
	body: Buffer,
	contentType: string
): { filename: string; data: Buffer } | null {
	const m = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType);
	if (!m) return null;
	const boundary = Buffer.from(`--${m[1] ?? m[2]}`);
	let pos = body.indexOf(boundary);
	while (pos !== -1) {
		const next = body.indexOf(boundary, pos + boundary.length);
		if (next === -1) break;
		const part = body.subarray(pos + boundary.length + 2, next - 2);
		const sep = part.indexOf('\r\n\r\n');
		if (sep !== -1) {
			const head = part.subarray(0, sep).toString('utf8');
			if (/name="file"/.test(head)) {
				const fn = /filename="([^"]*)"/.exec(head)?.[1] ?? 'upload.csv';
				return { filename: fn, data: part.subarray(sep + 4) };
			}
		}
		pos = next;
	}
	return null;
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------- plugin

export function mockApi(): Plugin {
	return {
		name: 'tempera-mock-api',
		apply: 'serve',
		configureServer(server) {
			const st = new MockState();
			const wss = new WebSocketServer({ noServer: true });

			server.httpServer?.on('upgrade', (req: IncomingMessage, socket: Duplex, head: Buffer) => {
				const m = /^\/api\/v1\/jobs\/([^/?]+)\/events/.exec(req.url ?? '');
				if (!m) return;
				wss.handleUpgrade(req, socket, head, (ws) => {
					const job = st.jobs.get(decodeURIComponent(m[1]));
					if (!job) {
						ws.send(
							JSON.stringify({
								type: 'job_finished',
								status: 'unknown',
								report_url: null,
								error: 'unknown job'
							})
						);
						ws.close();
						return;
					}
					// Replay the history, then stream; close after `job_finished` like the server.
					for (const ev of job.history) ws.send(JSON.stringify(ev));
					if (job.finished) {
						ws.close();
						return;
					}
					const off = job.subscribe((ev) => {
						ws.send(JSON.stringify(ev));
						if (ev.type === 'job_finished') {
							off();
							ws.close();
						}
					});
					ws.on('close', off);
				});
			});

			server.middlewares.use(async (req, res, next) => {
				const url = new URL(req.url ?? '/', 'http://mock');
				if (!url.pathname.startsWith('/api/v1/')) return next();
				const path = url.pathname.slice('/api/v1'.length);
				const method = req.method ?? 'GET';
				await delay(120 + Math.random() * 180); // feel like a real server
				try {
					let m: RegExpExecArray | null;
					if (method === 'GET' && path === '/health')
						return send(res, 200, { ok: true, version: '0.1.0-mock' });

					if (method === 'POST' && path === '/datasets') {
						const body = await readBody(req);
						const file = multipartFile(body, req.headers['content-type'] ?? '');
						if (!file) return fail(res, 400, 'missing multipart field `file`');
						const ds = ingest(file.filename, file.data);
						st.datasets.set(ds.dataset.id, ds);
						await delay(500);
						return send(res, 200, { dataset: ds.dataset, profile: ds.profile });
					}
					if (method === 'GET' && (m = /^\/datasets\/([^/]+)$/.exec(path))) {
						const ds = st.datasets.get(m[1]);
						return ds
							? send(res, 200, { dataset: ds.dataset, profile: ds.profile })
							: fail(res, 404, 'unknown dataset');
					}
					if (method === 'POST' && path === '/specs/validate') {
						const { yaml } = await readJson<{ yaml: string }>(req);
						const issues = validate(yaml ?? '');
						return send(res, 200, { valid: issues.length === 0, issues });
					}
					if (method === 'POST' && path === '/specs/draft') {
						const b = await readJson<{ dataset_id: string; target?: string; name?: string }>(req);
						const ds = st.datasets.get(b.dataset_id);
						if (!ds) return fail(res, 404, `unknown dataset \`${b.dataset_id}\``);
						const name =
							b.name ??
							ds.dataset.source
								.split('/')
								.pop()!
								.replace(/\.[^.]+$/, '');
						return send(res, 200, { yaml: draftYaml(ds, b.target ?? null, name) });
					}
					if (method === 'POST' && path === '/audit') {
						const { yaml } = await readJson<{ yaml: string }>(req);
						const issues = validate(yaml ?? '');
						if (issues.length) return fail(res, 422, 'invalid spec', issues);
						const spec = readSpec(yaml);
						const ds = st.findDataset(spec.source);
						if (!ds)
							return fail(res, 404, `dataset \`${spec.source}\` not found (upload it first)`);
						await delay(300);
						return send(res, 200, runAudit(ds, spec));
					}
					if (method === 'GET' && path === '/algorithms') return send(res, 200, ALGORITHMS);
					if (method === 'GET' && path === '/machines') return send(res, 200, MACHINES);
					if (method === 'POST' && path === '/jobs') {
						const { yaml } = await readJson<{ yaml: string }>(req);
						const issues = validate(yaml ?? '');
						if (issues.length) return fail(res, 422, 'invalid spec', issues);
						const job = st.createJob(yaml);
						return send(res, 202, { job_id: job.id });
					}
					if (method === 'GET' && path === '/jobs') {
						const rows = [...st.jobs.values()]
							.map((j) => j.row)
							.sort((a, b) => b.created_at.localeCompare(a.created_at));
						return send(res, 200, rows);
					}
					if ((m = /^\/jobs\/([^/]+)(\/cancel|\/report)?$/.exec(path))) {
						const job = st.jobs.get(decodeURIComponent(m[1]));
						if (!job) return fail(res, 404, `unknown job \`${m[1]}\``);
						if (method === 'GET' && !m[2]) return send(res, 200, { job: job.row, runs: job.runs });
						if (method === 'POST' && m[2] === '/cancel') {
							if (job.finished) return fail(res, 409, 'the job already finished');
							job.cancel();
							return send(res, 202, {});
						}
						if (method === 'GET' && m[2] === '/report') {
							if (!job.row.report_path) return fail(res, 404, 'no report for this job');
							return send(res, 200, job.reportHtml(), 'text/html; charset=utf-8');
						}
					}
					if (method === 'GET' && (m = /^\/runs\/([^/]+)\/metrics$/.exec(path))) {
						const name = url.searchParams.get('name');
						for (const job of st.jobs.values()) {
							if (!job.runs.some((r) => r.id === m![1])) continue;
							const pts = job.history
								.filter(
									(e) => e.type === 'metric' && e.run_id === m![1] && (!name || e.name === name)
								)
								.map((e) =>
									e.type === 'metric'
										? { name: e.name, step: e.step, value: e.value, unix_ms: Date.now() }
										: null
								);
							return send(res, 200, pts);
						}
						return fail(res, 404, 'unknown run');
					}
					return fail(res, 404, `no route for ${method} ${url.pathname}`);
				} catch (e) {
					return fail(res, 400, (e as Error).message);
				}
			});
		}
	};
}
