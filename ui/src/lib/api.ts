// Typed client for the Têmpera HTTP API (docs/wiki/Contracts.md §9).

import type {
	Algorithm,
	AuditResponse,
	DatasetResponse,
	JobDetail,
	JobRow,
	Machine,
	MetricPoint,
	SpecIssue,
	ValidateResponse
} from './types';

export const API_BASE = '/api/v1';

/** Error body: `{"error": "...", "issues": [...]?}` with a 4xx/5xx status. */
export class ApiError extends Error {
	readonly status: number;
	readonly issues: SpecIssue[];

	constructor(status: number, message: string, issues: SpecIssue[] = []) {
		super(message);
		this.name = 'ApiError';
		this.status = status;
		this.issues = issues;
	}
}

type Fetch = typeof fetch;

async function request<T>(
	path: string,
	init: RequestInit = {},
	fetchFn: Fetch = fetch
): Promise<T> {
	let res: Response;
	try {
		res = await fetchFn(API_BASE + path, init);
	} catch (e) {
		throw new ApiError(0, `Cannot reach the Têmpera server (${(e as Error).message}).`);
	}
	const text = await res.text();
	let body: unknown = undefined;
	if (text) {
		try {
			body = JSON.parse(text);
		} catch {
			body = text;
		}
	}
	if (!res.ok) {
		const b = (body ?? {}) as { error?: string; issues?: SpecIssue[] };
		const msg =
			typeof body === 'string' && body ? body : (b.error ?? `${res.status} ${res.statusText}`);
		throw new ApiError(res.status, msg, b.issues ?? []);
	}
	return body as T;
}

function json(method: string, body: unknown): RequestInit {
	return {
		method,
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify(body)
	};
}

export const api = {
	health: (f?: Fetch) => request<{ ok: boolean; version: string }>('/health', {}, f),

	uploadDataset(file: File, f?: Fetch): Promise<DatasetResponse> {
		const form = new FormData();
		form.append('file', file, file.name);
		return request('/datasets', { method: 'POST', body: form }, f);
	},

	dataset: (id: string, f?: Fetch) =>
		request<DatasetResponse>(`/datasets/${encodeURIComponent(id)}`, {}, f),

	validateSpec: (yaml: string, f?: Fetch) =>
		request<ValidateResponse>('/specs/validate', json('POST', { yaml }), f),

	draftSpec: (body: { dataset_id: string; target?: string; name?: string }, f?: Fetch) =>
		request<{ yaml: string }>('/specs/draft', json('POST', body), f),

	audit: (yaml: string, f?: Fetch) => request<AuditResponse>('/audit', json('POST', { yaml }), f),

	algorithms: (f?: Fetch) => request<Algorithm[]>('/algorithms', {}, f),

	machines: (f?: Fetch) => request<Machine[]>('/machines', {}, f),

	createJob: (yaml: string, f?: Fetch) =>
		request<{ job_id: string }>('/jobs', json('POST', { yaml }), f),

	jobs: (f?: Fetch) => request<JobRow[]>('/jobs', {}, f),

	job: (id: string, f?: Fetch) => request<JobDetail>(`/jobs/${encodeURIComponent(id)}`, {}, f),

	cancelJob: (id: string, f?: Fetch) =>
		request<Record<string, never>>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }, f),

	runMetrics: (runId: string, name?: string, f?: Fetch) =>
		request<MetricPoint[]>(
			`/runs/${encodeURIComponent(runId)}/metrics` +
				(name ? `?name=${encodeURIComponent(name)}` : ''),
			{},
			f
		)
};

export function reportUrl(jobId: string): string {
	return `${API_BASE}/jobs/${encodeURIComponent(jobId)}/report`;
}

/** Absolute ws:// or wss:// URL of a job's live event stream. */
export function eventsUrl(jobId: string, loc: Location = window.location): string {
	const proto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
	return `${proto}//${loc.host}${API_BASE}/jobs/${encodeURIComponent(jobId)}/events`;
}
