// Dev mock: a small CSV profiler, task inference, audit and spec validation that imitate the
// Rust server closely enough to develop and demo the UI. Not used in production builds.

import { createHash } from 'node:crypto';
import { parse as parseYaml } from 'yaml';
import type {
	AuditReport,
	ColumnKind,
	ColumnProfile,
	Dataset,
	Finding,
	Metric,
	Profile,
	SpecIssue,
	TaskInference,
	TaskType
} from '../src/lib/types';

export interface StoredDataset {
	dataset: Dataset;
	profile: Profile;
	/** Raw cells per column (null = missing). */
	cells: Record<string, (string | null)[]>;
}

const NULLS = new Set(['', 'na', 'nan', 'null', 'none', 'n/a']);
const BOOLS = new Set(['true', 'false', 'yes', 'no', 't', 'f', 'y', 'n']);
const DATE_RE = /^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?/;

function splitLine(line: string, sep: string): string[] {
	const out: string[] = [];
	let cur = '';
	let q = false;
	for (let i = 0; i < line.length; i++) {
		const ch = line[i];
		if (q) {
			if (ch === '"' && line[i + 1] === '"') {
				cur += '"';
				i++;
			} else if (ch === '"') q = false;
			else cur += ch;
		} else if (ch === '"') q = true;
		else if (ch === sep) {
			out.push(cur);
			cur = '';
		} else cur += ch;
	}
	out.push(cur);
	return out;
}

function parseCsv(text: string): { header: string[]; rows: string[][]; sep: string } {
	const lines = text
		.replace(/^\uFEFF/, '')
		.split(/\r?\n/)
		.filter((l) => l.length > 0);
	const first = lines[0] ?? '';
	const sep = [',', ';', '\t', '|'].sort(
		(a, b) => first.split(b).length - first.split(a).length
	)[0];
	const header = splitLine(first, sep).map((h) => h.trim());
	const rows = lines.slice(1).map((l) => splitLine(l, sep));
	return { header, rows, sep };
}

function parseJsonl(text: string): { header: string[]; rows: string[][] } {
	const objs = text
		.split(/\r?\n/)
		.filter((l) => l.trim())
		.map((l) => JSON.parse(l) as Record<string, unknown>);
	const header = [...new Set(objs.flatMap((o) => Object.keys(o)))];
	const rows = objs.map((o) =>
		header.map((h) => (o[h] === null || o[h] === undefined ? '' : String(o[h])))
	);
	return { header, rows };
}

function quantile(sorted: number[], q: number): number {
	if (sorted.length === 0) return NaN;
	const pos = (sorted.length - 1) * q;
	const lo = Math.floor(pos);
	const hi = Math.ceil(pos);
	return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function profileColumn(name: string, values: (string | null)[], rows: number): ColumnProfile {
	const present = values.filter((v): v is string => v !== null);
	const nullCount = rows - present.length;
	const distinct = new Map<string, number>();
	for (const v of present) distinct.set(v, (distinct.get(v) ?? 0) + 1);
	const nUnique = distinct.size;
	const uniqueRatio = present.length ? nUnique / present.length : 0;
	const nums = present.map(Number);
	const isNum = present.length > 0 && nums.every((n) => Number.isFinite(n));
	const isBool =
		present.length > 0 && nUnique <= 2 && present.every((v) => BOOLS.has(v.toLowerCase()));
	const isDate = present.length > 0 && !isNum && present.every((v) => DATE_RE.test(v));
	const meanLen = present.reduce((a, v) => a + v.length, 0) / Math.max(1, present.length);

	let kind: ColumnKind;
	if (nUnique <= 1) kind = 'constant';
	else if (isBool) kind = 'boolean';
	else if (isNum) kind = 'numeric';
	else if (isDate) kind = 'datetime';
	else if (meanLen > 40) kind = 'text';
	else if (uniqueRatio > 0.9 && nUnique > 50) kind = 'high_cardinality';
	else kind = 'categorical';

	const col: ColumnProfile = {
		name,
		dtype:
			kind === 'numeric'
				? nums.every(Number.isInteger)
					? 'Int64'
					: 'Float64'
				: kind === 'boolean'
					? 'Boolean'
					: kind === 'datetime'
						? 'Date'
						: 'String',
		kind,
		null_count: nullCount,
		null_fraction: rows ? nullCount / rows : 0,
		n_unique: nUnique,
		unique_ratio: uniqueRatio
	};

	if (kind === 'numeric') {
		const s = [...nums].sort((a, b) => a - b);
		const min = s[0];
		const max = s[s.length - 1];
		const mean = s.reduce((a, b) => a + b, 0) / s.length;
		const std = Math.sqrt(s.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, s.length - 1));
		const histogram = new Array(10).fill(0);
		const w = (max - min) / 10;
		for (const v of s) histogram[w > 0 ? Math.min(9, Math.floor((v - min) / w)) : 0]++;
		col.numeric = {
			min,
			max,
			mean,
			std,
			p01: quantile(s, 0.01),
			p25: quantile(s, 0.25),
			p50: quantile(s, 0.5),
			p75: quantile(s, 0.75),
			p99: quantile(s, 0.99),
			histogram
		};
	}
	if (kind !== 'numeric') {
		col.top_values = [...distinct.entries()]
			.sort((a, b) => b[1] - a[1])
			.slice(0, 10)
			.map(([value, count]) => ({ value, count }));
	}
	if (kind === 'text' || kind === 'categorical' || kind === 'high_cardinality')
		col.mean_length = meanLen;
	if (kind === 'datetime') {
		const s = [...present].sort();
		col.min_time = s[0];
		col.max_time = s[s.length - 1];
	}
	return col;
}

export function ingest(filename: string, bytes: Buffer): StoredDataset {
	const text = bytes.toString('utf8');
	const isJsonl = /\.(jsonl|ndjson)$/i.test(filename);
	if (/\.parquet$/i.test(filename))
		throw new Error(
			'The mock server only profiles CSV/TSV/JSONL files (Parquet needs the real server).'
		);
	const { header, rows } = isJsonl ? parseJsonl(text) : parseCsv(text);
	const sep = isJsonl ? null : parseCsv(text.slice(0, 2000)).sep;
	if (header.length === 0 || rows.length === 0) throw new Error('The file has no rows.');
	const cells: Record<string, (string | null)[]> = {};
	header.forEach((h, i) => {
		cells[h] = rows.map((r) => {
			const v = (r[i] ?? '').trim();
			return NULLS.has(v.toLowerCase()) ? null : v;
		});
	});
	const id = createHash('sha256').update(bytes).update('ingest-v1').digest('hex');
	const columns = header.map((h) => profileColumn(h, cells[h], rows.length));
	const sample = rows.slice(0, 5).map((r) => {
		const o: Record<string, unknown> = {};
		header.forEach((h, i) => {
			const v = r[i] ?? '';
			const c = columns[i];
			o[h] = NULLS.has(v.toLowerCase()) ? null : c.kind === 'numeric' ? Number(v) : v;
		});
		return o;
	});
	const dataset: Dataset = {
		id,
		source: `uploads/${filename}`,
		format: isJsonl ? 'jsonl' : 'csv',
		...(sep
			? { csv: { separator: sep, encoding: 'utf-8', had_bom: text.startsWith('\uFEFF') } }
			: {}),
		rows: rows.length,
		columns: columns.map((c) => ({ name: c.name, dtype: c.dtype })),
		ingest_version: 1
	};
	const profile: Profile = { version: 2, dataset_id: id, rows: rows.length, columns, sample };
	return { dataset, profile, cells };
}

// ------------------------------------------------------------------------ spec

export interface ParsedSpec {
	name: string;
	source: string;
	target: string | null;
	timeColumn: string | null;
	groupColumn: string | null;
	drop: string[];
	acknowledge: string[];
	policy: 'block_on_red' | 'warn_only';
	taskType: TaskType | null;
	metric: Metric | null;
	splitStrategy: string;
	candidates: 'auto' | string[];
	baseline: boolean;
}

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj =>
	v && typeof v === 'object' && !Array.isArray(v) ? (v as Obj) : {};
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);

export function readSpec(yaml: string): ParsedSpec {
	const doc = obj(parseYaml(yaml));
	const data = obj(doc.data);
	const task = obj(doc.task);
	const audit = obj(doc.audit);
	const split = obj(doc.split);
	const strategy = obj(doc.strategy);
	return {
		name: String(doc.name ?? 'job'),
		source: String(data.source ?? ''),
		target: typeof data.target === 'string' ? data.target : null,
		timeColumn: typeof data.time_column === 'string' ? data.time_column : null,
		groupColumn: typeof data.group_column === 'string' ? data.group_column : null,
		drop: strs(data.drop),
		acknowledge: strs(audit.acknowledge),
		policy: audit.policy === 'warn_only' ? 'warn_only' : 'block_on_red',
		taskType: (task.type as TaskType) ?? null,
		metric: (task.metric as Metric) ?? null,
		splitStrategy: String(split.strategy ?? 'auto'),
		candidates: Array.isArray(strategy.candidates) ? strs(strategy.candidates) : 'auto',
		baseline: strategy.baseline !== false
	};
}

const TASKS = [
	'binary_classification',
	'multiclass_classification',
	'regression',
	'time_series_forecast',
	'clustering',
	'anomaly_detection',
	'image_classification',
	'text_classification',
	'llm_fine_tune'
];
const METRICS = [
	'roc_auc',
	'pr_auc',
	'f1',
	'balanced_accuracy',
	'accuracy',
	'log_loss',
	'rmse',
	'mae',
	'mape',
	'mase',
	'r2',
	'silhouette',
	'perplexity'
];
const TOP_KEYS = [
	'version',
	'name',
	'data',
	'task',
	'audit',
	'split',
	'strategy',
	'budget',
	'compute',
	'output'
];
const DATA_KEYS = ['source', 'target', 'time_column', 'group_column', 'sensitive', 'drop'];

export function validate(yaml: string): SpecIssue[] {
	let raw: unknown;
	try {
		raw = parseYaml(yaml);
	} catch (e) {
		return [{ path: '', message: (e as Error).message }];
	}
	const issues: SpecIssue[] = [];
	const doc = obj(raw);
	for (const k of Object.keys(doc))
		if (!TOP_KEYS.includes(k)) issues.push({ path: k, message: `unknown field \`${k}\`` });
	if (doc.version !== 1) issues.push({ path: 'version', message: 'must be 1' });
	if (typeof doc.name !== 'string' || !doc.name.trim())
		issues.push({ path: 'name', message: 'missing job name' });
	const data = obj(doc.data);
	if (!doc.data) issues.push({ path: 'data', message: 'missing field `data`' });
	else {
		if (typeof data.source !== 'string')
			issues.push({ path: 'data.source', message: 'missing field `source`' });
		for (const k of Object.keys(data))
			if (!DATA_KEYS.includes(k))
				issues.push({ path: `data.${k}`, message: `unknown field \`${k}\`` });
	}
	const task = obj(doc.task);
	if (task.type !== undefined && !TASKS.includes(String(task.type)))
		issues.push({ path: 'task.type', message: `unknown task \`${String(task.type)}\`` });
	if (task.metric !== undefined && !METRICS.includes(String(task.metric)))
		issues.push({ path: 'task.metric', message: `unknown metric \`${String(task.metric)}\`` });
	const split = obj(doc.split);
	if (split.test_size !== undefined) {
		const t = Number(split.test_size);
		if (!(t > 0 && t < 1))
			issues.push({ path: 'split.test_size', message: 'must be between 0 and 1' });
	}
	if (
		split.strategy !== undefined &&
		!['auto', 'random', 'stratified', 'group', 'temporal'].includes(String(split.strategy))
	)
		issues.push({
			path: 'split.strategy',
			message: `unknown strategy \`${String(split.strategy)}\``
		});
	const strategy = obj(doc.strategy);
	if (
		strategy.candidates !== undefined &&
		strategy.candidates !== 'auto' &&
		!Array.isArray(strategy.candidates)
	)
		issues.push({
			path: 'strategy.candidates',
			message: 'expected `auto` or a list of algorithms'
		});
	const budget = obj(doc.budget);
	if (budget.max_time !== undefined && !/^\d+[smhd]$/.test(String(budget.max_time)))
		issues.push({
			path: 'budget.max_time',
			message: `invalid duration \`${String(budget.max_time)}\` (use e.g. 30m, 2h)`
		});
	const audit = obj(doc.audit);
	if (audit.policy !== undefined && !['block_on_red', 'warn_only'].includes(String(audit.policy)))
		issues.push({ path: 'audit.policy', message: 'expected `block_on_red` or `warn_only`' });
	return issues;
}

// ----------------------------------------------------------------------- audit

function encodeTarget(values: (string | null)[]): {
	y: (number | null)[];
	positive: string | null;
} {
	const counts = new Map<string, number>();
	for (const v of values) if (v !== null) counts.set(v, (counts.get(v) ?? 0) + 1);
	if (counts.size === 2) {
		const [a, b] = [...counts.keys()].sort();
		const pos = ['1', 'true', 'yes', 'y', 't'].includes(b.toLowerCase())
			? b
			: counts.get(a)! < counts.get(b)!
				? a
				: b;
		return { y: values.map((v) => (v === null ? null : v === pos ? 1 : 0)), positive: pos };
	}
	return {
		y: values.map((v) => (v === null || !Number.isFinite(Number(v)) ? null : Number(v))),
		positive: null
	};
}

function corr(x: (number | null)[], y: (number | null)[]): number {
	let n = 0;
	let sx = 0;
	let sy = 0;
	let sxx = 0;
	let syy = 0;
	let sxy = 0;
	for (let i = 0; i < x.length; i++) {
		const a = x[i];
		const b = y[i];
		if (a === null || b === null) continue;
		n++;
		sx += a;
		sy += b;
		sxx += a * a;
		syy += b * b;
		sxy += a * b;
	}
	if (n < 3) return 0;
	const cov = sxy / n - (sx / n) * (sy / n);
	const vx = sxx / n - (sx / n) ** 2;
	const vy = syy / n - (sy / n) ** 2;
	return vx > 0 && vy > 0 ? cov / Math.sqrt(vx * vy) : 0;
}

export function inferTask(ds: StoredDataset, spec: ParsedSpec): TaskInference {
	const reasons: string[] = [];
	let task: TaskType;
	let metric: Metric;
	let minority: number | undefined;
	const t = spec.target ? ds.profile.columns.find((c) => c.name === spec.target) : undefined;
	if (!spec.target || !t) {
		task = 'clustering';
		metric = 'silhouette';
		reasons.push('No target column: the job groups similar rows (clustering).');
		reasons.push('Silhouette measures how well separated the clusters are, without labels.');
	} else if (
		t.kind === 'boolean' ||
		(t.n_unique === 2 && t.kind !== 'numeric') ||
		(t.kind === 'numeric' && t.n_unique === 2)
	) {
		task = 'binary_classification';
		const counts = (t.top_values ?? []).map((v) => v.count);
		const total = counts.reduce((a, b) => a + b, 0) || 1;
		minority = counts.length ? Math.min(...counts) / total : 0.5;
		reasons.push(`\`${t.name}\` has exactly 2 distinct values → binary classification.`);
		if (minority < 0.2) {
			metric = 'pr_auc';
			reasons.push(
				`The rarer class is only ${(minority * 100).toFixed(1)}% of the rows: PR AUC focuses on that class, where ROC AUC would look optimistic.`
			);
		} else {
			metric = 'roc_auc';
			reasons.push(
				'Classes are reasonably balanced: ROC AUC ranks the rows independently of a threshold.'
			);
		}
	} else if (t.kind === 'numeric' && t.n_unique > 20) {
		task = 'regression';
		metric = 'rmse';
		reasons.push(`\`${t.name}\` is numeric with ${t.n_unique} distinct values → regression.`);
		reasons.push('RMSE is in the same unit as the target and penalises large errors.');
	} else {
		task = 'multiclass_classification';
		metric = 'f1';
		reasons.push(`\`${t.name}\` has ${t.n_unique} classes → multiclass classification.`);
		reasons.push('Macro F1 weighs every class equally, so rare classes count.');
	}
	if (spec.taskType && spec.taskType !== task) {
		reasons.push(`Overridden by \`task.type: ${spec.taskType}\` in the spec.`);
		task = spec.taskType;
	}
	if (spec.metric && spec.metric !== metric) {
		reasons.push(`Metric set to \`${spec.metric}\` by the spec.`);
		metric = spec.metric;
	}
	return { task, metric, reasons, ...(minority !== undefined ? { minority_share: minority } : {}) };
}

export function runAudit(
	ds: StoredDataset,
	spec: ParsedSpec
): { inference: TaskInference; audit: AuditReport } {
	const inference = inferTask(ds, spec);
	const findings: Finding[] = [];
	const ack = new Set(spec.acknowledge);
	const dropped: Record<string, string> = {};
	for (const d of spec.drop) dropped[d] = 'dropped in data.drop';
	const cols = ds.profile.columns.filter(
		(c) => c.name !== spec.target && !spec.drop.includes(c.name)
	);

	const add = (f: Omit<Finding, 'resolution'>, auto?: string) =>
		findings.push({
			...f,
			resolution: auto
				? { status: 'auto', action: auto }
				: ack.has(f.id)
					? { status: 'acknowledged' }
					: { status: 'unresolved' }
		});

	// A1: leakage (|corr| with the target).
	let leaks = 0;
	if (spec.target && ds.cells[spec.target]) {
		const { y } = encodeTarget(ds.cells[spec.target]);
		for (const c of cols.filter((c) => c.kind === 'numeric')) {
			const x = ds.cells[c.name].map((v) => (v === null ? null : Number(v)));
			const r = Math.abs(corr(x, y));
			if (r >= 0.9 || (r >= 0.6 && /refund|chargeback|cancel|closed/i.test(c.name))) {
				leaks++;
				add({
					id: `A1:${c.name}`,
					check: 'A1',
					severity: 'red',
					column: c.name,
					title: 'Probable target leakage',
					evidence: `\`${c.name}\` alone predicts \`${spec.target}\` almost perfectly (|r| = ${r.toFixed(2)}); it is probably only known after the outcome.`,
					suggestion: `Drop \`${c.name}\` unless it is truly available at prediction time.`
				});
			} else if (r >= 0.6) {
				leaks++;
				add({
					id: `A1:${c.name}`,
					check: 'A1',
					severity: 'yellow',
					column: c.name,
					title: 'Suspiciously predictive column',
					evidence: `\`${c.name}\` has |r| = ${r.toFixed(2)} with the target.`,
					suggestion: 'Check that it is known at prediction time.'
				});
			}
		}
	}
	if (spec.target && leaks === 0)
		add({
			id: 'A1',
			check: 'A1',
			severity: 'green',
			title: 'No leakage detected',
			evidence: 'No single column predicts the target suspiciously well.',
			suggestion: 'Nothing to do.'
		});

	// A10: identifier-like columns.
	for (const c of cols.filter((c) => c.kind === 'high_cardinality')) {
		dropped[c.name] = 'identifier-like (one value per row)';
		add(
			{
				id: `A10:${c.name}`,
				check: 'A10',
				severity: 'yellow',
				column: c.name,
				title: 'Identifier-like column',
				evidence: `\`${c.name}\` has ${c.n_unique} distinct values for ${ds.profile.rows} rows; a model would memorise it.`,
				suggestion: 'Dropped automatically. Encode it yourself if it carries signal.'
			},
			'column dropped'
		);
	}
	for (const c of cols.filter((c) => c.kind === 'constant')) dropped[c.name] = 'constant';

	// A4: time-ordered data with a random split.
	const dt = cols.find((c) => c.kind === 'datetime');
	if (dt && !spec.timeColumn && spec.splitStrategy !== 'temporal' && spec.target)
		add({
			id: `A4:${dt.name}`,
			check: 'A4',
			severity: 'yellow',
			column: dt.name,
			title: 'Data has a date column but the split is not temporal',
			evidence: `\`${dt.name}\` spans ${dt.min_time?.slice(0, 10)} → ${dt.max_time?.slice(0, 10)}. A random split lets the model see the future.`,
			suggestion: `If you will predict future rows, set \`data.time_column: ${dt.name}\`.`
		});

	// A3: group column with a random split.
	if (spec.groupColumn && (spec.splitStrategy === 'random' || spec.splitStrategy === 'stratified'))
		add({
			id: 'A3',
			check: 'A3',
			severity: 'yellow',
			column: spec.groupColumn,
			title: 'Same entity can land in train and test',
			evidence: `\`data.group_column: ${spec.groupColumn}\` is set but the split ignores groups.`,
			suggestion: 'Use `split.strategy: group` (or `auto`).'
		});

	// A5: imbalance.
	if (inference.minority_share !== undefined && inference.minority_share < 0.2)
		add(
			{
				id: 'A5',
				check: 'A5',
				severity: 'yellow',
				title: 'Imbalanced classes',
				evidence: `The rarest class is ${(inference.minority_share * 100).toFixed(1)}% of the rows.`,
				suggestion: 'Stratified split and a metric that is fair to the rare class.'
			},
			`stratified split, metric ${inference.metric}`
		);

	// A13: rows without a target.
	const tcol = ds.profile.columns.find((c) => c.name === spec.target);
	if (tcol && tcol.null_count > 0)
		add(
			{
				id: 'target-nulls',
				check: 'A13',
				severity: 'yellow',
				column: tcol.name,
				title: 'Rows without a target',
				evidence: `${tcol.null_count} rows (${(tcol.null_fraction * 100).toFixed(1)}%) have no \`${tcol.name}\`.`,
				suggestion: 'They are removed before training. Check why the label is missing.'
			},
			'rows removed'
		);

	// A2: exact duplicates.
	const names = Object.keys(ds.cells);
	const seen = new Set<string>();
	let dup = 0;
	for (let i = 0; i < ds.profile.rows; i++) {
		const key = names.map((n) => ds.cells[n][i] ?? '\u0000').join('\u0001');
		if (seen.has(key)) dup++;
		else seen.add(key);
	}
	if (dup > 0)
		add(
			{
				id: 'A2',
				check: 'A2',
				severity: 'yellow',
				title: 'Exact duplicate rows',
				evidence: `${dup} rows are exact copies of another row; copies split across train and test inflate the score.`,
				suggestion: 'Deduplicate before splitting.'
			},
			`${dup} duplicates removed before the split`
		);
	else
		add({
			id: 'A2',
			check: 'A2',
			severity: 'green',
			title: 'No duplicate rows',
			evidence: 'Every row is unique.',
			suggestion: 'Nothing to do.'
		});

	const features = cols
		.filter((c) => !(c.name in dropped) && c.kind !== 'datetime' && c.kind !== 'text')
		.map((c) => ({
			name: c.name,
			kind:
				c.kind === 'numeric'
					? ('numeric' as const)
					: c.kind === 'boolean'
						? ('boolean' as const)
						: ('categorical' as const)
		}));
	const blocked =
		spec.policy === 'block_on_red' &&
		findings.some((f) => f.severity === 'red' && f.resolution.status === 'unresolved');
	return {
		inference,
		audit: {
			dataset_id: ds.dataset.id,
			policy: spec.policy,
			findings,
			plan: {
				features,
				datetime: cols.filter((c) => c.kind === 'datetime').map((c) => c.name),
				dropped,
				dedupe: dup > 0
			},
			blocked
		}
	};
}

export function draftYaml(ds: StoredDataset, target: string | null, name: string): string {
	const cols = ds.profile.columns.map((c) => c.name);
	if (target && !cols.includes(target))
		throw new Error(`\`${target}\` is not a column (columns: ${cols.join(', ')})`);
	const safe = name.replace(/[^A-Za-z0-9_-]/g, '-');
	const dt = ds.profile.columns.find((c) => c.kind === 'datetime' && c.name !== target);
	const timeHint = dt
		? `  # time_column: ${dt.name}   # uncomment if you will predict future rows\n`
		: '';
	const targetLine = target
		? `  target: ${target}\n`
		: '  # no target: the job clusters the rows (set task.type: anomaly_detection for outliers)\n';
	return (
		'# Têmpera job spec. Reference: docs/wiki/Contracts.md · schema: `tempera schema`\n' +
		`version: 1\nname: ${safe}\ndata:\n  source: ${ds.dataset.source}\n${targetLine}${timeHint}` +
		'  # group_column: customer_id   # rows of the same entity never split across train/test\n' +
		'  # drop: []                    # columns to ignore (e.g. audit findings)\n' +
		'# audit:\n#   acknowledge: []             # finding ids you reviewed and accept\n' +
		'split:\n  test_size: 0.2\n  seed: 42\n'
	);
}

// ----------------------------------------------------------- churn fixture

/** A churn-like CSV with a planted leak (`refund_amount`), like examples/data/churn.csv. */
export function churnCsv(rows = 1200): string {
	let seed = 7;
	const rnd = () => {
		seed = (seed * 1103515245 + 12345) % 2 ** 31;
		return seed / 2 ** 31;
	};
	const plans = ['basic', 'pro', 'team'];
	const countries = ['BR', 'US', 'PT', 'DE', 'MX'];
	const lines = [
		'customer_id,signup_date,tenure_months,plan,monthly_fee,support_tickets,has_discount,country,refund_amount,churned'
	];
	for (let i = 0; i < rows; i++) {
		const tenure = Math.floor(rnd() * 48) + 1;
		const plan = plans[Math.floor(rnd() * 3)];
		const fee = plan === 'basic' ? 29 : plan === 'pro' ? 59 : 99;
		const tickets = Math.floor(rnd() * rnd() * 8);
		const disc = rnd() < 0.3;
		const p = 0.08 + tickets * 0.05 - tenure * 0.002 + (disc ? -0.04 : 0);
		const churned = rnd() < p;
		const refund = churned
			? (5 + rnd() * 40).toFixed(2)
			: rnd() < 0.02
				? (rnd() * 3).toFixed(2)
				: '0.0';
		const d = new Date(Date.UTC(2023, 0, 1) + Math.floor(rnd() * 600) * 86400000)
			.toISOString()
			.slice(0, 10);
		const country = countries[Math.floor(rnd() * countries.length)];
		lines.push(
			`C${100000 + i},${d},${tenure},${plan},${fee},${tickets},${disc},${country},${rnd() < 0.03 ? '' : refund},${churned ? 'yes' : 'no'}`
		);
	}
	return lines.join('\n') + '\n';
}
