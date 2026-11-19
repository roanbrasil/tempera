// Serde shapes of the Rust types exposed by the HTTP API (docs/wiki/Contracts.md §9).
// Field names are snake_case on the wire and kept as-is here.

// ----------------------------------------------------------------- tempera-core

export type TaskType =
	| 'binary_classification'
	| 'multiclass_classification'
	| 'regression'
	| 'time_series_forecast'
	| 'clustering'
	| 'anomaly_detection'
	| 'image_classification'
	| 'text_classification'
	| 'llm_fine_tune';

export type Metric =
	| 'roc_auc'
	| 'pr_auc'
	| 'f1'
	| 'balanced_accuracy'
	| 'accuracy'
	| 'log_loss'
	| 'rmse'
	| 'mae'
	| 'mape'
	| 'mase'
	| 'r2'
	| 'silhouette'
	| 'perplexity';

export type AuditPolicy = 'block_on_red' | 'warn_only';
export type SplitStrategy = 'auto' | 'random' | 'stratified' | 'group' | 'temporal';

// ------------------------------------------------------- tempera-data · ingest

export type SourceFormat = 'csv' | 'parquet' | 'jsonl';

export interface CsvDialect {
	separator: string;
	encoding: string;
	had_bom: boolean;
}

export interface ColumnInfo {
	name: string;
	dtype: string;
}

export interface Dataset {
	id: string;
	source: string;
	format: SourceFormat;
	csv?: CsvDialect;
	rows: number;
	columns: ColumnInfo[];
	ingest_version: number;
}

// ------------------------------------------------------ tempera-data · profile

export type ColumnKind =
	'numeric' | 'boolean' | 'categorical' | 'datetime' | 'text' | 'high_cardinality' | 'constant';

export interface NumericStats {
	min: number;
	max: number;
	mean: number;
	std: number;
	p01: number;
	p25: number;
	p50: number;
	p75: number;
	p99: number;
	/** Equal-width bins over [min, max]. */
	histogram: number[];
}

export interface ValueCount {
	value: string;
	count: number;
}

export interface ColumnProfile {
	name: string;
	dtype: string;
	kind: ColumnKind;
	null_count: number;
	null_fraction: number;
	n_unique: number;
	unique_ratio: number;
	numeric?: NumericStats;
	top_values?: ValueCount[];
	mean_length?: number;
	min_time?: string;
	max_time?: string;
}

export interface Profile {
	version: number;
	dataset_id: string;
	rows: number;
	columns: ColumnProfile[];
	sample: Record<string, unknown>[];
	sampled_rows?: number;
}

export interface DatasetResponse {
	dataset: Dataset;
	profile: Profile;
}

// -------------------------------------------------------- tempera-data · infer

export interface TaskInference {
	task: TaskType;
	metric: Metric;
	reasons: string[];
	minority_share?: number;
}

// -------------------------------------------------------- tempera-data · split

export type FeatureKind = 'numeric' | 'categorical' | 'boolean';

export interface Feature {
	name: string;
	kind: FeatureKind;
}

export interface FeaturePlan {
	features: Feature[];
	datetime: string[];
	dropped: Record<string, string>;
	dedupe: boolean;
}

// --------------------------------------------------------- tempera-data · eval

export interface Evaluation {
	rows: number;
	metrics: Record<string, number>;
	confusion?: number[][];
	classes?: string[];
	roc?: [number, number][];
	pr?: [number, number][];
	residuals?: [number, number][];
}

// --------------------------------------------------------------- tempera-audit

export type Severity = 'green' | 'yellow' | 'red';

export type Resolution =
	{ status: 'unresolved' } | { status: 'auto'; action: string } | { status: 'acknowledged' };

export interface Finding {
	id: string;
	check: string;
	severity: Severity;
	column?: string;
	title: string;
	evidence: string;
	suggestion: string;
	resolution: Resolution;
}

export interface AuditReport {
	dataset_id: string;
	policy: AuditPolicy;
	findings: Finding[];
	plan: FeaturePlan;
	blocked: boolean;
}

export interface AuditResponse {
	inference: TaskInference;
	audit: AuditReport;
}

// --------------------------------------------------------------- tempera-track

export type JobStatus =
	'created' | 'running' | 'succeeded' | 'failed' | 'blocked' | 'cancelled' | (string & {});

export interface JobRow {
	id: string;
	name: string;
	status: JobStatus;
	task: string | null;
	metric: string | null;
	dataset_id: string | null;
	winner_run_id: string | null;
	report_path: string | null;
	error: string | null;
	created_at: string;
	finished_at: string | null;
}

export interface RunRow {
	id: string;
	job_id: string;
	algorithm: string;
	backend: string;
	is_baseline: boolean;
	status: string;
	valid_score: number | null;
	valid_metrics: Record<string, number>;
	model_dir: string | null;
	error: string | null;
	duration_s: number | null;
}

export interface MetricPoint {
	name: string;
	step: number | null;
	value: number;
	unix_ms: number;
}

export interface JobDetail {
	job: JobRow;
	runs: RunRow[];
}

// --------------------------------------------------------------------- server

export interface Algorithm {
	key: string;
	backend: string;
	tasks: string[];
}

export interface Gpu {
	name: string;
	vram_bytes: number;
}

export interface Hardware {
	os: string;
	arch: string;
	cpu_cores: number;
	ram_bytes: number;
	gpus: Gpu[];
	/** Whether a Python environment for the workers was found. */
	python_available?: boolean;
}

export interface Machine {
	id: string;
	kind: string;
	hardware: Hardware;
	status: string;
}

export interface SpecIssue {
	path: string;
	message: string;
}

export interface ValidateResponse {
	valid: boolean;
	issues: SpecIssue[];
}
