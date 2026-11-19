// State of the guided "new job" flow (docs/wiki/User-Journey.md). The YAML spec is the single
// source of truth: every step edits it and the audit/validation always run on the exact text.

import { api, ApiError } from '#lib/api.js';
import { setIn, toggleAcknowledge, toggleDrop, YamlEditError } from '#lib/spec-yaml.js';
import type {
	Algorithm,
	AuditReport,
	Dataset,
	Machine,
	Profile,
	TaskInference,
	ValidateResponse
} from '#lib/types.js';

export const STEPS = [
	{ key: 'data', label: 'Data', hint: 'Upload a file' },
	{ key: 'goal', label: 'Goal', hint: 'What to predict' },
	{ key: 'audit', label: 'Audit', hint: 'Check the data' },
	{ key: 'strategy', label: 'Strategy', hint: 'Algorithms & split' },
	{ key: 'machine', label: 'Machine', hint: 'Where to run' },
	{ key: 'review', label: 'Review', hint: 'Spec & train' }
] as const;

export type StepKey = (typeof STEPS)[number]['key'];

function message(e: unknown): string {
	if (e instanceof ApiError || e instanceof YamlEditError || e instanceof Error) return e.message;
	return String(e);
}

export function defaultJobName(source: string): string {
	const file = source.split(/[\\/]/).pop() ?? 'job';
	const stem = file.replace(/\.[^.]+$/, '') || 'job';
	return stem.replace(/[^A-Za-z0-9_-]/g, '-').toLowerCase();
}

export class Journey {
	step: StepKey = $state('data');

	dataset: Dataset | null = $state.raw(null);
	profile: Profile | null = $state.raw(null);
	uploading = $state(false);
	uploadError: string | null = $state(null);

	/** `undefined` = not chosen yet, `null` = no target (clustering). */
	target: string | null | undefined = $state(undefined);
	name = $state('');
	yaml = $state('');
	drafting = $state(false);
	goalError: string | null = $state(null);

	inference: TaskInference | null = $state.raw(null);
	audit: AuditReport | null = $state.raw(null);
	auditing = $state(false);
	auditError: string | null = $state(null);
	/** YAML text the current `audit` was computed from. */
	auditedYaml = $state('');

	algorithms: Algorithm[] | null = $state.raw(null);
	machines: Machine[] | null = $state.raw(null);
	machineId: string | null = $state(null);

	validation: ValidateResponse | null = $state.raw(null);
	validating = $state(false);

	submitting = $state(false);
	submitError: string | null = $state(null);

	private auditSeq = 0;
	private validateSeq = 0;

	get blocked(): boolean {
		return this.audit?.blocked ?? false;
	}

	get auditStale(): boolean {
		return this.audit !== null && this.auditedYaml !== this.yaml;
	}

	/** Whether the user may jump to `step` from the stepper. */
	reachable(step: StepKey): boolean {
		switch (step) {
			case 'data':
				return true;
			case 'goal':
				return this.profile !== null;
			case 'audit':
				return this.audit !== null;
			case 'strategy':
			case 'machine':
			case 'review':
				return this.audit !== null && !this.blocked;
		}
	}

	go(step: StepKey): void {
		if (this.reachable(step)) {
			this.step = step;
			if (typeof window !== 'undefined') window.scrollTo({ top: 0, behavior: 'smooth' });
		}
	}

	async upload(file: File): Promise<void> {
		this.uploading = true;
		this.uploadError = null;
		try {
			const res = await api.uploadDataset(file);
			this.dataset = res.dataset;
			this.profile = res.profile;
			this.name = defaultJobName(file.name || res.dataset.source);
			// A new dataset invalidates everything downstream.
			this.target = undefined;
			this.yaml = '';
			this.audit = null;
			this.inference = null;
			this.validation = null;
		} catch (e) {
			this.uploadError = message(e);
		} finally {
			this.uploading = false;
		}
	}

	/** Step 2: draft the spec for `target` (null = clustering), then audit it. */
	async chooseTarget(target: string | null): Promise<void> {
		if (!this.dataset) return;
		this.target = target;
		this.drafting = true;
		this.goalError = null;
		try {
			const { yaml } = await api.draftSpec({
				dataset_id: this.dataset.id,
				...(target ? { target } : {}),
				...(this.name ? { name: this.name } : {})
			});
			this.yaml = yaml;
			await this.runAudit();
		} catch (e) {
			this.goalError = message(e);
		} finally {
			this.drafting = false;
		}
	}

	/** Applies a YAML edit; returns false (and records the error) when the YAML cannot be edited. */
	edit(fn: (yaml: string) => string): boolean {
		try {
			this.yaml = fn(this.yaml);
			return true;
		} catch (e) {
			this.auditError = message(e);
			return false;
		}
	}

	set(path: (string | number)[], value: unknown): boolean {
		return this.edit((y) => setIn(y, path, value));
	}

	async runAudit(): Promise<void> {
		const seq = ++this.auditSeq;
		const yaml = this.yaml;
		this.auditing = true;
		this.auditError = null;
		try {
			const res = await api.audit(yaml);
			if (seq !== this.auditSeq) return;
			this.inference = res.inference;
			this.audit = res.audit;
			this.auditedYaml = yaml;
		} catch (e) {
			if (seq !== this.auditSeq) return;
			this.auditError = message(e);
		} finally {
			if (seq === this.auditSeq) this.auditing = false;
		}
	}

	async setDrop(column: string, on: boolean): Promise<void> {
		if (this.edit((y) => toggleDrop(y, column, on))) await this.runAudit();
	}

	async setAcknowledged(id: string, on: boolean): Promise<void> {
		if (this.edit((y) => toggleAcknowledge(y, id, on))) await this.runAudit();
	}

	async loadCatalog(): Promise<void> {
		const [algos, machines] = await Promise.allSettled([api.algorithms(), api.machines()]);
		if (algos.status === 'fulfilled') this.algorithms = algos.value;
		if (machines.status === 'fulfilled') {
			this.machines = machines.value;
			if (!this.machineId && machines.value.length > 0) this.machineId = machines.value[0].id;
		}
	}

	async validate(): Promise<ValidateResponse | null> {
		const seq = ++this.validateSeq;
		this.validating = true;
		try {
			const res = await api.validateSpec(this.yaml);
			if (seq === this.validateSeq) this.validation = res;
			return res;
		} catch (e) {
			const res: ValidateResponse = {
				valid: false,
				issues:
					e instanceof ApiError && e.issues.length ? e.issues : [{ path: '', message: message(e) }]
			};
			if (seq === this.validateSeq) this.validation = res;
			return res;
		} finally {
			if (seq === this.validateSeq) this.validating = false;
		}
	}

	/** Step 7: validate, re-audit if the YAML changed, then start the job. Returns the job id. */
	async train(): Promise<string | null> {
		this.submitting = true;
		this.submitError = null;
		try {
			const v = await this.validate();
			if (!v?.valid) {
				this.submitError = 'The spec has errors; fix them before training.';
				return null;
			}
			if (this.auditStale || !this.audit) await this.runAudit();
			if (this.auditError) {
				this.submitError = this.auditError;
				return null;
			}
			if (this.blocked) {
				this.submitError = 'The audit blocks training: resolve the red findings first.';
				this.step = 'audit';
				return null;
			}
			const { job_id } = await api.createJob(this.yaml);
			return job_id;
		} catch (e) {
			this.submitError = message(e);
			return null;
		} finally {
			this.submitting = false;
		}
	}
}
