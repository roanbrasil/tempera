// Comment-preserving edits of a `tempera.yaml` document (docs/wiki/Contracts.md §1).
//
// The YAML text is the single source of truth in the journey: every control edits the text
// through these helpers, so whatever the user sees in the editor is exactly what gets validated
// and submitted.

import { isMap, isScalar, isSeq, parseDocument, type Document } from 'yaml';

export type YamlPath = readonly (string | number)[];

export class YamlEditError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'YamlEditError';
	}
}

function parse(text: string): Document {
	const doc = parseDocument(text, { keepSourceTokens: false });
	if (doc.errors.length > 0) throw new YamlEditError(doc.errors[0].message);
	reattachTopLevelComments(doc, text);
	return doc;
}

/**
 * The parser attaches every comment line after the last entry of a nested map to that map, even
 * the ones written at column 0 that introduce the next top-level key (e.g. the commented-out
 * `# audit:` block of a drafted spec). Re-printing would then indent them under the map. Move
 * such column-0 lines back in front of the next top-level key (or to the end of the document).
 */
function reattachTopLevelComments(doc: Document, text: string): void {
	const top = doc.contents;
	if (!isMap(top)) return;
	const col0 = new Set(
		text
			.split(/\r?\n/)
			.filter((l) => l.startsWith('#'))
			.map((l) => l.slice(1))
	);
	top.items.forEach((pair, i) => {
		const val = pair.value;
		if (!isMap(val) && !isSeq(val)) return;
		if (!val.comment) return;
		const lines = val.comment.split('\n');
		const cut = lines.findIndex((l) => col0.has(l));
		if (cut === -1) return;
		const moved = lines.slice(cut).join('\n');
		const keep = lines.slice(0, cut).join('\n');
		val.comment = keep || null;
		const next = top.items[i + 1];
		if (next && isScalar(next.key)) {
			next.key.commentBefore = next.key.commentBefore
				? `${moved}\n${next.key.commentBefore}`
				: moved;
		} else {
			doc.comment = doc.comment ? `${moved}\n${doc.comment}` : moved;
		}
	});
}

function isEmpty(value: unknown): boolean {
	return (
		value === undefined ||
		value === null ||
		value === '' ||
		(Array.isArray(value) && value.length === 0)
	);
}

/** Value at `path` as plain JS, or `undefined`. Invalid YAML yields `undefined`. */
export function getIn(text: string, path: YamlPath): unknown {
	try {
		const v = parse(text).getIn(path, false);
		return v && typeof v === 'object' && 'toJSON' in v ? (v as { toJSON(): unknown }).toJSON() : v;
	} catch {
		return undefined;
	}
}

/** Whole document as plain JS (`{}` when empty or invalid). */
export function toObject(text: string): Record<string, unknown> {
	try {
		const v = parse(text).toJS() as unknown;
		return v && typeof v === 'object' ? (v as Record<string, unknown>) : {};
	} catch {
		return {};
	}
}

/** Removes maps left empty by a deletion (e.g. `audit: {}`), walking up `path`. */
function pruneEmpty(doc: Document, path: YamlPath): void {
	for (let i = path.length - 1; i > 0; i--) {
		const parent = path.slice(0, i);
		const node = doc.getIn(parent, true);
		if (isMap(node) && node.items.length === 0 && !node.commentBefore && !node.comment)
			doc.deleteIn(parent);
		else break;
	}
}

/**
 * Sets `path` to `value` (creating parent maps as needed). An empty value (`undefined`, `null`,
 * `''`, `[]`) removes the key instead, as the server omits empty optional fields.
 */
export function setIn(text: string, path: YamlPath, value: unknown): string {
	const doc = parse(text);
	if (isEmpty(value)) {
		if (doc.hasIn(path)) {
			doc.deleteIn(path);
			pruneEmpty(doc, path);
		}
	} else {
		const flow = Array.isArray(value) && value.every((v) => typeof v !== 'object');
		doc.setIn(path, doc.createNode(value, { flow }));
	}
	return doc.toString({ lineWidth: 0 });
}

/** Scalar strings of the sequence at `path` (`[]` when absent or not a list). */
export function listAt(text: string, path: YamlPath): string[] {
	const v = getIn(text, path);
	return Array.isArray(v) ? v.map((x) => String(x)) : [];
}

/** Adds (`on`) or removes `item` from the list at `path`, keeping order and no duplicates. */
export function toggleListItem(text: string, path: YamlPath, item: string, on: boolean): string {
	const doc = parse(text);
	const node = doc.getIn(path, true);
	if (node !== undefined && node !== null && !isSeq(node))
		throw new YamlEditError(`\`${path.join('.')}\` must be a list`);
	const current = listAt(text, path);
	const has = current.includes(item);
	if (on === has) return text;
	const next = on ? [...current, item] : current.filter((x) => x !== item);
	return setIn(text, path, next);
}

export const DROP_PATH = ['data', 'drop'] as const;
export const ACK_PATH = ['audit', 'acknowledge'] as const;

/** `data.drop`: columns the job ignores (audit decision). */
export function toggleDrop(text: string, column: string, on: boolean): string {
	return toggleListItem(text, DROP_PATH, column, on);
}

/** `audit.acknowledge`: finding ids the user reviewed and accepts. */
export function toggleAcknowledge(text: string, findingId: string, on: boolean): string {
	return toggleListItem(text, ACK_PATH, findingId, on);
}

/** Checks the text parses as YAML; returns the first syntax error or null. */
export function syntaxError(text: string): string | null {
	const doc = parseDocument(text);
	return doc.errors.length > 0 ? doc.errors[0].message : null;
}
