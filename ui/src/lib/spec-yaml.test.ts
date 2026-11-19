import { describe, expect, it } from 'vitest';
import {
	getIn,
	listAt,
	setIn,
	syntaxError,
	toggleAcknowledge,
	toggleDrop,
	YamlEditError
} from './spec-yaml';

// Same text as `tempera init` / POST /specs/draft.
const DRAFT = `# Têmpera job spec. Reference: docs/wiki/Contracts.md · schema: \`tempera schema\`
version: 1
name: churn
data:
  source: uploads/churn.csv
  target: churned
  # time_column: signup_date   # uncomment if you will predict future rows
  # group_column: customer_id   # rows of the same entity never split across train/test
  # drop: []                    # columns to ignore (e.g. audit findings)
# audit:
#   acknowledge: []             # finding ids you reviewed and accept
split:
  test_size: 0.2
  seed: 42
`;

describe('toggleDrop', () => {
	it('adds data.drop as a flow list and keeps the comments', () => {
		const out = toggleDrop(DRAFT, 'refund_amount', true);
		expect(listAt(out, ['data', 'drop'])).toEqual(['refund_amount']);
		expect(out).toContain('drop: [ refund_amount ]');
		expect(out).toContain('# Têmpera job spec.');
		expect(out).toContain('# time_column: signup_date');
		expect(getIn(out, ['data', 'target'])).toBe('churned');
		// Column-0 comment blocks stay at column 0 (not re-indented under `data:`).
		expect(out).toContain('\n# audit:\n#   acknowledge: []');
		expect(out).toContain('\n  # drop: []');
		expect(out.indexOf('# audit:')).toBeLessThan(out.indexOf('split:'));
	});

	it('appends without duplicates and removes the key when emptied', () => {
		let y = toggleDrop(DRAFT, 'a', true);
		y = toggleDrop(y, 'b', true);
		y = toggleDrop(y, 'a', true);
		expect(listAt(y, ['data', 'drop'])).toEqual(['a', 'b']);
		y = toggleDrop(y, 'a', false);
		expect(listAt(y, ['data', 'drop'])).toEqual(['b']);
		y = toggleDrop(y, 'b', false);
		expect(getIn(y, ['data', 'drop'])).toBeUndefined();
		expect(getIn(y, ['data', 'source'])).toBe('uploads/churn.csv');
	});

	it('is a no-op when the state already matches', () => {
		expect(toggleDrop(DRAFT, 'x', false)).toBe(DRAFT);
	});

	it('works with an existing block-style list', () => {
		const y = DRAFT.replace('  target: churned\n', '  target: churned\n  drop:\n    - a\n');
		const out = toggleDrop(y, 'b', true);
		expect(listAt(out, ['data', 'drop'])).toEqual(['a', 'b']);
	});
});

describe('toggleAcknowledge', () => {
	it('creates the audit section and prunes it when empty', () => {
		let y = toggleAcknowledge(DRAFT, 'A1:refund_amount', true);
		expect(listAt(y, ['audit', 'acknowledge'])).toEqual(['A1:refund_amount']);
		y = toggleAcknowledge(y, 'A5', true);
		expect(listAt(y, ['audit', 'acknowledge'])).toEqual(['A1:refund_amount', 'A5']);
		y = toggleAcknowledge(y, 'A1:refund_amount', false);
		y = toggleAcknowledge(y, 'A5', false);
		expect(getIn(y, ['audit'])).toBeUndefined();
	});

	it('keeps other audit settings when the list is emptied', () => {
		const y = DRAFT + 'audit:\n  policy: warn_only\n  acknowledge: [A5]\n';
		const out = toggleAcknowledge(y, 'A5', false);
		expect(getIn(out, ['audit', 'policy'])).toBe('warn_only');
		expect(getIn(out, ['audit', 'acknowledge'])).toBeUndefined();
	});
});

describe('setIn', () => {
	it('sets nested scalars and objects, creating parents', () => {
		let y = setIn(DRAFT, ['split', 'test_size'], 0.25);
		y = setIn(y, ['budget', 'max_cost'], { amount: 5, currency: 'USD' });
		y = setIn(y, ['strategy', 'candidates'], ['lightgbm', 'xgboost']);
		expect(getIn(y, ['split', 'test_size'])).toBe(0.25);
		expect(getIn(y, ['budget', 'max_cost'])).toEqual({ amount: 5, currency: 'USD' });
		expect(getIn(y, ['strategy', 'candidates'])).toEqual(['lightgbm', 'xgboost']);
		expect(getIn(y, ['split', 'seed'])).toBe(42);
	});

	it('removes keys set to empty values', () => {
		let y = setIn(DRAFT, ['budget', 'max_time'], '2h');
		expect(getIn(y, ['budget', 'max_time'])).toBe('2h');
		y = setIn(y, ['budget', 'max_time'], null);
		expect(getIn(y, ['budget'])).toBeUndefined();
	});

	it('throws a YamlEditError on invalid YAML and getIn returns undefined', () => {
		const bad = 'data: [unclosed\n';
		expect(() => setIn(bad, ['name'], 'x')).toThrow(YamlEditError);
		expect(getIn(bad, ['data'])).toBeUndefined();
		expect(syntaxError(bad)).toBeTruthy();
		expect(syntaxError(DRAFT)).toBeNull();
	});

	it('refuses to treat a scalar as a list', () => {
		const y = DRAFT.replace('  target: churned\n', '  target: churned\n  drop: oops\n');
		expect(() => toggleDrop(y, 'a', true)).toThrow(/must be a list/);
	});
});
