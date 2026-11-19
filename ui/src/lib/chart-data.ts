// Aligns metric curves on a shared x axis (the format uPlot expects).

import type { Series } from './events';

export type Aligned = [number[], ...(number | null)[][]];

/**
 * Builds `[xs, ys1, ys2, ...]` where `xs` is the sorted union of all steps and each `ys` has a
 * value (or `null`) per x. The common case (all curves share the same steps) is a copy-free fast path.
 */
export function alignSeries(curves: Series[]): Aligned {
	if (curves.length === 0) return [[]];
	const first = curves[0].steps;
	const same = curves.every((c) => {
		if (c.steps === first) return true;
		if (c.steps.length !== first.length) return false;
		for (let i = 0; i < first.length; i++) if (c.steps[i] !== first[i]) return false;
		return true;
	});
	if (same) return [first, ...curves.map((c) => c.values)];

	// k-way merge of sorted step arrays.
	const xs: number[] = [];
	const idx = curves.map(() => 0);
	for (;;) {
		let min = Infinity;
		for (let k = 0; k < curves.length; k++) {
			const i = idx[k];
			if (i < curves[k].steps.length && curves[k].steps[i] < min) min = curves[k].steps[i];
		}
		if (min === Infinity) break;
		xs.push(min);
		for (let k = 0; k < curves.length; k++) {
			while (idx[k] < curves[k].steps.length && curves[k].steps[idx[k]] === min) idx[k]++;
		}
	}
	const ys = curves.map((c) => {
		const out: (number | null)[] = new Array(xs.length).fill(null);
		let j = 0;
		for (let i = 0; i < c.steps.length; i++) {
			while (xs[j] < c.steps[i]) j++;
			out[j] = c.values[i];
		}
		return out;
	});
	return [xs, ...ys];
}
