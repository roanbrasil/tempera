import { defineConfig } from 'vitest/config';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { mockApi } from './mock/plugin.ts';

// `npm run dev`       → proxies /api to a running `tempera up` (http://127.0.0.1:7070).
// `npm run dev:mock`  → serves a simulated API + WebSocket stream from ./mock (no Rust server).
const API_TARGET = process.env.TEMPERA_API ?? 'http://127.0.0.1:7070';

export default defineConfig(({ mode }) => {
	const mock = mode === 'mock';
	return {
		plugins: [
			sveltekit({
				compilerOptions: {
					// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
					runes: ({ filename }) =>
						filename.split(/[/\\]/).includes('node_modules') ? undefined : true
				},
				// Static SPA: the Rust binary embeds `build/` and serves index.html for unknown paths.
				adapter: adapter({ pages: 'build', assets: 'build', fallback: 'index.html' })
			}),
			mock && mockApi()
		],
		server: {
			proxy: mock ? undefined : { '/api': { target: API_TARGET, changeOrigin: true, ws: true } }
		},
		test: {
			expect: { requireAssertions: true },
			projects: [
				{
					extends: './vite.config.ts',
					test: {
						name: 'unit',
						environment: 'node',
						include: ['src/**/*.{test,spec}.{js,ts}'],
						exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
					}
				}
			]
		}
	};
});
