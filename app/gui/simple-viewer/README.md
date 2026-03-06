# FastSurfer Simple Viewer

Svelte + Vite frontend for selecting medical image inputs, launching FastSurfer processing, and previewing processed outputs.

Shared coding-agent playbook: [../../SKILLS.md](../../SKILLS.md)

## Development

- Install dependencies: `npm install`
- Run dev server: `npm run dev`
- Build production bundle: `npm run build`
- Preview build: `npm run preview`

## Testing

All testing files and generated artifacts are grouped under `testing/`.

### Folder Layout

- `testing/e2e/`
	- Playwright configs: `playwright.config.ts`, `playwright.ci.config.ts`
	- E2E tests: `*.spec.ts`
	- Playwright outputs: `testing/e2e/results/`
- `testing/vitest/`
	- Vitest config: `vitest.config.ts`
	- Vitest setup + tests: `setup.ts`, `*.test.ts`
	- Vitest outputs: `testing/vitest/results/`

### Commands

- Unit/component tests (Vitest): `npm run test`
- Unit/component tests in watch mode: `npm run test:watch`
- E2E tests (Playwright): `npm run test:e2e`
- E2E tests for CI (headless): `npm run test:e2e:ci`

### Notes

- Playwright browsers can be installed with: `npx playwright install chromium`
- Local Playwright HTML reports are written to `testing/e2e/results/playwright-report`
- Vitest JSON report is written to `testing/vitest/results/vitest-report.json`
