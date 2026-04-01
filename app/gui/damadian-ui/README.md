# FastSurfer Damadian UI

Svelte + Vite frontend for the Damadian Imaging mockup integration used in the FastSurfer app workspace. The current app is a multi-view clinical dashboard shell with dedicated pages for dashboard monitoring, scan viewing, pipeline configuration, and analytics.

Shared coding-agent playbook: [../../SKILLS.md](../../SKILLS.md)

## References

- Component authoring guide: [./COMPONENTS.md](./COMPONENTS.md)
- Styling direction: [../../specs/ui/aetheris_medical/DESIGN.md](../../specs/ui/aetheris_medical/DESIGN.md)

## Project Layout

- `src/App.svelte`: top-level shell that composes layout chrome and switches between high-level pages.
- `src/components/layout/`: app chrome and shared layout types (`AppHeader`, `AppSidebar`, `MobileNav`, `StatusFooter`, `types.ts`).
- `src/components/pages/`: page-level compositions for the four mockup screens.
- `src/components/dashboard/`: dashboard-specific building blocks.
- `src/components/scan-viewer/`: scan viewer-specific building blocks.
- `src/components/pipelines/`: pipeline configuration building blocks.
- `src/components/analytics/`: analytics-specific building blocks.
- `src/components/shared/`: cross-page presentational primitives.
- `src/app.css`: global helpers for recurring visual treatments like glass panels and backlit buttons.
- `index.html`: Tailwind theme tokens, fonts, and Material Symbols configuration.

## Development

From `app/gui/damadian-ui`:

- Install dependencies: `npm install`
- Run dev server: `npm run dev`
- Build production bundle: `npm run build`
- Preview build: `npm run preview`

## Validation

Run these checks from `app/gui/damadian-ui` after UI or component changes:

- Unit/component tests: `npm run test`
- TypeScript validation: `./node_modules/.bin/tsc --noEmit --project tsconfig.json`
- Production build: `npm run build`

## Testing

All testing files and generated artifacts are grouped under `testing/`.

### Folder Layout

- `testing/e2e/`: Playwright configs, browser tests, and generated reports.
- `testing/vitest/`: Vitest config, setup, component tests, and JSON reports.

### Commands

- Unit/component tests (Vitest): `npm run test`
- Unit/component tests in watch mode: `npm run test:watch`
- E2E tests (Playwright): `npm run test:e2e`
- E2E tests for CI (headless): `npm run test:e2e:ci`

### Notes

- Install Playwright browsers with: `npx playwright install chromium`
- Local Playwright HTML reports are written to `testing/e2e/results/playwright-report`
- Vitest JSON report is written to `testing/vitest/results/vitest-report.json`
