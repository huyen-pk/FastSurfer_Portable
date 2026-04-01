# Simple Viewer Components

This guide explains how to reuse existing Svelte components and how to add new ones without drifting away from the implemented project structure or the Damadian Lab design system defined in [../../specs/ui/aetheris_medical/DESIGN.md](../../specs/ui/aetheris_medical/DESIGN.md).

## Design Source of Truth

When creating or editing components, follow these sources in order:

1. [../../specs/ui/aetheris_medical/DESIGN.md](../../specs/ui/aetheris_medical/DESIGN.md) for visual rules and interaction tone.
2. `index.html` for the Tailwind color, font, and radius tokens available to the app.
3. `src/app.css` for project-wide helper classes such as `.glass-panel`, `.backlit-button`, `.pipeline-connector`, and `.custom-scrollbar`.

Do not introduce a separate styling system, a second token source, or page-local color constants when an existing theme token already expresses the intent.

## Folder Responsibilities

Use the existing folder split before creating another category:

- `src/components/layout/`: global chrome, navigation, footer, and shared layout contracts.
- `src/components/pages/`: high-level page compositions that assemble a full screen from lower-level components.
- `src/components/shared/`: reusable visual primitives that work across multiple pages.
- `src/components/dashboard/`: dashboard-only components.
- `src/components/scan-viewer/`: scan viewer-only components.
- `src/components/pipelines/`: pipeline-only components.
- `src/components/analytics/`: analytics-only components.

Place a component in the narrowest folder that still matches its reuse scope. If a component is useful on more than one page, move it to `shared` rather than duplicating it.

## Reuse Rules

Before adding a new component:

1. Check whether the change is only a new configuration of an existing component.
2. Prefer extending an existing presentational component with typed props or a slot when the visual pattern is already present.
3. Keep page-level data arrays and composition logic in `src/components/pages/` unless multiple pages genuinely need the same data helper.
4. Avoid copying markup between pages just to change text, icon names, or accent classes.

Current reuse patterns already in the app:

- `PageSectionHeader.svelte` handles page eyebrow, heading, and descriptive copy.
- `InsightMediaCard.svelte` supports either prop-driven overlays or a custom slot for richer content.
- Layout chrome uses typed view contracts from `src/components/layout/types.ts`.

## How to Create a New Component

Use this decision path:

1. Start from the page that needs the new UI.
2. Extract only when a markup block has a stable purpose, a repeatable visual pattern, or enough props to justify reuse.
3. Define a small typed prop surface in `script lang="ts"`.
4. Keep the component presentational unless it truly owns interaction state.
5. Keep page-switching, page-specific demo data, and screen composition in page components.

Good candidates for extraction:

- Repeated cards with the same surface treatment.
- Repeated table, metric, or media blocks.
- Shared headers and page framing.
- Domain-specific blocks reused within one feature area.

Poor candidates for extraction:

- One-off wrappers used once with no reusable meaning.
- Components that exist only to rename a `div`.
- Components that duplicate an existing shared primitive with minor copy changes.

## Styling Rules for New Components

The implemented UI follows the Damadian Lab direction. New components should keep the same visual logic.

### Surface and Separation

- Follow the no-line rule: prefer surface shifts and spacing over visible 1px separators.
- Use tonal layering with the existing Tailwind tokens such as `bg-surface`, `bg-surface-container`, `bg-surface-container-high`, and `bg-surface-container-lowest`.
- Only use faint outline treatments when accessibility truly needs more separation, and keep them subtle with `outline-variant` opacity.

### Typography

- Use `font-headline` for large editorial headings and `font-body` for body copy and dense interface text.
- Keep metadata visually quieter with smaller sizes and reduced contrast, similar to existing `text-primary/60` or `text-on-surface-variant` usage.
- Do not use pure white. Stay within the existing `on-*` token palette.

### Accent Usage

- Reserve `tertiary` for key actions, live states, and important progress indicators.
- Use `primary` and `secondary` for lower-intensity emphasis.
- Avoid saturating the screen with cyan; it should remain the focal accent.

### Depth and Effects

- Prefer stacked surfaces over obvious drop shadows.
- Use `.glass-panel` for floating overlays and temporary translucent surfaces.
- Use `.backlit-button` for primary CTA styling when the component needs the signature illuminated treatment.
- Imaging containers should bias toward `surface-container-lowest` for maximum scan contrast.

### Radius and Motion

- Stay within the configured radius scale already used by the app.
- Keep transitions subtle and purposeful, matching the existing hover and transform patterns.
- Avoid decorative animation that does not reinforce hierarchy or state.

## Composition Patterns

Keep the layering consistent with the current app:

- `App.svelte` should remain a thin shell that owns page selection and composes layout plus one active page.
- Components in `pages/` should orchestrate screen sections and provide data to lower-level components.
- Components in feature folders should focus on rendering a specific block cleanly from props.
- Components in `shared/` should be generic enough to survive reuse on future pages without bringing domain-specific assumptions.

If a component needs the active view type or footer/navigation contracts, import them from `src/components/layout/types.ts` instead of redefining them.

## Practical Checklist Before Merging a New Component

- The component lives in the correct folder for its reuse scope.
- Props are typed and named by intent rather than raw styling details where possible.
- The component uses existing Tailwind tokens from `index.html`.
- The component respects the no-line rule and tonal layering from the design spec.
- Shared helpers from `src/app.css` are reused when appropriate.
- Page-level composition remains in `src/components/pages/`.
- Validation still passes from `app/gui/damadian-ui`:
  - `npm run test`
  - `./node_modules/.bin/tsc --noEmit --project tsconfig.json`
  - `npm run build`

## When to Refactor Existing Components

Refactor instead of adding another component when:

- A component in `shared/` can absorb the new use case through a small prop or slot extension.
- Two feature-specific components only differ in content and accent tokens.
- A page component has grown large because one section has become independently meaningful and reusable.

If a refactor would make a component less focused or introduce many conditional branches, keep the specialized component separate inside its domain folder.