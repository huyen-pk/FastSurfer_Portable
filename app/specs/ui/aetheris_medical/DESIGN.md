# Design System Strategy: Clinical Precision & Tonal Depth

## 1. Overview & Creative North Star: "The Damadian Lab"
The objective of this design system is to move beyond the cold, sterile "hospital app" aesthetic and toward a sophisticated, editorial high-performance environment. Our Creative North Star is **"The Damadian Lab."** 

In medical contexts, clarity is a matter of life and death, but high data density often leads to cognitive overload. We break the "template" look by using a dark, immersive canvas where information isn't just displayed—it is curated. We avoid rigid, boxed grids in favor of **Tonal Layering**, using depth and light to guide the clinician’s eye. The interface should feel like a premium piece of medical equipment: heavy, precise, and illuminated from within.

---

## 2. Colors & The Surface Philosophy
This palette is designed to minimize eye strain during long shifts while making diagnostic data "pop" against a deep, oceanic background.

### The "No-Line" Rule
**Strict Mandate:** Designers are prohibited from using 1px solid borders to section off the UI. 
Structure must be achieved through background shifts. For example, a diagnostic sidebar using `surface_container_low` sits directly against a `background` workspace. The transition in hex values is the boundary. This creates a seamless, "infinite" feel that reduces visual noise.

### Surface Hierarchy & Nesting
Treat the UI as a series of stacked, semi-polished slabs. 
- **Base Level:** `surface` (#0a151a) for the primary workspace.
- **Mid Level:** `surface_container` (#162127) for secondary navigation or utility panels.
- **Top Level:** `surface_container_high` (#202b32) for active modal elements or high-priority alerts.

### The "Glass & Gradient" Rule
To elevate the "out-of-the-box" look, floating elements (like floating action buttons or temporary overlays) must use **Glassmorphism**. Apply `surface_variant` at 60% opacity with a `20px` backdrop blur. 
*   **Signature Textures:** Main CTAs should not be flat. Apply a subtle linear gradient from `primary` (#a1cced) to `primary_fixed_dim` (#a1cced) at a 45-degree angle to give buttons a slight "backlit" sheen.

---

## 3. Typography: Editorial Authority
We utilize two distinct sans-serif families to balance clinical data with high-end readability.

*   **The Display Choice (Manrope):** Used for `display-` and `headline-` scales. Manrope’s geometric yet warm nature provides an authoritative, modern feel. Use `headline-lg` (2rem) for patient names or primary diagnostic categories to create an editorial focal point.
*   **The Utility Choice (Inter):** Used for `title-`, `body-`, and `label-` scales. Inter is the industry standard for legibility at small sizes. 
*   **Hierarchy Note:** To emphasize data density without clutter, use `label-sm` (0.6875rem) in `secondary_fixed_dim` for metadata (e.g., timestamps, unit measurements), keeping the primary `body-md` text clear for vital signs.

---

## 4. Elevation & Depth: Tonal Layering
In "The Damadian Lab," we do not use traditional shadows. We use **Ambient Light**.

*   **The Layering Principle:** Depth is achieved by "stacking." A `surface_container_highest` (#2b363d) card placed on a `surface` (#0a151a) background creates a natural lift.
*   **Ambient Shadows:** If an element must float (e.g., a critical diagnostic popover), use a highly diffused shadow: `box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4)`. The shadow color should never be pure black, but a deeper tint of the background.
*   **The "Ghost Border" Fallback:** For accessibility in high-contrast situations, use the `outline_variant` (#44474c) at **15% opacity**. It should be felt, not seen.

---

## 5. Components: Precision Primitives

### Buttons & Interaction
*   **Primary:** A "backlit" pill using `tertiary` (#00daf3) for the background and `on_tertiary` (#00363d) for text. High contrast for critical actions (e.g., "Submit Order").
*   **Secondary:** Ghost-style with no background. Use `primary` (#a1cced) for text. On hover, transition the background to `surface_container_high`.
*   **Corner Radius:** All interactive elements use the `md` (0.375rem) or `lg` (0.5rem) scale. Avoid `full` rounding except for status indicators.

### Cards & Medical Imaging Containers
*   **Rule:** Forbid divider lines. Separate "Patient History" from "Current Meds" using a `3.5rem` (`spacing.16`) vertical gap or a subtle shift from `surface_container_low` to `surface_container`.
*   **Imaging Viewport:** Imaging (X-rays, MRIs) must be housed in `surface_container_lowest` (#051015) to provide the highest possible contrast for the blacks and greys of the scan.

### Inputs & Vital Fields
*   **Style:** Minimalist. No bottom border. Use `surface_container_highest` as a solid block background. 
*   **Focus State:** A `tertiary` (#00daf3) glow (2px outer spread) rather than a heavy border.

### Contextual Components for Medical UI
*   **The "Vitals Strip":** A horizontal, non-bordered container using `surface_container_low` that houses high-density data chips for heart rate, SpO2, and BP.
*   **Data Density Chips:** Small, high-contrast labels using `secondary_container` with `on_secondary_container` text for quick categorization (e.g., "STAT," "Urgent," "Stable").

---

## 6. Do’s and Don’ts

### Do:
*   **Use Asymmetry:** Place medical imagery (large) next to dense data tables (small) to create a dynamic, professional layout.
*   **Embrace the Dark:** Keep 90% of the UI in the `surface` and `surface_container` range to protect the user's night vision in dark clinical environments.
*   **Prioritize the Cyan:** Use the `tertiary` (#00daf3) sparingly. It is a laser, not a paint brush. Use it only for the most important interactive "action" on the screen.

### Don't:
*   **Don't use Divider Lines:** If you feel the need for a line, increase the spacing (`spacing.6` or `8`) or change the surface tier.
*   **Don't use Pure White:** `on_surface` is `#d8e4ec` (a soft blue-gray). Pure white (#FFFFFF) will cause "halation" effects against the dark background and fatigue the eye.
*   **Don't use Standard Drop Shadows:** Stick to tonal shifts. If you must use a shadow, ensure it is nearly invisible and extra-wide.