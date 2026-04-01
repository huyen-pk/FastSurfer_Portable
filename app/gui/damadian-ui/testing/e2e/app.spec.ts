import { expect, test } from "@playwright/test";

test.describe("fastsurfer_simple_viewer_behavior", () => {
  test("open_page_should_render_primary_controls", async ({ page }) => {
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "Clinical Overview" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Dashboard" })).toBeVisible();
    await expect(page.getByText("Recent MRI Studies")).toBeVisible();
  });

  test("navigation_should_switch_between_new_Damadian_screens", async ({ page }) => {
    test.setTimeout(120000);

    await page.goto("/");

    await page.getByRole("button", { name: "Scan Viewer" }).click();
    await expect(page.getByRole("heading", { name: "Scan Information" })).toBeVisible();
    await expect(page.locator('[data-niivue-root="true"]')).toHaveCount(4);
    await expect(page.locator('[data-testid="niivue-canvas"]')).toHaveCount(4);
    await expect(page.locator('[data-niivue-root="true"][data-niivue-status="ready"]')).toHaveCount(4, { timeout: 30000 });
    await expect(page.getByText("AXIAL")).toBeVisible();
    await expect(page.getByText("SAGITTAL")).toBeVisible();
    await expect(page.getByText("CORONAL")).toBeVisible();
    await expect(page.getByText("3D VOLUME")).toBeVisible();

    await page.getByRole("button", { name: "Pipelines" }).click();
    await expect(page.getByRole("heading", { name: "Automated Pipeline Overview" })).toBeVisible();

    await page.getByRole("button", { name: "Analytics" }).click();
    await expect(page.getByRole("heading", { name: "Comparative Study Analysis" })).toBeVisible();
  });
});