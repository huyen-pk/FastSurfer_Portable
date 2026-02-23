import { expect, test } from "@playwright/test";

test.describe("fastsurfer_simple_viewer_behavior", () => {
  test("open_page_should_render_primary_controls", async ({ page }) => {
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "FastSurfer Processing" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Pick Files/Folders" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Start Processing" })).toBeDisabled();
  });

  test("start_processing_should_render_processed_file_from_mocked_backend", async ({ page }) => {
    await page.route("**/process/run", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          ackMessage: "Processing completed.",
          results: [
            {
              inputPath: "subject01.nii.gz",
              outputPath: "/tmp/subject01.mgz",
              outputFilename: "subject01.mgz",
              runResult: "ok"
            }
          ]
        })
      });
    });

    await page.goto("/");

    await page.locator('[data-testid="file-picker-input"]').setInputFiles({
      name: "subject01.nii.gz",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("dummy")
    });

    await expect(page.getByTitle("subject01.nii.gz")).toBeVisible();

    await page.getByRole("button", { name: "Start Processing" }).click();

    await expect(page.getByText("Processing completed.")).toBeVisible();
    await expect(page.getByRole("button", { name: "subject01.nii.gz" })).toBeVisible();
    await expect(page.getByTitle("/tmp/subject01.mgz")).toBeVisible();
  });
});