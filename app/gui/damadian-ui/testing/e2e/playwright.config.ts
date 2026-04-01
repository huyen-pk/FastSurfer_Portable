import { defineConfig, devices } from "@playwright/test";

const port = 4173;

export default defineConfig({
  testDir: ".",
  outputDir: "./results/test-results",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [["html", { outputFolder: "./results/playwright-report", open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "on-first-retry",
    headless: !!process.env.CI
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ],
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${port}`,
    port,
    reuseExistingServer: !process.env.CI,
    timeout: 120000
  }
});