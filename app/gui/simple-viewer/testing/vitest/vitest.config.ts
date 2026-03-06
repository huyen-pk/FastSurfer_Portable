import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    conditions: ["browser"]
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./testing/vitest/setup.ts"],
    include: ["testing/vitest/**/*.test.ts"],
    globals: true,
    reporters: ["default", "json"],
    outputFile: {
      json: "./testing/vitest/results/vitest-report.json"
    }
  }
});