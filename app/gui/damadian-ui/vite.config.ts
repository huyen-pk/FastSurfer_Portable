import { defineConfig, loadEnv } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = (env.VITE_FASTSURFER_TARGET || "tauri").toLowerCase();
  const isWebTarget = target === "web";

  return {
    plugins: [
      svelte(),
      VitePWA({
        disable: !isWebTarget,
        registerType: "autoUpdate",
        includeAssets: ["icons/pwa-icon.svg"],
        manifest: {
          name: "FastSurfer Simple Viewer",
          short_name: "FastSurfer",
          description: "FastSurfer simple viewer web app",
          theme_color: "#ffffff",
          background_color: "#ffffff",
          display: "standalone",
          start_url: "/",
          icons: [
            {
              src: "icons/pwa-icon.svg",
              sizes: "any",
              type: "image/svg+xml"
            }
          ]
        }
      })
    ]
  };
});