import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const input = process.env.INPUT ?? "dashboard-app.html";

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    cssMinify: true,
    emptyOutDir: false,
    minify: true,
    outDir: "dist/ui",
    rollupOptions: { input },
  },
});

