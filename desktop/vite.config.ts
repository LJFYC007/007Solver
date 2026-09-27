import { realpathSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const desktopRoot = realpathSync.native(process.cwd());

export default defineConfig({
    root: desktopRoot,
    plugins: [react()],
    clearScreen: false,
    server: {
        fs: {
            allow: [desktopRoot],
        },
        host: "127.0.0.1",
        port: 1420,
        strictPort: true,
        // The browser build's solver requests go to `007solver --serve`.
        proxy: { "/api": "http://127.0.0.1:8007" },
        watch: {
            ignored: ["**/src-tauri/**"],
        },
    },
    build: {
        target: "es2021",
    },
});
