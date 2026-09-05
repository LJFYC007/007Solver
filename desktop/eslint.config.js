import js from "@eslint/js";
import eslintConfigPrettier from "eslint-config-prettier/flat";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";
import { defineConfig, globalIgnores } from "eslint/config";

export default defineConfig([
    globalIgnores(["dist", "node_modules", "src-tauri/gen", "src-tauri/target"]),
    {
        files: ["src/**/*.{ts,tsx}"],
        extends: [
            js.configs.recommended,
            tseslint.configs.recommended,
            reactHooks.configs.flat.recommended,
            reactRefresh.configs.vite,
            eslintConfigPrettier,
        ],
        languageOptions: {
            ecmaVersion: "latest",
            globals: globals.browser,
        },
    },
    {
        files: ["vite.config.ts"],
        extends: [js.configs.recommended, tseslint.configs.recommended, eslintConfigPrettier],
        languageOptions: {
            ecmaVersion: "latest",
            globals: globals.node,
        },
    },
    {
        files: ["eslint.config.js"],
        extends: [js.configs.recommended, eslintConfigPrettier],
        languageOptions: {
            ecmaVersion: "latest",
            globals: globals.node,
        },
    },
]);
