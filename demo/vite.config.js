import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

// Derive __dirname in ES module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Start the Go server immediately when Vite loads this config
const goServerPath = path.resolve(__dirname, "server.go");
console.log("Starting Go server...");
spawn("go", ["run", goServerPath], {
  stdio: "inherit", // Pipe Go server output to Vite's console
  cwd: __dirname, // Run 'go run' from the demo directory
  shell: true, // Use shell
});

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5432", // Go server address
        changeOrigin: true,
      },
    },
  },
});
