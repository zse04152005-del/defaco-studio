import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// API address: DEFACO_API_PORT picks another backend (e.g. one started on the paper-trading data copy)
const api = `127.0.0.1:${process.env.DEFACO_API_PORT ?? "8000"}`;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": { target: `http://${api}`, changeOrigin: true },
      "/ws": { target: `ws://${api}`, ws: true },
    },
  },
});
