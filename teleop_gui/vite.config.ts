import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import electron from 'vite-plugin-electron/simple';

export default defineConfig(({ command }) => ({
  plugins: [
    react(),
    // The Electron shell is only built for `vite build` / `vite` with an
    // Electron binary present. `vite dev` alone still serves the renderer in a
    // browser, which is the quickest way to review the interface.
    electron({
      main: { entry: 'electron/main.ts' },
      preload: { input: 'electron/preload.ts' },
      renderer: command === 'serve' ? undefined : {},
    }),
  ],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
}));
