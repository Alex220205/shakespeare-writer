/**
 * Vite configuration: dev server, build, and the Vitest environment.
 *
 * One file configures both the build and the test run, because Vitest reads
 * Vite's config. Anything the app can import, a test can import.
 */

import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  server: {
    // Not Vite's default 5173, which other projects on the same machine are
    // likely to be using. The API's default CORS_ORIGINS allows this port;
    // strictPort makes a clash an error rather than a silent move to a port
    // the API would then refuse.
    port: 5180,
    strictPort: true,
  },

  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/setupTests.js',
  },
})
