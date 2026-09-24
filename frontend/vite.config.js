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
    // The port the API's default CORS_ORIGINS allows.
    port: 5173,
    strictPort: true,
  },

  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/setupTests.js',
  },
})
