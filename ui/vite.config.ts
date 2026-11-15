import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// Served by the daemon at /ui/ (embedded in the binary). `npm run dev`
// proxies the APIs to a daemon on the default port.
export default defineConfig({
  base: '/ui/',
  plugins: [svelte()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:11435',
      '/v1': 'http://127.0.0.1:11435',
    },
  },
})
