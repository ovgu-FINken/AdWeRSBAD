import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api/get_image': 'http://localhost:8000',  // Your backend URL
      '/api/label_weather': 'http://localhost:8000',
      '/api': 'http://localhost:8000',  // Your backend URL
    },
  },
});

