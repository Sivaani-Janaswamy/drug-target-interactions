import { defineConfig, devices } from '@playwright/test';

const frontendPort = 5173;
const backendPort = 8000;

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  webServer: [
    {
      command: 'venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000',
      cwd: '..',
      url: `http://127.0.0.1:${backendPort}/api/health`,
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 5173',
      url: `http://127.0.0.1:${frontendPort}`,
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
      grepInvert: /@responsive/,
    },
    ...[
      ['desktop', 1440, 900],
      ['laptop', 1280, 800],
      ['tablet', 768, 1024],
      ['mobile', 390, 844],
    ].map(([name, width, height]) => ({
      name: `responsive-${name}`,
      use: { ...devices['Desktop Chrome'], viewport: { width, height } },
      grep: /@responsive/,
    })),
  ],
});
