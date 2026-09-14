import { test, expect } from '@playwright/test';

test('@responsive preserves layout and captures a representative screenshot', async ({ page }, testInfo) => {
  await page.goto('/');
  const expectedWidths = { 'responsive-desktop': 1440, 'responsive-laptop': 1280, 'responsive-tablet': 768, 'responsive-mobile': 390 };
  expect(await page.evaluate(() => window.innerWidth)).toBe(expectedWidths[testInfo.project.name]);
  await expect(page.getByRole('heading', { name: /Predicting whether a drug/i })).toBeVisible();
  const horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  expect(horizontalOverflow).toBe(false);
  await expect(page.locator('header')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Try a prediction →' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('home.png'), fullPage: true });
});
