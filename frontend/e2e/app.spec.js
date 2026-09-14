import { test, expect } from '@playwright/test';

test.describe('DTI-ML browser flows', () => {
  test.setTimeout(120_000);
  test('loads the application and primary navigation works', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', message => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });

    await page.goto('/');
    await expect(page.getByRole('heading', { name: /Predicting whether a drug/i })).toBeVisible();
    await expect(page.locator('header')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Ask about this result' })).toBeVisible();

    for (const [label, heading] of [
      ['How it works', /Five terms this project relies on/i],
      ['Try the predictor', /Try a prediction/i],
      ['Sample result', /No prediction yet/i],
      ['Model & science', /How the models actually compare/i],
      ['About', /About this project/i],
    ]) {
      await page.getByRole('button', { name: label, exact: true }).click();
      await expect(page.getByRole('heading', { name: heading })).toBeVisible();
    }

    expect(consoleErrors).toEqual([]);
  });

  test('invalid predictor input shows an error without replacing the form', async ({ page }) => {
    await page.route('**/api/predict', route => route.fulfill({
      status: 400,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'A protein sequence is required.' }),
    }));
    await page.goto('/');
    await page.getByRole('button', { name: 'Try a prediction →' }).click();
    await page.getByRole('button', { name: 'Predict binding affinity →' }).click();
    await expect(page.getByRole('alert')).toContainText('protein sequence is required');
    await expect(page.locator('textarea#drug')).toBeVisible();
  });

  test('valid preset reaches results through the real FastAPI and ML model', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Try a prediction →' }).click();
    await page.getByRole('button', { name: 'Aspirin', exact: true }).click();
    await expect(page.getByRole('button', { name: /^Random Forest/ })).toHaveClass(/selected/);

    const predictionResponse = page.waitForResponse(response => response.url().endsWith('/api/predict') && response.status() === 200);
    await page.getByRole('button', { name: 'Predict binding affinity →' }).click();
    await predictionResponse;

    await expect(page.getByRole('heading', { name: /Aspirin × ABL1/i })).toBeVisible({ timeout: 120_000 });
    await expect(page.getByText('predicted KIBA score')).toBeVisible();
    await expect(page.locator('.reason')).toHaveCount(4);
  });

  test('chat opens and handles a local deterministic suggested question', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Ask about this result' }).click({ force: true });
    await expect(page.getByRole('heading', { name: 'Ask about this result' })).toBeVisible();
    const chatResponse = page.waitForResponse(response => response.url().endsWith('/api/chat') && response.status() === 200);
    await page.getByRole('button', { name: 'Explain SHAP simply' }).click();
    await chatResponse;
    await expect(page.getByText(/SHAP measures how much each input feature/i)).toBeVisible();
    await page.getByRole('button', { name: 'Close chat' }).click();
    await expect(page.getByRole('heading', { name: 'Ask about this result' })).not.toBeVisible();
  });

  test('frontend handles a simulated server error gracefully', async ({ page }) => {
    await page.route('**/api/predict', route => route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Prediction failed. Please try again.' }),
    }));
    await page.goto('/');
    await page.getByRole('button', { name: 'Try a prediction →' }).click();
    await page.getByRole('button', { name: 'Aspirin', exact: true }).click();
    await page.getByRole('button', { name: 'Predict binding affinity →' }).click();
    await expect(page.getByRole('alert')).toContainText('Prediction failed');
  });
});
