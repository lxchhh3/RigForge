import { test, expect } from '@playwright/test'

const SAMPLE_RUN = {
  id: 'classic_chic',
  schema_version: 1,
  generated_at: '2026-05-21T06:20:59+00:00',
  input: {
    donor_id: 'maya',
    target_id: 'maya',
    donor_score: 1.0,
    candidates: [{ id: 'maya', score: 1.0 }],
  },
  output_fbx: '/tmp/out.fbx',
  cache: { key: 'k1', hit: false },
  edit_plan: { drops: [], renames: {}, n_drops: 0, n_renames: 0 },
  warnings: [],
  decisions_summary: { kept: 105, dropped: 73, llm_model_id: null },
  notes: [],
}

async function mockAssembliesApi(page: import('@playwright/test').Page, body: unknown) {
  await page.route('**/api/assemblies', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    })
  })
}

test('history page renders RigForge title and recent runs panel', async ({ page }) => {
  await mockAssembliesApi(page, [])
  await page.goto('/history')

  await expect(page.getByRole('heading', { name: 'RigForge', level: 1 })).toBeVisible()

  const panel = page.getByTestId('recent-assemblies')
  await expect(panel).toBeVisible()
  await expect(panel.getByRole('heading', { name: /recent assembly runs/i })).toBeVisible()
})

test('empty state shows a clear no recent runs message', async ({ page }) => {
  await mockAssembliesApi(page, [])
  await page.goto('/history')

  const panel = page.getByTestId('recent-assemblies')
  await expect(panel.getByTestId('empty-state')).toHaveText(/no recent runs/i)
})

test('renders assembly runs returned by the API', async ({ page }) => {
  await mockAssembliesApi(page, [SAMPLE_RUN])
  await page.goto('/history')

  const items = page.getByTestId('assembly-item')
  await expect(items).toHaveCount(1)
  await expect(items.first()).toContainText('classic_chic')
  await expect(items.first()).toContainText('kept 105')
  await expect(items.first()).toContainText('dropped 73')
})

test('shows error message when the API is unreachable', async ({ page }) => {
  await page.route('**/api/assemblies', (route) => route.abort('failed'))
  await page.goto('/history')

  await expect(page.getByTestId('error')).toBeVisible()
  await expect(page.getByTestId('error')).toContainText(/api/i)
})
