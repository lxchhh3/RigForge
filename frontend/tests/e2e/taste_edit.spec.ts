import { test, expect } from '@playwright/test'

// Minimal self-contained mocks for the "Cut sole height" (taste-edit) flow.
const AVATARS = [{ id: 'maya', display_name: 'Maya v1.02.2', canonical_roles: ['Hips'] }]

// Clothing with a footwear mesh ("Heel") so the panel's default selection
// (footwear-name heuristic) picks it.
const INSPECT = {
  donor_id: 'maya',
  donor_score: 0.96,
  total_bones: 3,
  bones: [
    { model_id: 1, name: 'Hips', type_class: 'LimbNode', parent_id: null, subtree_size: 1, cluster_weight_count: 100, deforms_meshes: ['Body'] },
    { model_id: 10, name: 'Body', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 11, name: 'Heel', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
  ],
  blend_shape_channels: [],
}

// 1x1 PNG data URL — enough for the <img> to render in the test.
const PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

const TASTE_RESULT = {
  edited_path: 'C:/fixtures/clothing__Heel_cut.fbx',
  mesh: 'Heel',
  verts_before: 11008,
  verts_after: 10488,
  verts_removed: 520,
  weights_ok: true,
  preview_before: PNG,
  preview_after: PNG,
}

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

function jsonFulfill(route: import('@playwright/test').Route, body: unknown) {
  if (route.request().method() === 'OPTIONS') {
    return route.fulfill({ status: 204, headers: CORS, body: '' })
  }
  return route.fulfill({
    status: 200, contentType: 'application/json', headers: CORS, body: JSON.stringify(body),
  })
}

async function mockBase(page: import('@playwright/test').Page) {
  await page.route('**/api/avatars', (r) => jsonFulfill(r, AVATARS))
  await page.route('**/api/avatars/*/inspect', (r) => jsonFulfill(r, INSPECT))
  await page.route('**/api/clothings/inspect', (r) => jsonFulfill(r, INSPECT))
  await page.route('**/api/assemblies', (r) => jsonFulfill(r, []))
}

async function addClothing(page: import('@playwright/test').Page) {
  await page.goto('/')
  await page.getByTestId('clothing-path-input').fill('C:/fixtures/clothing.fbx')
  await page.getByTestId('clothing-add-button').click()
  const item = page.getByTestId('clothing-item').first()
  await expect(item).toBeVisible()
  return item
}

test('cut sole height: preview shows before/after, apply marks the clothing edited', async ({ page }) => {
  await mockBase(page)
  const reqs: any[] = []
  await page.route('**/api/clothings/taste-edit', (route) => {
    if (route.request().method() === 'OPTIONS') {
      return route.fulfill({ status: 204, headers: CORS, body: '' })
    }
    reqs.push(route.request().postDataJSON())
    return route.fulfill({
      status: 200, contentType: 'application/json', headers: CORS,
      body: JSON.stringify(TASTE_RESULT),
    })
  })

  const item = await addClothing(page)
  await item.getByTestId('taste-edit-toggle').click()

  // Default mesh = the footwear mesh.
  await expect(item.getByTestId('taste-edit-mesh')).toHaveValue('Heel')

  await item.getByTestId('taste-edit-preview').click()

  const result = item.getByTestId('taste-edit-result')
  await expect(result).toBeVisible()
  await expect(item.getByTestId('taste-edit-before')).toBeVisible()
  await expect(item.getByTestId('taste-edit-after')).toBeVisible()
  await expect(item.getByTestId('taste-edit-weights')).toContainText('intact')

  // The FE sent the right payload.
  expect(reqs.length).toBe(1)
  expect(reqs[0].clothing_path).toBe('C:/fixtures/clothing.fbx')
  expect(reqs[0].mesh).toBe('Heel')
  expect(reqs[0].op).toBe('bisect_cut')

  // Apply adopts the edited clothing.
  await item.getByTestId('taste-edit-apply').click()
  await expect(item.getByTestId('taste-edit-applied')).toBeVisible()
})

test('a cut that strips weights cannot be applied', async ({ page }) => {
  await mockBase(page)
  await page.route('**/api/clothings/taste-edit', (route) => {
    if (route.request().method() === 'OPTIONS') {
      return route.fulfill({ status: 204, headers: CORS, body: '' })
    }
    return route.fulfill({
      status: 200, contentType: 'application/json', headers: CORS,
      body: JSON.stringify({ ...TASTE_RESULT, weights_ok: false }),
    })
  })

  const item = await addClothing(page)
  await item.getByTestId('taste-edit-toggle').click()
  await item.getByTestId('taste-edit-preview').click()

  await expect(item.getByTestId('taste-edit-weights')).toContainText('broke')
  await expect(item.getByTestId('taste-edit-apply')).toBeDisabled()
})
