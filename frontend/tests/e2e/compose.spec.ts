import { test, expect } from '@playwright/test'

const AVATARS = [
  { id: 'maya', display_name: 'Maya v1.02.2', canonical_roles: ['Hips', 'Spine', 'Hand.L', 'Hand.R'] },
  { id: 'moe', display_name: 'Moe', canonical_roles: ['Hips', 'Spine'] },
]

// Clothing inspect — the user sees the Blender outliner view: armature +
// parallel mesh names (Body, Hair, Tie, Skirt). Bones are not shown.
const INSPECT_TREE = {
  donor_id: 'maya',
  donor_score: 0.96,
  total_bones: 7,
  bones: [
    // Armature + bones (backend only)
    { model_id: 1, name: 'Hips', type_class: 'LimbNode', parent_id: null, subtree_size: 3, cluster_weight_count: 100, deforms_meshes: ['Body'] },
    { model_id: 2, name: 'Spine', type_class: 'LimbNode', parent_id: 1, subtree_size: 2, cluster_weight_count: 50, deforms_meshes: ['Body'] },
    { model_id: 3, name: 'Chest', type_class: 'LimbNode', parent_id: 2, subtree_size: 1, cluster_weight_count: 50, deforms_meshes: ['Body'] },
    // Meshes (what the user sees)
    { model_id: 10, name: 'Body', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 11, name: 'Hair', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 12, name: 'Tie', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 13, name: 'Skirt', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
  ],
}

// Target avatar inspect — Maya ships with a bundled outfit. Modder needs
// to strip Cloth/Shoes/Hat before splicing new clothing on.
const TARGET_INSPECT_TREE = {
  donor_id: 'maya',
  donor_score: 1.0,
  total_bones: 7,
  bones: [
    { model_id: 101, name: 'Hips', type_class: 'LimbNode', parent_id: null, subtree_size: 1, cluster_weight_count: 200, deforms_meshes: ['Body'] },
    // Default outfit meshes
    { model_id: 200, name: 'Body', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 201, name: 'Hair', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 202, name: 'Cloth', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 203, name: 'Shoes', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 204, name: 'Hat', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
    { model_id: 205, name: 'Underwear', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [] },
  ],
}

const CORS_HEADERS = {
  // Wildcard origin — Playwright's baseURL is 127.0.0.1, the real FastAPI
  // server allows localhost; tests don't need real-server parity here.
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

function jsonFulfill(route: import('@playwright/test').Route, body: unknown) {
  // Handle CORS preflight (OPTIONS) separately — Playwright's route mock
  // matches all methods at the same URL, so we need to gate on request method.
  if (route.request().method() === 'OPTIONS') {
    return route.fulfill({ status: 204, headers: CORS_HEADERS, body: '' })
  }
  return route.fulfill({
    status: 200,
    contentType: 'application/json',
    headers: CORS_HEADERS,
    body: JSON.stringify(body),
  })
}

async function mockApi(page: import('@playwright/test').Page, opts: {
  avatars?: unknown
  inspect?: unknown
  assemble?: unknown
  assemblies?: unknown
  targetInspect?: unknown
} = {}) {
  await page.route('**/api/avatars', (route) => jsonFulfill(route, opts.avatars ?? AVATARS))
  await page.route('**/api/avatars/*/inspect', (route) =>
    jsonFulfill(route, opts.targetInspect ?? TARGET_INSPECT_TREE))
  await page.route('**/api/clothings/inspect', (route) => jsonFulfill(route, opts.inspect ?? INSPECT_TREE))
  await page.route('**/api/assemble', (route) => jsonFulfill(route, opts.assemble ?? {
    id: 'fake', output_fbx: '/tmp/fake.fbx', manifest: {},
  }))
  await page.route('**/api/assemblies', (route) => jsonFulfill(route, opts.assemblies ?? []))
}

test('compose home shows avatar picker populated from API', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'RigForge', level: 1 })).toBeVisible()
  const avatarSelect = page.getByTestId('avatar-picker')
  await expect(avatarSelect).toBeVisible()
  await expect(avatarSelect.locator('option')).toHaveCount(2)
  await expect(avatarSelect.locator('option').first()).toContainText('Maya')
})

test('adding a clothing path inspects and renders the mesh list', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')

  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()

  const clothing = page.getByTestId('clothing-item').first()
  await expect(clothing).toBeVisible()
  // Donor identification surfaced
  await expect(clothing).toContainText('maya')
  // The Blender-outliner view: armature + parallel mesh names. Bones do not show.
  await expect(clothing.getByTestId('armature-row')).toBeVisible()
  await expect(clothing.locator('[data-mesh-name="Body"]')).toBeVisible()
  await expect(clothing.locator('[data-mesh-name="Skirt"]')).toBeVisible()
  // Bones must NOT appear as rows in the list — the user doesn't see them.
  await expect(clothing.locator('[data-mesh-name="Hips"]')).toHaveCount(0)
  await expect(clothing.locator('[data-mesh-name="Spine"]')).toHaveCount(0)
})

test('unchecking a mesh row marks it dropped', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')

  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()

  const skirt = page.locator('[data-testid="clothing-item"] [data-mesh-name="Skirt"]')
  await skirt.locator('input[type=checkbox]').uncheck()
  await expect(skirt).toHaveAttribute('data-state', 'dropped')
})

test('assemble button posts the mesh-drop set', async ({ page }) => {
  await mockApi(page)
  const assembleRequests: unknown[] = []
  await page.route('**/api/assemble', async (route) => {
    if (route.request().method() === 'OPTIONS') {
      return route.fulfill({ status: 204, headers: CORS_HEADERS, body: '' })
    }
    const body = route.request().postDataJSON()
    assembleRequests.push(body)
    await route.fulfill({
      status: 200, contentType: 'application/json', headers: CORS_HEADERS,
      body: JSON.stringify({ id: 'fake', output_fbx: '/tmp/fake.fbx', manifest: {} }),
    })
  })

  await page.goto('/')

  // Add one clothing
  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()

  // Drop the Skirt mesh (model_id 13)
  const skirt = page.locator('[data-testid="clothing-item"] [data-mesh-name="Skirt"]')
  await skirt.locator('input[type=checkbox]').uncheck()

  await page.getByTestId('assemble-button').click()
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()

  expect(assembleRequests.length).toBe(1)
  const body = assembleRequests[0] as {
    target_id: string; clothing_path: string;
    drop_mesh_ids: number[]; target_drop_mesh_ids: number[];
  }
  expect(body.target_id).toBe('maya')
  expect(body.clothing_path).toBe('C:/fixtures/classic_chic.fbx')
  expect(new Set(body.drop_mesh_ids)).toEqual(new Set([13]))
  expect(body.target_drop_mesh_ids).toEqual([])
})

test('target panel shows target avatar meshes for pre-strip', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')

  const targetPanel = page.getByTestId('target-panel')
  await expect(targetPanel).toBeVisible()
  // Maya's default-outfit meshes show up so the modder can uncheck them
  await expect(targetPanel.locator('[data-mesh-name="Cloth"]')).toBeVisible()
  await expect(targetPanel.locator('[data-mesh-name="Shoes"]')).toBeVisible()
  await expect(targetPanel.locator('[data-mesh-name="Hat"]')).toBeVisible()
  // Bones do NOT appear
  await expect(targetPanel.locator('[data-mesh-name="Hips"]')).toHaveCount(0)
})

test('unchecking a target mesh forwards target_drop_mesh_ids on assemble', async ({ page }) => {
  await mockApi(page)
  const assembleRequests: unknown[] = []
  await page.route('**/api/assemble', async (route) => {
    if (route.request().method() === 'OPTIONS') {
      return route.fulfill({ status: 204, headers: CORS_HEADERS, body: '' })
    }
    const body = route.request().postDataJSON()
    assembleRequests.push(body)
    await route.fulfill({
      status: 200, contentType: 'application/json', headers: CORS_HEADERS,
      body: JSON.stringify({ id: 'fake', output_fbx: '/tmp/fake.fbx', manifest: {} }),
    })
  })

  await page.goto('/')

  // Strip Maya's bundled Cloth + Shoes (model_ids 202 and 203)
  const targetPanel = page.getByTestId('target-panel')
  await targetPanel.locator('[data-mesh-name="Cloth"] input[type=checkbox]').uncheck()
  await targetPanel.locator('[data-mesh-name="Shoes"] input[type=checkbox]').uncheck()

  // Add a clothing so assemble is enabled
  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()

  await page.getByTestId('assemble-button').click()
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()

  expect(assembleRequests.length).toBe(1)
  const body = assembleRequests[0] as {
    target_id: string
    clothing_path: string
    drop_mesh_ids: number[]
    target_drop_mesh_ids: number[]
  }
  expect(new Set(body.target_drop_mesh_ids)).toEqual(new Set([202, 203]))
  // Clothing side has no drops
  expect(body.drop_mesh_ids).toEqual([])
})

test('history route still shows the recent runs panel', async ({ page }) => {
  await mockApi(page)
  await page.goto('/history')
  await expect(page.getByTestId('recent-assemblies')).toBeVisible()
})
