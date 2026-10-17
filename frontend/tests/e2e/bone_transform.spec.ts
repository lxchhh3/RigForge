import { test, expect } from '@playwright/test'

// "Move bones" panel: pick a bone, edit the numbers Unity's Inspector would
// show for it (parent-local position / rotation / scale), apply. The assemble
// payload carries one entry per moved bone with ONLY the components the user
// changed, in the Unity frame; the backend converts to FBX.

const AVATARS = [{ id: 'maya', display_name: 'Maya v1.02.2', canonical_roles: ['Hips'] }]

const LT = (position: number[], rotation: number[], scale: number[]) => ({ position, rotation, scale })

const INSPECT = {
  donor_id: 'maya',
  donor_score: 0.96,
  total_bones: 4,
  bones: [
    { model_id: 1, name: 'Hips', type_class: 'LimbNode', parent_id: null, subtree_size: 3, cluster_weight_count: 100, deforms_meshes: ['Body'], local_transform_unity: LT([0, 0.9, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 2, name: 'Head', type_class: 'LimbNode', parent_id: 1, subtree_size: 2, cluster_weight_count: 50, deforms_meshes: ['Body'], local_transform_unity: LT([0, 0.4, 0], [0, 0, 0], [1, 1, 1]) },
    // The real case: the hat rides on its own bone.
    { model_id: 5, name: 'E#5_Hat.Root', type_class: 'LimbNode', parent_id: 2, subtree_size: 1, cluster_weight_count: 4721, deforms_meshes: ['E#5_Top_Hat'], local_transform_unity: LT([0.0372, 0.0787, 0.0527], [5.19, 0.09, -29.93], [1, 1, 1]) },
    { model_id: 10, name: 'E#5_Top_Hat', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
  ],
  blend_shape_channels: [],
}

const TARGET_INSPECT = {
  donor_id: 'maya',
  donor_score: 1.0,
  total_bones: 3,
  bones: [
    { model_id: 101, name: 'Hips', type_class: 'LimbNode', parent_id: null, subtree_size: 2, cluster_weight_count: 200, deforms_meshes: ['Body'], local_transform_unity: LT([0, 0.9, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 150, name: 'Hat Root', type_class: 'LimbNode', parent_id: 101, subtree_size: 1, cluster_weight_count: 30, deforms_meshes: ['Hat'], local_transform_unity: LT([0, 0.12, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 200, name: 'Body', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
  ],
  blend_shape_channels: [],
}

const STREAM = [
  { event: 'started', stem: 'clothing' },
  { event: 'progress', phase: 'phase_c', note: 'merge complete' },
  { event: 'done', id: 'fake', output_fbx: '/tmp/fake.fbx', manifest: {} },
]

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

async function mockAll(page: import('@playwright/test').Page, sink: unknown[]) {
  await page.route('**/api/avatars', (r) => jsonFulfill(r, AVATARS))
  await page.route('**/api/avatars/*/inspect', (r) => jsonFulfill(r, TARGET_INSPECT))
  await page.route('**/api/clothings/inspect', (r) => jsonFulfill(r, INSPECT))
  await page.route('**/api/assemblies', (r) => jsonFulfill(r, []))
  await page.route('**/api/assemble/stream', async (route) => {
    if (route.request().method() === 'OPTIONS') {
      return route.fulfill({ status: 204, headers: CORS, body: '' })
    }
    sink.push(route.request().postDataJSON())
    await route.fulfill({
      status: 200, contentType: 'application/x-ndjson', headers: CORS,
      body: STREAM.map((e) => JSON.stringify(e)).join('\n') + '\n',
    })
  })
}

async function addClothing(page: import('@playwright/test').Page) {
  await page.goto('/')
  await page.getByTestId('clothing-path-input').fill('C:/fixtures/clothing.fbx')
  await page.getByTestId('clothing-add-button').click()
  const item = page.getByTestId('clothing-item').first()
  await expect(item).toBeVisible()
  return item
}

async function assembleAndCapture(page: import('@playwright/test').Page, sink: unknown[]) {
  await page.getByTestId('assemble-button').click()
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()
  expect(sink.length).toBe(1)
  const body = sink[0] as {
    clothings: { bone_transforms: unknown[] }[]
    target_bone_transforms: unknown[]
  }
  // One request carries every garment: clothing-side moves ride inside
  // clothings[0]; target-side moves stay top-level.
  return {
    bone_transforms: body.clothings[0].bone_transforms,
    target_bone_transforms: body.target_bone_transforms,
  }
}

test('move a clothing bone: inputs prefill from the file, apply sends only the changed component', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  const item = await addClothing(page)

  await item.getByTestId('bone-move-toggle').click()
  await item.getByTestId('bone-move-bone').selectOption('5')
  // Prefilled with what Unity's Inspector shows for this bone.
  await expect(item.getByTestId('bone-move-pos-y')).toHaveValue('0.0787')
  await expect(item.getByTestId('bone-move-rot-z')).toHaveValue('-29.93')
  await expect(item.getByTestId('bone-move-apply')).toBeDisabled()

  await item.getByTestId('bone-move-pos-y').fill('0.1')
  await expect(item.getByTestId('bone-move-apply')).toBeEnabled()
  await item.getByTestId('bone-move-apply').click()

  const row = item.locator('[data-testid="bone-move-row"][data-bone-id="5"]')
  await expect(row).toBeVisible()
  await expect(row).toContainText('E#5_Hat.Root')

  const body = await assembleAndCapture(page, sink)
  expect(body.bone_transforms).toEqual([
    { bone_id: 5, position: [0.0372, 0.1, 0.0527], rotation: null, scale: null, frame: 'unity' },
  ])
  expect(body.target_bone_transforms).toEqual([])
})

test('removing a move drops it from the payload', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  const item = await addClothing(page)

  await item.getByTestId('bone-move-toggle').click()
  await item.getByTestId('bone-move-bone').selectOption('5')
  await item.getByTestId('bone-move-scale-x').fill('1.2')
  await item.getByTestId('bone-move-apply').click()
  await expect(item.getByTestId('bone-move-row')).toHaveCount(1)

  await item.getByTestId('bone-move-remove').click()
  await expect(item.getByTestId('bone-move-row')).toHaveCount(0)

  const body = await assembleAndCapture(page, sink)
  expect(body.bone_transforms).toEqual([])
})

test('a target-avatar bone move goes out as target_bone_transforms', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  await addClothing(page)

  const target = page.getByTestId('target-panel')
  await target.getByTestId('bone-move-toggle').click()
  await target.getByTestId('bone-move-bone').selectOption('150')
  await target.getByTestId('bone-move-rot-z').fill('10')
  await target.getByTestId('bone-move-apply').click()
  await expect(target.locator('[data-testid="bone-move-row"][data-bone-id="150"]')).toBeVisible()

  const body = await assembleAndCapture(page, sink)
  expect(body.bone_transforms).toEqual([])
  expect(body.target_bone_transforms).toEqual([
    { bone_id: 150, position: null, rotation: [0, 0, 10], scale: null, frame: 'unity' },
  ])
})

test('bone moves survive a reload', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  const item = await addClothing(page)
  await item.getByTestId('bone-move-toggle').click()
  await item.getByTestId('bone-move-bone').selectOption('5')
  await item.getByTestId('bone-move-pos-y').fill('0.1')
  await item.getByTestId('bone-move-apply').click()

  await page.reload()
  const again = page.getByTestId('clothing-item').first()
  await expect(again).toBeVisible()
  // The panel opens collapsed; the badge on the toggle already shows the
  // restored move, and opening it lists the row.
  await expect(again.getByTestId('bone-move-count')).toContainText('1 moved')
  await again.getByTestId('bone-move-toggle').click()
  await expect(again.locator('[data-testid="bone-move-row"][data-bone-id="5"]')).toBeVisible()
})
