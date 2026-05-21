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
  blend_shape_channels: [
    { channel_id: 901, name: 'Smile', owner_id: 800, owner_name: 'Face', deform_percent: 0 },
    { channel_id: 902, name: 'Wink', owner_id: 800, owner_name: 'Face', deform_percent: 0 },
    { channel_id: 903, name: 'CapeBillow', owner_id: 801, owner_name: 'Cape', deform_percent: 0 },
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
  blend_shape_channels: [
    { channel_id: 701, name: 'TargetSmile', owner_id: 700, owner_name: 'Face', deform_percent: 0 },
    { channel_id: 702, name: 'TargetBlink', owner_id: 700, owner_name: 'Face', deform_percent: 0 },
    { channel_id: 703, name: 'OutfitFlex', owner_id: 710, owner_name: 'Cloth', deform_percent: 0 },
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

// NDJSON fulfill — same `events` list as the backend would emit; serialized
// one JSON object per line. The FE store reads chunked but Playwright sends
// the whole body in one go, which is fine: the parser handles any chunking.
function ndjsonFulfill(
  route: import('@playwright/test').Route,
  events: unknown[],
) {
  if (route.request().method() === 'OPTIONS') {
    return route.fulfill({ status: 204, headers: CORS_HEADERS, body: '' })
  }
  return route.fulfill({
    status: 200,
    contentType: 'application/x-ndjson',
    headers: CORS_HEADERS,
    body: events.map((e) => JSON.stringify(e)).join('\n') + '\n',
  })
}

// Default scripted stream — happy path; mirrors the orchestrator's emit
// boundaries (phase_a → phase_b → phase_c → write → done).
const DEFAULT_ASSEMBLE_STREAM = [
  { event: 'started', stem: 'classic_chic' },
  { event: 'progress', phase: 'phase_a', note: 'donor=maya score=0.96' },
  { event: 'progress', phase: 'phase_b', note: '12 drops + 3 renames (cache hit)' },
  { event: 'progress', phase: 'phase_c', note: 'merge complete' },
  { event: 'progress', phase: 'write', note: 'wrote classic_chic.fbx' },
  { event: 'done', id: 'fake', output_fbx: '/tmp/fake.fbx', manifest: {} },
]

// Helper that captures the JSON body of the streaming assemble POST and
// returns a scripted NDJSON success stream. Used by tests that need to
// inspect what the FE actually sent (drop ids, etc.).
async function captureAssembleRequests(
  page: import('@playwright/test').Page,
  sink: unknown[],
  events: unknown[] = DEFAULT_ASSEMBLE_STREAM,
) {
  await page.route('**/api/assemble/stream', async (route) => {
    if (route.request().method() === 'OPTIONS') {
      return route.fulfill({ status: 204, headers: CORS_HEADERS, body: '' })
    }
    sink.push(route.request().postDataJSON())
    await route.fulfill({
      status: 200,
      contentType: 'application/x-ndjson',
      headers: CORS_HEADERS,
      body: events.map((e) => JSON.stringify(e)).join('\n') + '\n',
    })
  })
}

async function mockApi(page: import('@playwright/test').Page, opts: {
  avatars?: unknown
  inspect?: unknown
  assembleStream?: unknown[]
  assemblies?: unknown
  targetInspect?: unknown
} = {}) {
  await page.route('**/api/avatars', (route) => jsonFulfill(route, opts.avatars ?? AVATARS))
  await page.route('**/api/avatars/*/inspect', (route) =>
    jsonFulfill(route, opts.targetInspect ?? TARGET_INSPECT_TREE))
  await page.route('**/api/clothings/inspect', (route) => jsonFulfill(route, opts.inspect ?? INSPECT_TREE))
  await page.route('**/api/assemble/stream', (route) =>
    ndjsonFulfill(route, opts.assembleStream ?? DEFAULT_ASSEMBLE_STREAM))
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
  await captureAssembleRequests(page, assembleRequests)

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
    drop_blend_shape_channel_ids: number[];
    target_drop_blend_shape_channel_ids: number[];
  }
  expect(body.target_id).toBe('maya')
  expect(body.clothing_path).toBe('C:/fixtures/classic_chic.fbx')
  expect(new Set(body.drop_mesh_ids)).toEqual(new Set([13]))
  expect(body.target_drop_mesh_ids).toEqual([])
  // No blendshape drops in this run
  expect(body.drop_blend_shape_channel_ids).toEqual([])
  expect(body.target_drop_blend_shape_channel_ids).toEqual([])
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
  await captureAssembleRequests(page, assembleRequests)

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

test('clothing item lists blendshape channels for the modder to uncheck', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')

  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()

  const clothing = page.getByTestId('clothing-item').first()
  const bsList = clothing.getByTestId('blendshape-list')
  await expect(bsList).toBeVisible()
  await expect(bsList.locator('[data-channel-name="Smile"]')).toBeVisible()
  await expect(bsList.locator('[data-channel-name="Wink"]')).toBeVisible()
  await expect(bsList.locator('[data-channel-name="CapeBillow"]')).toBeVisible()
})

test('unchecking blendshapes forwards drop_blend_shape_channel_ids on assemble', async ({ page }) => {
  await mockApi(page)
  const assembleRequests: unknown[] = []
  await captureAssembleRequests(page, assembleRequests)

  await page.goto('/')

  // Strip Maya's TargetSmile from the target
  const targetPanel = page.getByTestId('target-panel')
  await targetPanel.locator('[data-channel-name="TargetSmile"] input[type=checkbox]').uncheck()

  // Add clothing and drop its CapeBillow morph
  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()
  const clothing = page.getByTestId('clothing-item').first()
  await clothing.locator('[data-channel-name="CapeBillow"] input[type=checkbox]').uncheck()

  await page.getByTestId('assemble-button').click()
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()

  expect(assembleRequests.length).toBe(1)
  const body = assembleRequests[0] as {
    drop_blend_shape_channel_ids: number[]
    target_drop_blend_shape_channel_ids: number[]
  }
  expect(new Set(body.drop_blend_shape_channel_ids)).toEqual(new Set([903]))
  expect(new Set(body.target_drop_blend_shape_channel_ids)).toEqual(new Set([701]))
})

test('blendshape slider sends deform_percent overrides on assemble', async ({ page }) => {
  await mockApi(page)
  const assembleRequests: unknown[] = []
  await captureAssembleRequests(page, assembleRequests)

  await page.goto('/')

  // Move the target's TargetSmile slider to 75
  const targetPanel = page.getByTestId('target-panel')
  const targetSmileRow = targetPanel.locator('[data-channel-name="TargetSmile"]')
  await targetSmileRow.locator('input[type=range]').fill('75')

  // Add a clothing and move its Wink slider to 40
  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()
  const clothing = page.getByTestId('clothing-item').first()
  const winkRow = clothing.locator('[data-channel-name="Wink"]')
  await winkRow.locator('input[type=range]').fill('40')

  await page.getByTestId('assemble-button').click()
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()

  expect(assembleRequests.length).toBe(1)
  const body = assembleRequests[0] as {
    blend_shape_channel_overrides: { channel_id: number; deform_percent: number }[]
    target_blend_shape_channel_overrides: { channel_id: number; deform_percent: number }[]
  }
  expect(body.blend_shape_channel_overrides).toEqual([
    { channel_id: 902, deform_percent: 40 },
  ])
  expect(body.target_blend_shape_channel_overrides).toEqual([
    { channel_id: 701, deform_percent: 75 },
  ])
})

test('snapping a blendshape slider back to its on-disk value clears the override', async ({ page }) => {
  // Fixture has Smile at on-disk value 0. Move slider away then back to 0 —
  // the assemble payload should NOT carry an override entry for it.
  await mockApi(page)
  const assembleRequests: unknown[] = []
  await captureAssembleRequests(page, assembleRequests)

  await page.goto('/')

  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()
  const clothing = page.getByTestId('clothing-item').first()
  const smile = clothing.locator('[data-channel-name="Smile"]')

  await smile.locator('input[type=range]').fill('60')
  // Snap back
  await smile.locator('input[type=range]').fill('0')

  await page.getByTestId('assemble-button').click()
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()

  const body = assembleRequests[0] as {
    blend_shape_channel_overrides: { channel_id: number; deform_percent: number }[]
  }
  expect(body.blend_shape_channel_overrides).toEqual([])
})

test('assemble renders a progress row with phase + note per clothing', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')

  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()
  await page.getByTestId('assemble-button').click()

  // Wait for the success terminal state on the row
  const row = page.getByTestId('assemble-progress-row').first()
  await expect(row).toBeVisible()
  await expect(row).toHaveAttribute('data-status', 'succeeded')
  // The last progress note's text should have made it into the note element
  const note = row.getByTestId('assemble-progress-note')
  await expect(note).toBeVisible()
  // The bar should reflect 100% on done
  const bar = row.getByTestId('assemble-progress-bar')
  await expect(bar).toHaveCSS('width', /\d+px/)
  // Final result is also shown
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()
})

test('assemble surfaces backend error events in the progress row', async ({ page }) => {
  await mockApi(page, {
    assembleStream: [
      { event: 'started', stem: 'classic_chic' },
      { event: 'progress', phase: 'phase_a', note: 'donor=maya score=0.96' },
      { event: 'progress', phase: 'phase_b', note: 'calling Ollama' },
      { event: 'error', error: 'PhaseBError: validator failure after re-prompt' },
    ],
  })
  await page.goto('/')

  await page.getByTestId('clothing-path-input').fill('C:/fixtures/classic_chic.fbx')
  await page.getByTestId('clothing-add-button').click()
  await page.getByTestId('assemble-button').click()

  const row = page.getByTestId('assemble-progress-row').first()
  await expect(row).toHaveAttribute('data-status', 'failed')
  await expect(row.getByTestId('assemble-progress-note')).toContainText('PhaseBError')
})

test('history route still shows the recent runs panel', async ({ page }) => {
  await mockApi(page)
  await page.goto('/history')
  await expect(page.getByTestId('recent-assemblies')).toBeVisible()
})
