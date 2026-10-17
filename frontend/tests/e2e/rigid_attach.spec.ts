import { test, expect } from '@playwright/test'

// A garment with no bones (a rigid prop — kevin's GG headset) merges as-is.
// The BE flags it `rigid`; the FE then offers "attach to bone": a target bone
// it hangs from so it follows her (default: the target's Armature, as in Unity,
// which follows no bone). The pick goes out per garment as `attach_to_bone_id`.

const AVATARS = [{ id: 'maya', display_name: 'Maya v1.02.2', canonical_roles: ['Hips'] }]

const LT = (position: number[], rotation: number[], scale: number[]) => ({ position, rotation, scale })

const RIGID_INSPECT = {
  donor_id: null,
  donor_score: 0,
  total_bones: 1,
  rigid: true,
  bones: [
    { model_id: 10, name: 'GG_headset', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
  ],
  blend_shape_channels: [],
}

const RIGGED_INSPECT = {
  donor_id: 'maya',
  donor_score: 1,
  total_bones: 2,
  rigid: false,
  bones: [
    { model_id: 1, name: 'Hips', type_class: 'LimbNode', parent_id: null, subtree_size: 1, cluster_weight_count: 100, deforms_meshes: ['Skirt'], local_transform_unity: LT([0, 0.9, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 20, name: 'Skirt', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
  ],
  blend_shape_channels: [],
}

const TARGET_INSPECT = {
  donor_id: 'maya',
  donor_score: 1.0,
  total_bones: 4,
  rigid: false,
  bones: [
    { model_id: 101, name: 'Hips', type_class: 'LimbNode', parent_id: null, subtree_size: 3, cluster_weight_count: 200, deforms_meshes: ['Body'], local_transform_unity: LT([0, 0.9, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 140, name: 'Neck', type_class: 'LimbNode', parent_id: 101, subtree_size: 2, cluster_weight_count: 40, deforms_meshes: ['Body'], local_transform_unity: LT([0, 0.25, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 150, name: 'Head', type_class: 'LimbNode', parent_id: 140, subtree_size: 1, cluster_weight_count: 30, deforms_meshes: ['Body'], local_transform_unity: LT([0, 0.04, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 200, name: 'Body', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
  ],
  blend_shape_channels: [],
}

const STREAM = [
  { event: 'started', stem: 'garment' },
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
  await page.route('**/api/clothings/inspect', (r) => {
    const path = (r.request().postDataJSON() as { path?: string } | null)?.path ?? ''
    return jsonFulfill(r, path.includes('headset') ? RIGID_INSPECT : RIGGED_INSPECT)
  })
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

async function addClothing(page: import('@playwright/test').Page, path: string) {
  await page.getByTestId('clothing-path-input').fill(path)
  await page.getByTestId('clothing-add-button').click()
  const item = page.getByTestId('clothing-item').filter({ hasText: path })
  await expect(item).toBeVisible()
  await expect(item.getByText('Inspecting…')).toHaveCount(0)
  return item
}

type Payload = { clothings: { clothing_path: string; attach_to_bone_id: number | null }[] }

async function assembleAndCapture(page: import('@playwright/test').Page, sink: unknown[]) {
  await page.getByTestId('assemble-button').click()
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()
  expect(sink.length).toBe(1)
  return sink[0] as Payload
}

test('a rigid garment hangs from the picked target bone: it goes out as attach_to_bone_id', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  await page.goto('/')
  const item = await addClothing(page, 'C:/fixtures/GG_headset.fbx')

  await expect(item.getByTestId('rigid-badge')).toContainText('rigid')
  const pick = item.getByTestId('attach-bone')
  // Default: the avatar root. The choices are the TARGET's bones, not meshes.
  await expect(pick).toHaveValue('')
  const labels = await pick.locator('option').allTextContents()
  expect(labels.some((l) => l.includes('Neck'))).toBe(true)
  expect(labels.some((l) => l.includes('Body'))).toBe(false)

  await pick.selectOption('140')
  const body = await assembleAndCapture(page, sink)
  expect(body.clothings[0].attach_to_bone_id).toBe(140)
})

test('a rigged garment has no attach picker and sends null', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  await page.goto('/')
  const item = await addClothing(page, 'C:/fixtures/skirt.fbx')

  await expect(item.getByTestId('attach-bone')).toHaveCount(0)
  await expect(item.getByTestId('rigid-badge')).toHaveCount(0)
  const body = await assembleAndCapture(page, sink)
  expect(body.clothings[0].attach_to_bone_id).toBeNull()
})

// kevin's GG.unity (2026-10-15): the headset is a child of Maya's Armature, so
// the numbers he reads off Unity's Inspector are Armature-local. The BE gives
// a rigid mesh's prefill in the target's Armature frame when told the target.
const RIGID_INSPECT_IN_ARMATURE = {
  ...RIGID_INSPECT,
  rigid_frame: 'Armature',
  bones: [
    { ...RIGID_INSPECT.bones[0], local_transform_unity: LT([0, 0, 0], [0, 0, 0], [1, 1, 1]) },
  ],
}

async function mockRigidAgainstTarget(
  page: import('@playwright/test').Page, sink: unknown[], inspected: { path?: string; target_id?: string | null }[],
) {
  await mockAll(page, sink)
  await page.route('**/api/clothings/inspect', (r) => {
    if (r.request().method() === 'OPTIONS') return jsonFulfill(r, {})
    const req = (r.request().postDataJSON() ?? {}) as { path?: string; target_id?: string | null }
    inspected.push(req)
    return jsonFulfill(r, req.target_id ? RIGID_INSPECT_IN_ARMATURE : RIGID_INSPECT)
  })
}

test('a rigid garment is inspected against the target; its move reads in the Armature frame', async ({ page }) => {
  const sink: unknown[] = []
  const inspected: { path?: string; target_id?: string | null }[] = []
  await mockRigidAgainstTarget(page, sink, inspected)
  await page.goto('/')
  await expect(page.getByTestId('target-panel')).toBeVisible()
  const item = await addClothing(page, 'C:/fixtures/GG_headset.fbx')

  expect(inspected.at(-1)?.target_id).toBe('maya')
  // The default parent is the target's Armature, as in Unity.
  await expect(item.getByTestId('attach-bone').locator('option[value=""]')).toContainText('Armature')

  await item.getByTestId('bone-move-toggle').click()
  await item.getByTestId('bone-move-bone').selectOption('10')
  await expect(item.getByTestId('bone-move-frame')).toContainText('Armature')
  await expect(item.getByTestId('bone-move-rot-x')).toHaveValue('0')

  await item.getByTestId('bone-move-pos-y').fill('-0.06')
  await item.getByTestId('bone-move-pos-z').fill('1.08')
  await item.getByTestId('bone-move-rot-x').fill('-81.41')
  await item.getByTestId('bone-move-apply').click()
  await item.getByTestId('attach-bone').selectOption('140')

  const body = (await assembleAndCapture(page, sink)) as Payload & {
    clothings: { bone_transforms: { bone_id: number; position: number[]; rotation: number[] }[] }[]
  }
  expect(body.clothings[0].attach_to_bone_id).toBe(140)
  expect(body.clothings[0].bone_transforms).toEqual([
    expect.objectContaining({ bone_id: 10, position: [0, -0.06, 1.08], rotation: [-81.41, 0, 0] }),
  ])
})

test('switching the target re-inspects a rigid garment against the new avatar', async ({ page }) => {
  const sink: unknown[] = []
  const inspected: { path?: string; target_id?: string | null }[] = []
  await mockRigidAgainstTarget(page, sink, inspected)
  await page.route('**/api/avatars', (r) => jsonFulfill(r, [
    ...AVATARS, { id: 'moe', display_name: 'Moe', canonical_roles: ['Hips'] },
  ]))
  await page.goto('/')
  await expect(page.getByTestId('target-panel')).toBeVisible()
  await addClothing(page, 'C:/fixtures/GG_headset.fbx')
  expect(inspected.at(-1)?.target_id).toBe('maya')

  await page.getByTestId('avatar-picker').selectOption('moe')
  await expect.poll(() => inspected.at(-1)?.target_id).toBe('moe')
})

test('the attach pick survives a reload', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  await page.goto('/')
  const item = await addClothing(page, 'C:/fixtures/GG_headset.fbx')
  await item.getByTestId('attach-bone').selectOption('150')

  await page.reload()
  const again = page.getByTestId('clothing-item').filter({ hasText: 'GG_headset' })
  await expect(again.getByTestId('attach-bone')).toHaveValue('150')
})
