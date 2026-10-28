import { test, expect } from '@playwright/test'

// kevin's ask (2026-10-28): "it's harder and harder for me to remember all my
// changes so why not just let our system read them?" — Load Unity scene reads
// the saved .unity and pre-fills the compose: the target and its hidden
// meshes / blendshape values, each garment FBX with its own, rigid props with
// the bone they hang from and their placement. He reviews, then Assembles.

const AVATARS = [
  { id: 'moe', display_name: 'Moe', canonical_roles: ['Hips'] },
  { id: 'maya', display_name: 'Maya v1.02.2', canonical_roles: ['Hips'] },
]

const LT = (position: number[], rotation: number[], scale: number[]) => ({ position, rotation, scale })

const TARGET_INSPECT = {
  donor_id: 'maya', donor_score: 1, total_bones: 4, rigid: false,
  bones: [
    { model_id: 101, name: 'Hips', type_class: 'LimbNode', parent_id: null, subtree_size: 2, cluster_weight_count: 200, deforms_meshes: ['Body'], local_transform_unity: LT([0, 0.9, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 150, name: 'Head', type_class: 'LimbNode', parent_id: 101, subtree_size: 1, cluster_weight_count: 30, deforms_meshes: ['Body'], local_transform_unity: LT([0, 0.5, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 200, name: 'Body', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
    { model_id: 210, name: 'Cloth', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
  ],
  blend_shape_channels: [
    { channel_id: 900, name: 'う', owner_id: 1, owner_name: 'Body', owner_mesh: 'Body', deform_percent: 0 },
  ],
}

const OUTFIT_INSPECT = {
  donor_id: 'maya', donor_score: 1, total_bones: 3, rigid: false,
  bones: [
    { model_id: 1, name: 'Hips_MFBase', type_class: 'LimbNode', parent_id: null, subtree_size: 2, cluster_weight_count: 100, deforms_meshes: ['EC_Dress'], local_transform_unity: LT([0, 0.9, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 2, name: 'EC_PB_HRibbon_Root', type_class: 'LimbNode', parent_id: 1, subtree_size: 1, cluster_weight_count: 10, deforms_meshes: ['EC_Dress'], local_transform_unity: LT([0, 0.1, 0], [0, 0, 0], [1, 1, 1]) },
    { model_id: 20, name: 'EC_Dress', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
    { model_id: 21, name: 'EC_Shorts', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [-90, 0, 0], [1, 1, 1]) },
  ],
  blend_shape_channels: [
    { channel_id: 700, name: 'Source_Breast_big', owner_id: 5, owner_name: 'EC_Dress', owner_mesh: 'EC_Dress', deform_percent: 0 },
  ],
}

const PIN_INSPECT = {
  donor_id: null, donor_score: 0, total_bones: 1, rigid: true, rigid_frame: 'Armature',
  bones: [
    { model_id: 30, name: 'EC_HairAcc_01', type_class: 'Mesh', parent_id: null, subtree_size: 1, cluster_weight_count: 0, deforms_meshes: [], local_transform_unity: LT([0, 0, 0], [0, 0, 0], [1, 1, 1]) },
  ],
  blend_shape_channels: [],
}

const OUTFIT = 'D:/New_Mochi/Assets/Out/retargeted_Eku.fbx'
const PIN = 'D:/New_Mochi/Assets/Hairpin/EclatAndCrumb_HairPin_01.fbx'
const SCENE = 'D:/New_Mochi/Assets/Eclat_Coffee.unity'

const SCENE_READ = {
  scene: SCENE,
  target_id: 'maya',
  target: {
    drop_mesh_ids: [210],
    blend_shape_channel_overrides: [{ channel_id: 900, deform_percent: 100 }],
    bone_transforms: [],
  },
  clothings: [
    {
      clothing_path: OUTFIT, source: 'Armature/EclatAndCrumb_Eku', rigid: false, attach_to_bone_id: null,
      drop_mesh_ids: [21],
      blend_shape_channel_overrides: [{ channel_id: 700, deform_percent: 77.1 }],
      bone_transforms: [{ bone_id: 2, position: [-0.0929, 0.1221, 0.0289], rotation: [-1.79, 34.84, 72.54], scale: [1, 1, 1] }],
    },
    {
      clothing_path: PIN, source: 'Armature/EclatAndCrumb_Hairpin_MA', rigid: true, attach_to_bone_id: 150,
      drop_mesh_ids: [], blend_shape_channel_overrides: [],
      bone_transforms: [{ bone_id: 30, position: [0.0628, -0.1148, 1.2139], rotation: [54.8, 89.86, 89.82], scale: [1, 1, 1] }],
    },
  ],
  skipped: [
    { name: 'EclatAndCrumb_Toast', path: 'D:/New_Mochi/Assets/Toast.fbx', reason: "its bones (PB_Toast_01) match no avatar — RigForge can't merge a prop with its own armature yet" },
  ],
  notes: [
    'maya (target): hide Cloth; 1 blendshape value(s)',
    "EclatAndCrumb_Eku -> retargeted_Eku.fbx: 'Head_MFBase' is moved (-0.0, +2.8, +4.0) cm in the scene, but it merges into the avatar's own 'Head', so RigForge can't carry that move",
  ],
}

const STREAM = [
  { event: 'started', stem: 'x' },
  { event: 'done', id: 'fake', output_fbx: '/tmp/fake.fbx', manifest: {} },
]

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
}

type Route = import('@playwright/test').Route
type Page = import('@playwright/test').Page

function jsonFulfill(route: Route, body: unknown, status = 200) {
  if (route.request().method() === 'OPTIONS') {
    return route.fulfill({ status: 204, headers: CORS, body: '' })
  }
  return route.fulfill({ status, contentType: 'application/json', headers: CORS, body: JSON.stringify(body) })
}

async function mockAll(page: Page, sink: unknown[], sceneSink: unknown[] = [],
                       scene: { status: number; body: unknown } = { status: 200, body: SCENE_READ }) {
  await page.route('**/api/avatars', (r) => jsonFulfill(r, AVATARS))
  await page.route('**/api/avatars/*/inspect', (r) => jsonFulfill(r, TARGET_INSPECT))
  await page.route('**/api/clothings/inspect', (r) => {
    const path = (r.request().postDataJSON() as { path?: string } | null)?.path ?? ''
    return jsonFulfill(r, path.includes('HairPin') ? PIN_INSPECT : OUTFIT_INSPECT)
  })
  await page.route('**/api/unity/scene', (r) => {
    if (r.request().method() === 'POST') sceneSink.push(r.request().postDataJSON())
    return jsonFulfill(r, scene.body, scene.status)
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

async function loadScene(page: Page) {
  await page.getByTestId('unity-scene-input').fill(`"${SCENE}"`)
  await page.getByTestId('unity-scene-load').click()
  await expect(page.getByTestId('unity-scene-report')).toBeVisible()
  await expect(page.getByTestId('clothing-item')).toHaveCount(2)
  for (const item of await page.getByTestId('clothing-item').all()) {
    await expect(item.getByText('Inspecting…')).toHaveCount(0)
  }
}

type Payload = {
  target_id: string
  target_drop_mesh_ids: number[]
  target_blend_shape_channel_overrides: { channel_id: number; deform_percent: number }[]
  target_bone_transforms: unknown[]
  clothings: {
    clothing_path: string
    drop_mesh_ids: number[]
    blend_shape_channel_overrides: { channel_id: number; deform_percent: number }[]
    bone_transforms: { bone_id: number; position: number[]; rotation: number[]; scale: number[]; frame: string }[]
    attach_to_bone_id: number | null
  }[]
}

async function assemble(page: Page, sink: unknown[]): Promise<Payload> {
  await page.getByTestId('assemble-button').click()
  await expect(page.getByTestId('assemble-result').first()).toBeVisible()
  return sink[sink.length - 1] as Payload
}

test('Load Unity scene sends the scene path and pre-fills target, garments and props', async ({ page }) => {
  const sink: unknown[] = []
  const sceneSink: unknown[] = []
  await mockAll(page, sink, sceneSink)
  await page.goto('/')
  await expect(page.getByTestId('avatar-picker')).toHaveValue('moe')

  await loadScene(page)
  // Explorer's quotes are stripped before the path goes out.
  expect(sceneSink).toEqual([{ scene_path: SCENE }])
  // The scene's avatar becomes the target.
  await expect(page.getByTestId('avatar-picker')).toHaveValue('maya')

  // Target: Cloth unticked, Body kept; 'う' at 100.
  const target = page.getByTestId('target-panel')
  await expect(target.locator('[data-testid="mesh-row"][data-mesh-id="210"] input[type="checkbox"]')).not.toBeChecked()
  await expect(target.locator('[data-testid="mesh-row"][data-mesh-id="200"] input[type="checkbox"]')).toBeChecked()
  await expect(target.locator('[data-testid="blendshape-row"][data-channel-id="900"] [data-testid="blendshape-number"]')).toHaveValue('100')

  // Outfit: EC_Shorts unticked, one bone moved.
  const outfit = page.getByTestId('clothing-item').filter({ hasText: 'retargeted_Eku.fbx' })
  await expect(outfit.locator('[data-testid="mesh-row"][data-mesh-id="21"] input[type="checkbox"]')).not.toBeChecked()
  await expect(outfit.getByTestId('bone-move-count')).toContainText('1 moved')

  // The hair pin hangs from Head.
  const pin = page.getByTestId('clothing-item').filter({ hasText: 'HairPin_01.fbx' })
  await expect(pin.getByTestId('attach-bone')).toHaveValue('150')

  // What could not be carried is listed, the Toast among it.
  const report = page.getByTestId('unity-scene-report')
  await expect(report.getByTestId('unity-scene-skipped')).toContainText('EclatAndCrumb_Toast')
  await expect(report.getByTestId('unity-scene-note')).toHaveCount(2)
  await expect(report).toContainText("can't carry that move")
})

test('the pre-filled compose goes out whole on Assemble', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  await page.goto('/')
  await loadScene(page)

  const p = await assemble(page, sink)
  expect(p.target_id).toBe('maya')
  expect(p.target_drop_mesh_ids).toEqual([210])
  expect(p.target_blend_shape_channel_overrides).toEqual([{ channel_id: 900, deform_percent: 100 }])
  const byPath = Object.fromEntries(p.clothings.map((c) => [c.clothing_path, c]))
  expect(Object.keys(byPath)).toEqual([OUTFIT, PIN])
  expect(byPath[OUTFIT].drop_mesh_ids).toEqual([21])
  expect(byPath[OUTFIT].blend_shape_channel_overrides).toEqual([{ channel_id: 700, deform_percent: 77.1 }])
  expect(byPath[OUTFIT].bone_transforms).toEqual([
    { bone_id: 2, position: [-0.0929, 0.1221, 0.0289], rotation: [-1.79, 34.84, 72.54], scale: [1, 1, 1], frame: 'unity' },
  ])
  expect(byPath[OUTFIT].attach_to_bone_id).toBeNull()
  expect(byPath[PIN].attach_to_bone_id).toBe(150)
  expect(byPath[PIN].bone_transforms[0].bone_id).toBe(30)
})

test('loading a scene replaces what was on the page, and survives a reload', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink)
  await page.goto('/')
  await page.getByTestId('clothing-path-input').fill('C:/old/OldCoat.fbx')
  await page.getByTestId('clothing-add-button').click()
  await expect(page.getByTestId('clothing-item').filter({ hasText: 'OldCoat.fbx' })).toBeVisible()

  await loadScene(page)
  await expect(page.getByTestId('clothing-item').filter({ hasText: 'OldCoat.fbx' })).toHaveCount(0)

  await page.reload()
  await expect(page.getByTestId('clothing-item')).toHaveCount(2)
  await expect(page.getByTestId('avatar-picker')).toHaveValue('maya')
  await expect(page.getByTestId('unity-scene-report')).toContainText('EclatAndCrumb_Toast')
  const p = await assemble(page, sink)
  expect(p.target_drop_mesh_ids).toEqual([210])
  expect(p.clothings.map((c) => c.drop_mesh_ids)).toEqual([[21], []])
})

test('a scene the backend refuses shows its reason and leaves the page as it was', async ({ page }) => {
  const sink: unknown[] = []
  await mockAll(page, sink, [], { status: 400, body: { detail: 'no object in the scene is built on a registry avatar (maya, moe)' } })
  await page.goto('/')
  await page.getByTestId('clothing-path-input').fill('C:/old/OldCoat.fbx')
  await page.getByTestId('clothing-add-button').click()

  await page.getByTestId('unity-scene-input').fill(SCENE)
  await page.getByTestId('unity-scene-load').click()
  await expect(page.getByTestId('unity-scene-error')).toContainText('registry avatar')
  await expect(page.getByTestId('clothing-item').filter({ hasText: 'OldCoat.fbx' })).toBeVisible()
  await expect(page.getByTestId('avatar-picker')).toHaveValue('moe')
})
