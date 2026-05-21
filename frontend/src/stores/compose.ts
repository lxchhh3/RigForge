import { defineStore } from 'pinia'
import { reactive, ref } from 'vue'

export interface InspectedBone {
  model_id: number
  name: string
  type_class: string
  parent_id: number | null
  subtree_size: number
  cluster_weight_count: number
  // Names of meshes this bone is bound to (via cluster→skin→geometry or, for
  // Mesh-type Models, directly). Empty for pure structural bones with no skin.
  deforms_meshes: string[]
}

export interface InspectedClothing {
  donor_id: string | null
  donor_score: number
  total_bones: number
  bones: InspectedBone[]
}

export interface ComposedClothing {
  path: string
  loading: boolean
  error: string | null
  inspect: InspectedClothing | null
  // model_ids explicitly marked DROP by the user. Excludes implicit cascade —
  // cascade is computed at assemble time from this set + the bone tree.
  droppedRoots: Set<number>
}

// Anything that can back a BoneTree — a clothing or the target avatar.
// BoneTree only needs the inspect payload (for the bone list) and a mutable
// droppedRoots set, so the tree component works for both without a per-type
// branch.
export interface BoneTreeSource {
  inspect: InspectedClothing | null
  droppedRoots: Set<number>
}

export interface TargetState {
  avatarId: string | null
  loading: boolean
  error: string | null
  inspect: InspectedClothing | null
  droppedRoots: Set<number>
}

export interface AssembleResult {
  clothingPath: string
  id: string
  output_fbx: string
}

const API_BASE = (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/$/, '')

export const useComposeStore = defineStore('compose', () => {
  const targetId = ref<string | null>(null)
  const clothings = reactive<ComposedClothing[]>([])
  const assembling = ref(false)
  const results = reactive<AssembleResult[]>([])
  const assembleError = ref<string | null>(null)
  const target = reactive<TargetState>({
    avatarId: null,
    loading: false,
    error: null,
    inspect: null,
    droppedRoots: new Set(),
  })

  async function loadTargetInspect(avatarId: string): Promise<void> {
    target.avatarId = avatarId
    target.loading = true
    target.error = null
    target.inspect = null
    target.droppedRoots = new Set()
    try {
      const res = await fetch(`${API_BASE}/api/avatars/${encodeURIComponent(avatarId)}/inspect`)
      if (!res.ok) {
        let detail = `HTTP ${res.status}`
        try {
          const body = await res.json()
          if (body && typeof body.detail === 'string') detail = body.detail
        } catch { /* not JSON */ }
        throw new Error(detail)
      }
      target.inspect = (await res.json()) as InspectedClothing
    } catch (e) {
      target.error = e instanceof Error ? e.message : String(e)
    } finally {
      target.loading = false
    }
  }

  function addClothing(path: string): ComposedClothing {
    // Windows Explorer's "Copy as path" wraps the path in double quotes;
    // strip them so the backend gets the bare path string.
    const cleanPath = path.trim().replace(/^"|"$/g, '')
    clothings.push({
      path: cleanPath,
      loading: true,
      error: null,
      inspect: null,
      droppedRoots: new Set(),
    })
    // After push, the object in the reactive array is a proxy. Mutating the
    // original local reference would bypass reactivity, so we resolve the
    // proxy and pass IT to inspect().
    const proxied = clothings[clothings.length - 1]
    inspect(proxied)
    return proxied
  }

  function removeClothing(path: string): void {
    const idx = clothings.findIndex((c) => c.path === path)
    if (idx >= 0) clothings.splice(idx, 1)
  }

  async function inspect(c: ComposedClothing): Promise<void> {
    c.loading = true
    c.error = null
    try {
      const res = await fetch(`${API_BASE}/api/clothings/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: c.path }),
      })
      if (!res.ok) {
        // Surface FastAPI's `detail` so the user sees the actual reason
        // (e.g. "file not found: ...path..." or "parse failed: ...").
        let detail = `HTTP ${res.status}`
        try {
          const body = await res.json()
          if (body && typeof body.detail === 'string') detail = body.detail
        } catch { /* response wasn't JSON; keep the HTTP status */ }
        throw new Error(detail)
      }
      c.inspect = (await res.json()) as InspectedClothing
    } catch (e) {
      c.error = e instanceof Error ? e.message : String(e)
    } finally {
      c.loading = false
    }
  }

  function toggleBone(src: BoneTreeSource, bone_id: number, keep: boolean): void {
    if (keep) src.droppedRoots.delete(bone_id)
    else src.droppedRoots.add(bone_id)
  }

  async function assembleAll(): Promise<AssembleResult[]> {
    if (!targetId.value) {
      assembleError.value = 'pick a target avatar first'
      return []
    }
    assembling.value = true
    assembleError.value = null
    results.splice(0, results.length)
    // Target-side mesh drops: every entry in target.droppedRoots is a
    // Mesh-Model id (set by MeshList). Send them as target_drop_mesh_ids;
    // the pipeline drops the Mesh + Geometry + Skin + Clusters as a unit.
    const targetDropMeshIds: number[] = (() => {
      if (!target.inspect || target.droppedRoots.size === 0) return []
      return [...target.droppedRoots]
    })()

    try {
      for (const c of clothings) {
        if (!c.inspect) continue
        const dropMeshIds: number[] = [...c.droppedRoots]
        const res = await fetch(`${API_BASE}/api/assemble`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_id: targetId.value,
            clothing_path: c.path,
            drop_mesh_ids: dropMeshIds,
            target_drop_mesh_ids: targetDropMeshIds,
          }),
        })
        if (!res.ok) {
          assembleError.value = `assemble failed for ${c.path}: HTTP ${res.status}`
          break
        }
        const body = await res.json()
        results.push({ clothingPath: c.path, id: body.id, output_fbx: body.output_fbx })
      }
      return [...results]
    } finally {
      assembling.value = false
    }
  }

  return {
    targetId,
    target,
    clothings,
    assembling,
    results,
    assembleError,
    loadTargetInspect,
    addClothing,
    removeClothing,
    toggleBone,
    assembleAll,
  }
})
