import { defineStore } from 'pinia'
import { reactive, ref, watch } from 'vue'

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

export interface InspectedChannel {
  channel_id: number
  name: string
  owner_id: number | null
  owner_name: string | null
  // The mesh this morph deforms. Disambiguates duplicate channel names (several
  // meshes can each carry a "High heeled" / "Big breasts" channel).
  owner_mesh?: string | null
  // On-disk DeformPercent (0-100, typically 0). The slider starts here so
  // the user sees the file's actual baseline expression value.
  deform_percent: number
}

export interface InspectedClothing {
  donor_id: string | null
  donor_score: number
  total_bones: number
  bones: InspectedBone[]
  // Flat list of BlendShapeChannel SubDeformers. The FE renders one checkbox
  // per channel. Same shape on clothing + avatar inspect.
  blend_shape_channels: InspectedChannel[]
}

export interface ComposedClothing {
  path: string
  loading: boolean
  error: string | null
  inspect: InspectedClothing | null
  // model_ids explicitly marked DROP by the user. Excludes implicit cascade —
  // cascade is computed at assemble time from this set + the bone tree.
  droppedRoots: Set<number>
  // channel_ids the user wants stripped. Sent as drop_blend_shape_channel_ids.
  droppedChannels: Set<number>
  // channel_id → new DeformPercent (0-100). Only populated for channels the
  // user actually moved off the on-disk value; absence means "keep as-is".
  channelOverrides: Map<number, number>
}

// Anything that can back a BoneTree — a clothing or the target avatar.
// BoneTree only needs the inspect payload (for the bone list) and a mutable
// droppedRoots set, so the tree component works for both without a per-type
// branch.
export interface BoneTreeSource {
  inspect: InspectedClothing | null
  droppedRoots: Set<number>
  droppedChannels: Set<number>
  channelOverrides: Map<number, number>
}

export interface TargetState {
  avatarId: string | null
  loading: boolean
  error: string | null
  inspect: InspectedClothing | null
  droppedRoots: Set<number>
  droppedChannels: Set<number>
  channelOverrides: Map<number, number>
}

export interface AssembleResult {
  clothingPath: string
  id: string
  output_fbx: string
}

export type AssemblePhase = 'phase_a' | 'phase_b' | 'phase_c' | 'write'

export interface AssembleProgress {
  // null until the stream's first 'started' event; clothing-keyed via index.
  clothingPath: string
  // Latest phase reported by the backend. null while the run hasn't started
  // emitting progress events yet.
  phase: AssemblePhase | null
  // Latest human-readable note (one-liner from the orchestrator).
  note: string
  // 'running' from the started event through the final progress event;
  // 'succeeded' on the done event; 'failed' on the error event.
  status: 'running' | 'succeeded' | 'failed'
  error: string | null
}

// Phase order — drives the progress-bar fill on the FE. 'write' is the
// final phase before the response.
export const ASSEMBLE_PHASES: AssemblePhase[] = ['phase_a', 'phase_b', 'phase_c', 'write']

const API_BASE = (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/$/, '')

// localStorage key for the compose session. Bump the suffix if the persisted
// shape ever changes incompatibly.
const PERSIST_KEY = 'rigforge.compose.v1'

// The lean, JSON-serializable form of a per-source selection. inspect payloads
// are NOT stored — they're re-fetched on hydrate (cheap, and the backend's
// content-addressed conversion cache means a re-inspect returns the SAME ids).
interface PersistedSelection {
  droppedRoots: number[]
  droppedChannels: number[]
  channelOverrides: [number, number][]
}
interface PersistedSnapshot {
  targetId: string | null
  translateTargetMorphs?: boolean
  target: ({ avatarId: string | null } & PersistedSelection) | null
  clothings: ({ path: string } & PersistedSelection)[]
}

export const useComposeStore = defineStore('compose', () => {
  const targetId = ref<string | null>(null)
  // Translate the base avatar's own morph + mesh names to English. Clothing
  // names are always translated by the backend; this checkbox is the target's
  // opt-out for the rare downstream-Unity-binds-by-name case. Default on.
  const translateTargetMorphs = ref(true)
  const clothings = reactive<ComposedClothing[]>([])
  const assembling = ref(false)
  const results = reactive<AssembleResult[]>([])
  const assembleError = ref<string | null>(null)
  // One entry per clothing in flight. Cleared at the start of each assembleAll.
  const progress = reactive<AssembleProgress[]>([])
  const target = reactive<TargetState>({
    avatarId: null,
    loading: false,
    error: null,
    inspect: null,
    droppedRoots: new Set(),
    droppedChannels: new Set(),
    channelOverrides: new Map(),
  })

  // Set by _hydrate(); consumed by the next loadTargetInspect for the matching
  // avatar so a restored target-strip survives the inspect reset. One-shot.
  let pendingTargetRestore: ({ avatarId: string } & PersistedSelection) | null = null
  // While true, persistence writes are suppressed so the initial restore can't
  // overwrite the saved snapshot with half-built state.
  let hydrating = false

  async function loadTargetInspect(avatarId: string): Promise<void> {
    target.avatarId = avatarId
    target.loading = true
    target.error = null
    target.inspect = null
    target.droppedRoots = new Set()
    target.droppedChannels = new Set()
    target.channelOverrides = new Map()
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
      // Re-apply a restored target strip for this avatar (survives reload). The
      // reset above cleared the sets; put the persisted ones back now that the
      // tree is loaded.
      if (pendingTargetRestore && pendingTargetRestore.avatarId === avatarId) {
        target.droppedRoots = new Set(pendingTargetRestore.droppedRoots)
        target.droppedChannels = new Set(pendingTargetRestore.droppedChannels)
        target.channelOverrides = new Map(pendingTargetRestore.channelOverrides)
        pendingTargetRestore = null
      }
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
      droppedChannels: new Set(),
      channelOverrides: new Map(),
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

  function toggleChannel(src: BoneTreeSource, channel_id: number, keep: boolean): void {
    if (keep) src.droppedChannels.delete(channel_id)
    else src.droppedChannels.add(channel_id)
  }

  // Set or clear the DeformPercent override for a channel. The store stores
  // it as a delta from the on-disk value, but we always persist whatever the
  // slider committed — clearing requires the caller to pass the on-disk
  // value explicitly (or call clearChannelOverride).
  function setChannelOverride(
    src: BoneTreeSource, channel_id: number, deform_percent: number,
  ): void {
    src.channelOverrides.set(channel_id, deform_percent)
  }
  function clearChannelOverride(src: BoneTreeSource, channel_id: number): void {
    src.channelOverrides.delete(channel_id)
  }

  async function assembleAll(): Promise<AssembleResult[]> {
    if (!targetId.value) {
      assembleError.value = 'pick a target avatar first'
      return []
    }
    assembling.value = true
    assembleError.value = null
    results.splice(0, results.length)
    progress.splice(0, progress.length)
    // Target-side mesh drops: every entry in target.droppedRoots is a
    // Mesh-Model id (set by MeshList). Send them as target_drop_mesh_ids;
    // the pipeline drops the Mesh + Geometry + Skin + Clusters as a unit.
    const targetDropMeshIds: number[] = (() => {
      if (!target.inspect || target.droppedRoots.size === 0) return []
      return [...target.droppedRoots]
    })()
    const targetDropChannelIds: number[] = (() => {
      if (!target.inspect || target.droppedChannels.size === 0) return []
      return [...target.droppedChannels]
    })()
    const targetChannelOverrides = (() => {
      if (!target.inspect || target.channelOverrides.size === 0) return []
      return [...target.channelOverrides.entries()].map(
        ([channel_id, deform_percent]) => ({ channel_id, deform_percent }),
      )
    })()

    try {
      for (const c of clothings) {
        if (!c.inspect) continue
        const dropMeshIds: number[] = [...c.droppedRoots]
        const dropChannelIds: number[] = [...c.droppedChannels]
        const channelOverrides = [...c.channelOverrides.entries()].map(
          ([channel_id, deform_percent]) => ({ channel_id, deform_percent }),
        )
        const entry: AssembleProgress = reactive({
          clothingPath: c.path,
          phase: null,
          note: 'starting…',
          status: 'running',
          error: null,
        })
        progress.push(entry)

        const res = await fetch(`${API_BASE}/api/assemble/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_id: targetId.value,
            clothing_path: c.path,
            drop_mesh_ids: dropMeshIds,
            target_drop_mesh_ids: targetDropMeshIds,
            drop_blend_shape_channel_ids: dropChannelIds,
            target_drop_blend_shape_channel_ids: targetDropChannelIds,
            blend_shape_channel_overrides: channelOverrides,
            target_blend_shape_channel_overrides: targetChannelOverrides,
            translate_target_morphs: translateTargetMorphs.value,
          }),
        })
        if (!res.ok) {
          // Validation errors (400/404) come back as normal JSON; surface
          // the FastAPI detail string when available.
          let detail = `HTTP ${res.status}`
          try {
            const body = await res.json()
            if (body && typeof body.detail === 'string') detail = body.detail
          } catch { /* not JSON */ }
          entry.status = 'failed'
          entry.error = detail
          assembleError.value = `assemble failed for ${c.path}: ${detail}`
          break
        }
        if (!res.body) {
          entry.status = 'failed'
          entry.error = 'no response body'
          assembleError.value = entry.error
          break
        }

        // Read NDJSON line-by-line. The backend emits one JSON object per
        // newline; we buffer partial lines across chunks.
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        let runErrored = false
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          let nl = buf.indexOf('\n')
          while (nl >= 0) {
            const line = buf.slice(0, nl).trim()
            buf = buf.slice(nl + 1)
            nl = buf.indexOf('\n')
            if (!line) continue
            let evt: { event: string; phase?: AssemblePhase; note?: string; id?: string; output_fbx?: string; error?: string }
            try {
              evt = JSON.parse(line)
            } catch {
              continue
            }
            if (evt.event === 'progress' || evt.event === 'heartbeat') {
              if (evt.phase) entry.phase = evt.phase
              if (evt.note) entry.note = evt.note
            } else if (evt.event === 'done') {
              entry.status = 'succeeded'
              entry.note = 'done'
              results.push({
                clothingPath: c.path,
                id: evt.id ?? '',
                output_fbx: evt.output_fbx ?? '',
              })
            } else if (evt.event === 'error') {
              entry.status = 'failed'
              entry.error = evt.error ?? 'assemble failed'
              assembleError.value = `assemble failed for ${c.path}: ${entry.error}`
              runErrored = true
            }
          }
        }
        if (runErrored) break
      }
      return [...results]
    } finally {
      assembling.value = false
    }
  }

  // --- persistence (survive reload / app reopen) -------------------------
  // We persist only the user's selections (drops + overrides), never the heavy
  // inspect payloads — those are re-fetched on hydrate. Pairs with the backend
  // bin->ASCII cache: a re-inspect of the same clothing returns the same ids,
  // so persisted ids still line up after a BE restart.

  function snapshot(): PersistedSnapshot {
    return {
      targetId: targetId.value,
      translateTargetMorphs: translateTargetMorphs.value,
      target: target.avatarId
        ? {
            avatarId: target.avatarId,
            droppedRoots: [...target.droppedRoots],
            droppedChannels: [...target.droppedChannels],
            channelOverrides: [...target.channelOverrides.entries()],
          }
        : null,
      clothings: clothings.map((c) => ({
        path: c.path,
        droppedRoots: [...c.droppedRoots],
        droppedChannels: [...c.droppedChannels],
        channelOverrides: [...c.channelOverrides.entries()],
      })),
    }
  }

  function persist(): void {
    if (hydrating) return
    if (typeof localStorage === 'undefined') return
    try {
      localStorage.setItem(PERSIST_KEY, JSON.stringify(snapshot()))
    } catch {
      /* quota / private-mode — persistence is best-effort, never fatal */
    }
  }

  function hydrate(): void {
    if (typeof localStorage === 'undefined') return
    let snap: PersistedSnapshot | null = null
    try {
      const raw = localStorage.getItem(PERSIST_KEY)
      snap = raw ? (JSON.parse(raw) as PersistedSnapshot) : null
    } catch {
      snap = null
    }
    if (!snap) return

    hydrating = true
    // Restore the picked target + queue its strip for the upcoming inspect.
    if (snap.targetId) targetId.value = snap.targetId
    if (typeof snap.translateTargetMorphs === 'boolean') {
      translateTargetMorphs.value = snap.translateTargetMorphs
    }
    if (snap.target?.avatarId) {
      pendingTargetRestore = {
        avatarId: snap.target.avatarId,
        droppedRoots: snap.target.droppedRoots ?? [],
        droppedChannels: snap.target.droppedChannels ?? [],
        channelOverrides: snap.target.channelOverrides ?? [],
      }
    }
    // Restore clothings WITH their selections, then (re-)inspect to repaint.
    for (const cs of snap.clothings ?? []) {
      clothings.push({
        path: cs.path,
        loading: true,
        error: null,
        inspect: null,
        droppedRoots: new Set(cs.droppedRoots ?? []),
        droppedChannels: new Set(cs.droppedChannels ?? []),
        channelOverrides: new Map(cs.channelOverrides ?? []),
      })
      inspect(clothings[clothings.length - 1])
    }
    hydrating = false
  }

  hydrate()
  // Deep-watch the selection-bearing state; serialize on any change. inspect
  // payloads aren't in the snapshot, so inspect resolutions don't churn writes.
  watch([targetId, translateTargetMorphs, target, clothings], persist, { deep: true })

  return {
    targetId,
    translateTargetMorphs,
    target,
    clothings,
    assembling,
    results,
    assembleError,
    progress,
    loadTargetInspect,
    addClothing,
    removeClothing,
    toggleBone,
    toggleChannel,
    setChannelOverride,
    clearChannelOverride,
    assembleAll,
  }
})
