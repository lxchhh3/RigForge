import { defineStore } from 'pinia'
import { reactive, ref, watch } from 'vue'

export type Vec3 = [number, number, number]

// A bone's rest-pose local transform as Unity's Inspector shows it
// (parent-local; position in meters, rotation in Euler degrees).
export interface LocalTransformUnity {
  position: number[]
  rotation: number[]
  scale: number[]
}

// A requested bone move: only the components the user changed are set. Sent
// as `{bone_id, position|null, rotation|null, scale|null, frame: 'unity'}`.
export interface BoneTransformValue {
  position?: Vec3
  rotation?: Vec3
  scale?: Vec3
}

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
  // Current local transform (Unity frame) — prefill for the "Move bones"
  // panel. Optional so an older backend payload still renders.
  local_transform_unity?: LocalTransformUnity
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
  // No bones, only meshes (a rigid prop): merged as-is. The BE decides; the FE
  // then offers "attach to bone". Absent on payloads from older backends.
  rigid?: boolean
  // Rigid garment inspected against a target: the target Model (its Armature)
  // whose frame the top-level meshes' local_transform_unity — and their moves —
  // read in, as Unity's Inspector shows a prop under the avatar's Armature.
  // null/absent = the file's own values (no target yet).
  rigid_frame?: string | null
  bones: InspectedBone[]
  // Flat list of BlendShapeChannel SubDeformers. The FE renders one checkbox
  // per channel. Same shape on clothing + avatar inspect.
  blend_shape_channels: InspectedChannel[]
}

export interface TasteEditResult {
  edited_path: string
  mesh: string
  verts_before: number
  verts_after: number
  verts_removed: number
  // false = the cut stripped skin weights off the new edge (sole would stop
  // following the foot). The FE blocks "use this" when false.
  weights_ok: boolean
  // data: URLs (base64 PNG) or null if the render step was skipped/failed.
  preview_before: string | null
  preview_after: string | null
}

export interface ComposedClothing {
  path: string
  loading: boolean
  error: string | null
  inspect: InspectedClothing | null
  // true once a taste-edit (e.g. cut sole height) has been applied — `path`
  // now points at the edited FBX. Cosmetic; not persisted.
  edited?: boolean
  // model_ids explicitly marked DROP by the user. Excludes implicit cascade —
  // cascade is computed at assemble time from this set + the bone tree.
  droppedRoots: Set<number>
  // channel_ids the user wants stripped. Sent as drop_blend_shape_channel_ids.
  droppedChannels: Set<number>
  // channel_id → new DeformPercent (0-100). Only populated for channels the
  // user actually moved off the on-disk value; absence means "keep as-is".
  channelOverrides: Map<number, number>
  // model_id → bone move (rest-pose local transform, Unity frame). Only the
  // components the user changed are present.
  boneTransforms: Map<number, BoneTransformValue>
  // Rigid garment only: the TARGET bone it hangs from so it follows her.
  // null = the avatar root (follows no bone).
  attachBoneId: number | null
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
  boneTransforms: Map<number, BoneTransformValue>
}

export interface TargetState {
  avatarId: string | null
  loading: boolean
  error: string | null
  inspect: InspectedClothing | null
  droppedRoots: Set<number>
  droppedChannels: Set<number>
  channelOverrides: Map<number, number>
  boneTransforms: Map<number, BoneTransformValue>
}

export interface AssembleResult {
  // Every garment merged into this ONE output, in merge order.
  clothingPaths: string[]
  // Joined display form of clothingPaths (older consumers).
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
  // 'queued' until the stream announces this garment; 'running' through its
  // phases; 'succeeded' once the next garment starts (its merge is in) or on
  // the done event; 'failed' on the error event (the whole run is void).
  status: 'queued' | 'running' | 'succeeded' | 'failed'
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
  // Optional: snapshots written before bone moves existed have no key.
  boneTransforms?: [number, BoneTransformValue][]
}

// Wire form of one bone move — matches the backend's BoneTransform model.
export interface BoneTransformPayload {
  bone_id: number
  position: Vec3 | null
  rotation: Vec3 | null
  scale: Vec3 | null
  frame: 'unity'
}

export function boneTransformPayload(
  moves: Map<number, BoneTransformValue>,
): BoneTransformPayload[] {
  return [...moves.entries()].map(([bone_id, v]) => ({
    bone_id,
    position: v.position ?? null,
    rotation: v.rotation ?? null,
    scale: v.scale ?? null,
    frame: 'unity',
  }))
}
interface PersistedSnapshot {
  targetId: string | null
  translateTargetMorphs?: boolean
  target: ({ avatarId: string | null } & PersistedSelection) | null
  clothings: ({ path: string; attachBoneId?: number | null } & PersistedSelection)[]
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
    boneTransforms: new Map(),
  })

  // Set by _hydrate(); consumed by the next loadTargetInspect for the matching
  // avatar so a restored target-strip survives the inspect reset. One-shot.
  let pendingTargetRestore: ({ avatarId: string } & PersistedSelection) | null = null
  // While true, persistence writes are suppressed so the initial restore can't
  // overwrite the saved snapshot with half-built state.
  let hydrating = false

  async function loadTargetInspect(avatarId: string): Promise<void> {
    // An attach pick names a bone id of the PREVIOUS avatar; drop it on a
    // switch (a reload re-picks the same avatar and keeps it).
    if (target.avatarId !== null && target.avatarId !== avatarId) {
      for (const c of clothings) c.attachBoneId = null
    }
    // A rigid garment's Move prefill is in the target's Armature frame:
    // re-inspect those against the new avatar.
    if (target.avatarId !== avatarId) {
      for (const c of clothings) if (c.inspect?.rigid) inspect(c)
    }
    target.avatarId = avatarId
    target.loading = true
    target.error = null
    target.inspect = null
    target.droppedRoots = new Set()
    target.droppedChannels = new Set()
    target.channelOverrides = new Map()
    target.boneTransforms = new Map()
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
        target.boneTransforms = new Map(pendingTargetRestore.boneTransforms ?? [])
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
      boneTransforms: new Map(),
      attachBoneId: null,
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
    const forTarget = targetId.value
    c.loading = true
    c.error = null
    try {
      const res = await fetch(`${API_BASE}/api/clothings/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // The target lets a rigid garment report its placement in the
        // target's Armature frame (what its move reads in).
        body: JSON.stringify({ path: c.path, target_id: forTarget }),
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
    // The target changed while this was in flight: a rigid garment's prefill
    // is per target, so ask again.
    if (c.inspect?.rigid && targetId.value !== forTarget) await inspect(c)
  }

  // Geometry pre-step (taste-edit): cut a mesh's sole height via the headless
  // Blender backend. Returns the result (edited path + before/after preview)
  // WITHOUT mutating the clothing — the user reviews, then calls
  // applyEditedPath to adopt it.
  async function tasteEdit(
    c: ComposedClothing, mesh: string, cutPct: number,
  ): Promise<TasteEditResult> {
    const res = await fetch(`${API_BASE}/api/clothings/taste-edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        clothing_path: c.path, mesh, op: 'bisect_cut', cut_pct: cutPct,
      }),
    })
    if (!res.ok) {
      let detail = `HTTP ${res.status}`
      try {
        const body = await res.json()
        if (body && typeof body.detail === 'string') detail = body.detail
      } catch { /* not JSON */ }
      throw new Error(detail)
    }
    return (await res.json()) as TasteEditResult
  }

  // Adopt an edited clothing: swap the path to the edited FBX and re-inspect.
  // The cut produces a fresh FBX with new ids, so any prior drops/overrides no
  // longer line up — we reset them (the user re-picks against the edited mesh).
  function applyEditedPath(c: ComposedClothing, editedPath: string): void {
    c.path = editedPath
    c.edited = true
    c.droppedRoots = new Set()
    c.droppedChannels = new Set()
    c.channelOverrides = new Map()
    c.boneTransforms = new Map()
    inspect(c)
  }

  // Set (or replace) the move for one bone. A value with no components left
  // clears the entry so the payload never carries no-op moves.
  function setBoneTransform(
    src: BoneTreeSource, bone_id: number, value: BoneTransformValue,
  ): void {
    if (!value.position && !value.rotation && !value.scale) {
      src.boneTransforms.delete(bone_id)
    } else {
      src.boneTransforms.set(bone_id, value)
    }
  }
  function clearBoneTransform(src: BoneTreeSource, bone_id: number): void {
    src.boneTransforms.delete(bone_id)
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
    const targetBoneTransforms = target.inspect
      ? boneTransformPayload(target.boneTransforms)
      : []

    // ONE request carrying every inspected garment: the backend merges them
    // in list order into ONE file (garment 1 onto the target, garment 2 onto
    // that result, ...). Never one request/file per garment.
    const active = clothings.filter((c) => c.inspect)
    if (active.length === 0) {
      assembleError.value = 'no inspected clothing to assemble'
      assembling.value = false
      return []
    }
    for (const c of active) {
      progress.push(reactive({
        clothingPath: c.path, phase: null, note: 'queued', status: 'queued', error: null,
      }) as AssembleProgress)
    }
    const clothingPayload = active.map((c) => ({
      clothing_path: c.path,
      drop_mesh_ids: [...c.droppedRoots],
      drop_blend_shape_channel_ids: [...c.droppedChannels],
      blend_shape_channel_overrides: [...c.channelOverrides.entries()].map(
        ([channel_id, deform_percent]) => ({ channel_id, deform_percent }),
      ),
      bone_transforms: boneTransformPayload(c.boneTransforms),
      attach_to_bone_id: c.inspect?.rigid ? c.attachBoneId : null,
    }))
    const paths = active.map((c) => c.path)
    // Any failure voids the whole run (no file is written): every row fails,
    // the current garment's row carrying the real error.
    const failRun = (current: number, detail: string) => {
      progress.forEach((p, i) => {
        if (i === current) {
          p.status = 'failed'
          p.error = detail
        } else if (p.status !== 'failed') {
          p.status = 'failed'
          p.error = `run aborted: ${paths[current] ?? 'a garment'} failed`
        }
      })
      assembleError.value = `assemble failed for ${paths[current] ?? 'compose'}: ${detail}`
    }

    try {
      let current = 0
      const res = await fetch(`${API_BASE}/api/assemble/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_id: targetId.value,
          clothings: clothingPayload,
          target_drop_mesh_ids: targetDropMeshIds,
          target_drop_blend_shape_channel_ids: targetDropChannelIds,
          target_blend_shape_channel_overrides: targetChannelOverrides,
          translate_target_morphs: translateTargetMorphs.value,
          target_bone_transforms: targetBoneTransforms,
        }),
      })
      if (!res.ok) {
        // Validation errors (400/404/422) come back as normal JSON; surface
        // the FastAPI detail string when available.
        let detail = `HTTP ${res.status}`
        try {
          const body = await res.json()
          if (body && typeof body.detail === 'string') detail = body.detail
        } catch { /* not JSON */ }
        failRun(current, detail)
        return []
      }
      if (!res.body) {
        failRun(current, 'no response body')
        return []
      }

      // Read NDJSON line-by-line. One JSON object per newline; partial lines
      // are buffered across chunks. A 'clothing' event switches the row being
      // driven and progress events carry clothing_index too; an older single-
      // clothing stream has neither and drives row 0.
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
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
          let evt: {
            event: string; phase?: AssemblePhase; note?: string; id?: string
            output_fbx?: string; error?: string; index?: number; clothing_index?: number
          }
          try {
            evt = JSON.parse(line)
          } catch {
            continue
          }
          if (evt.event === 'started') {
            const p = progress[current]
            if (p && p.status === 'queued') {
              p.status = 'running'
              p.note = 'starting…'
            }
          } else if (evt.event === 'clothing') {
            const next = evt.index ?? current + 1
            // Every garment before this one has merged in.
            for (let i = 0; i < next; i++) {
              const prev = progress[i]
              if (prev && prev.status !== 'failed') {
                prev.status = 'succeeded'
                prev.note = 'merged'
              }
            }
            current = next
            const p = progress[current]
            if (p) {
              p.status = 'running'
              p.phase = null
              p.note = 'starting…'
            }
          } else if (evt.event === 'progress' || evt.event === 'heartbeat') {
            const i = evt.clothing_index ?? current
            const p = progress[i]
            if (p) {
              if (p.status === 'queued') p.status = 'running'
              if (evt.phase) p.phase = evt.phase
              if (evt.note) p.note = evt.note
            }
          } else if (evt.event === 'done') {
            for (const p of progress) {
              if (p.status !== 'failed') {
                p.status = 'succeeded'
                p.note = 'done'
              }
            }
            results.push({
              clothingPaths: [...paths],
              clothingPath: paths.join(' + '),
              id: evt.id ?? '',
              output_fbx: evt.output_fbx ?? '',
            })
          } else if (evt.event === 'error') {
            failRun(current, evt.error ?? 'assemble failed')
          }
        }
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
            boneTransforms: [...target.boneTransforms.entries()],
          }
        : null,
      clothings: clothings.map((c) => ({
        path: c.path,
        droppedRoots: [...c.droppedRoots],
        droppedChannels: [...c.droppedChannels],
        channelOverrides: [...c.channelOverrides.entries()],
        boneTransforms: [...c.boneTransforms.entries()],
        attachBoneId: c.attachBoneId,
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
        boneTransforms: snap.target.boneTransforms ?? [],
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
        boneTransforms: new Map(cs.boneTransforms ?? []),
        attachBoneId: cs.attachBoneId ?? null,
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
    tasteEdit,
    applyEditedPath,
    toggleBone,
    toggleChannel,
    setChannelOverride,
    clearChannelOverride,
    setBoneTransform,
    clearBoneTransform,
    assembleAll,
  }
})
