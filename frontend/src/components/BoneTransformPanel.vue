<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { BoneTreeSource, BoneTransformValue, InspectedBone, Vec3 } from '../stores/compose'
import { useComposeStore } from '../stores/compose'

// "Move bones": set a bone's rest-pose position / rotation / scale with the
// same numbers Unity's Inspector shows for it (parent-local). The mesh
// weighted to the bone follows, exactly like dragging the bone in Unity —
// but baked into the FBX. Works for both a clothing and the target avatar.

const props = defineProps<{ source: BoneTreeSource }>()
const compose = useComposeStore()

const open = ref(false)
const filter = ref('')
const selectedId = ref<number | null>(null)

type Comp = 'position' | 'rotation' | 'scale'
const COMPS: Comp[] = ['position', 'rotation', 'scale']
const AXES = ['x', 'y', 'z'] as const
const LABEL: Record<Comp, string> = { position: 'position (m)', rotation: 'rotation (°)', scale: 'scale' }
const SHORT: Record<Comp, string> = { position: 'pos', rotation: 'rot', scale: 'scale' }
const STEP: Record<Comp, string> = { position: '0.001', rotation: '1', scale: '0.01' }
const DEFAULTS: Record<Comp, Vec3> = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] }

// Movable parts: bones and meshes (a rigid mesh Model can be moved too).
const bones = computed<InspectedBone[]>(() => {
  if (!props.source.inspect) return []
  const needle = filter.value.trim().toLowerCase()
  return props.source.inspect.bones
    .filter((b) => b.type_class === 'LimbNode' || b.type_class === 'Mesh')
    .filter((b) => !needle || b.name.toLowerCase().includes(needle))
    .sort((a, b) => a.name.localeCompare(b.name))
})

function boneById(id: number | null): InspectedBone | undefined {
  if (id === null || !props.source.inspect) return undefined
  return props.source.inspect.bones.find((b) => b.model_id === id)
}

// A rigid garment's top-level mesh moves in the target's Armature frame (the
// BE inspected it against the target); null for every other bone or mesh.
const rigidFrame = computed<string | null>(() => {
  const b = boneById(selectedId.value)
  const frame = props.source.inspect?.rigid_frame ?? null
  return frame && b?.type_class === 'Mesh' && b.parent_id === null ? frame : null
})

// Display rounding: 4 decimals is what a modder reads off Unity's Inspector.
function round4(v: number): number {
  return Math.round(v * 10000) / 10000
}

// The file's values for the selected bone (Unity frame), rounded for display.
function fileValues(b: InspectedBone | undefined): Record<Comp, Vec3> {
  const lt = b?.local_transform_unity
  const pick = (c: Comp): Vec3 => {
    const src = lt?.[c]
    if (!src || src.length !== 3) return DEFAULTS[c]
    return [round4(src[0]), round4(src[1]), round4(src[2])]
  }
  return { position: pick('position'), rotation: pick('rotation'), scale: pick('scale') }
}

// Editable inputs. Prefilled from the file, overlaid with an existing move.
const inputs = ref<Record<Comp, Vec3>>({ position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] })
const baseline = ref<Record<Comp, Vec3>>({ position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] })

function loadSelected() {
  const b = boneById(selectedId.value)
  const file = fileValues(b)
  const existing = selectedId.value !== null ? props.source.boneTransforms.get(selectedId.value) : undefined
  baseline.value = { position: [...file.position], rotation: [...file.rotation], scale: [...file.scale] }
  inputs.value = {
    position: [...(existing?.position ?? file.position)] as Vec3,
    rotation: [...(existing?.rotation ?? file.rotation)] as Vec3,
    scale: [...(existing?.scale ?? file.scale)] as Vec3,
  }
}

watch(selectedId, loadSelected)

function sameVec(a: Vec3, b: Vec3): boolean {
  return a.every((v, i) => Number.isFinite(v) && v === b[i])
}

// A component is "changed" when it differs from the FILE value — that is
// what gets sent. Matching the file again removes the component (and the
// whole move once nothing differs).
const changed = computed<BoneTransformValue>(() => {
  const out: BoneTransformValue = {}
  for (const c of COMPS) {
    if (!sameVec(inputs.value[c], baseline.value[c])) out[c] = [...inputs.value[c]] as Vec3
  }
  return out
})
const existingForSelected = computed(() =>
  selectedId.value !== null ? props.source.boneTransforms.get(selectedId.value) : undefined,
)
const dirty = computed(() => {
  if (selectedId.value === null) return false
  const ex = existingForSelected.value ?? {}
  const ch = changed.value
  for (const c of COMPS) {
    const a = ex[c]
    const b = ch[c]
    if (!!a !== !!b) return true
    if (a && b && !sameVec(a, b)) return true
  }
  return false
})

function toggle() {
  open.value = !open.value
  if (open.value && selectedId.value === null && bones.value.length > 0) {
    // Default to the first moved bone, else the first bone in the list.
    const firstMoved = [...props.source.boneTransforms.keys()][0]
    selectedId.value = firstMoved ?? bones.value[0].model_id
  }
}

function apply() {
  if (selectedId.value === null) return
  compose.setBoneTransform(props.source, selectedId.value, changed.value)
}

function resetToFile() {
  const b = boneById(selectedId.value)
  const file = fileValues(b)
  inputs.value = { position: [...file.position], rotation: [...file.rotation], scale: [...file.scale] }
}

function remove(id: number) {
  compose.clearBoneTransform(props.source, id)
  if (selectedId.value === id) loadSelected()
}

function edit(id: number) {
  selectedId.value = id
  loadSelected()
}

const moves = computed(() => [...props.source.boneTransforms.entries()])

function fmt(v?: Vec3): string {
  return v ? v.map((n) => String(round4(n))).join(', ') : '—'
}
</script>

<template>
  <div class="bone-move" data-testid="bone-move">
    <button class="toggle" data-testid="bone-move-toggle" @click="toggle">
      {{ open ? '▾' : '▸' }} Move bones
      <span v-if="moves.length" class="count" data-testid="bone-move-count">{{ moves.length }} moved</span>
    </button>

    <div v-if="open" class="body">
      <div class="pick">
        <input
          v-model="filter" type="search" placeholder="Filter bones…"
          class="filter" data-testid="bone-move-filter"
        />
        <select v-model="selectedId" data-testid="bone-move-bone" class="bone-select">
          <option v-for="b in bones" :key="b.model_id" :value="b.model_id">
            {{ b.name }}{{ b.type_class === 'Mesh' ? ' (mesh)' : '' }}
          </option>
        </select>
      </div>

      <div class="grid">
        <template v-for="c in COMPS" :key="c">
          <span class="lbl">{{ LABEL[c] }}</span>
          <label v-for="(ax, i) in AXES" :key="ax" class="axis">
            <span class="ax">{{ ax }}</span>
            <input
              type="number" :step="STEP[c]"
              v-model.number="inputs[c][i]"
              :data-testid="`bone-move-${SHORT[c]}-${ax}`"
            />
          </label>
        </template>
      </div>

      <p v-if="rigidFrame" class="hint" data-testid="bone-move-frame">
        Same numbers as Unity's Inspector with this mesh under the avatar's
        {{ rigidFrame }} (relative to {{ rigidFrame }}), whichever bone it hangs from.
        Only the rows you change are written.
      </p>
      <p v-else class="hint">
        Same numbers as Unity's Inspector for this bone (relative to its parent).
        Only the rows you change are written; the mesh on this bone follows it.
      </p>

      <div class="actions">
        <button class="apply" data-testid="bone-move-apply" :disabled="!dirty" @click="apply">
          Apply
        </button>
        <button class="reset" data-testid="bone-move-reset" @click="resetToFile">Reset to file</button>
      </div>

      <ul v-if="moves.length" class="moves">
        <li
          v-for="[id, v] in moves" :key="id" class="move-row"
          data-testid="bone-move-row" :data-bone-id="id"
        >
          <span class="name">{{ boneById(id)?.name ?? `#${id}` }}</span>
          <span v-if="v.position" class="comp">pos {{ fmt(v.position) }}</span>
          <span v-if="v.rotation" class="comp">rot {{ fmt(v.rotation) }}</span>
          <span v-if="v.scale" class="comp">scale {{ fmt(v.scale) }}</span>
          <button class="link" data-testid="bone-move-edit" @click="edit(id)">edit</button>
          <button class="link" data-testid="bone-move-remove" @click="remove(id)">remove</button>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.bone-move {
  margin-top: 0.75rem;
  padding-top: 0.5rem;
  border-top: 1px solid #1a1a1a;
}
.toggle {
  background: none;
  border: none;
  color: #ccc;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  padding: 0;
}
.count {
  margin-left: 0.5rem;
  color: #8c8;
  font-weight: 400;
}
.body {
  margin-top: 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.pick {
  display: flex;
  gap: 0.5rem;
}
.filter, .bone-select {
  background: #1a1a1a;
  border: 1px solid #333;
  color: #ddd;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-size: 0.85rem;
}
.filter { flex: 0 0 10rem; }
.bone-select { flex: 1; min-width: 0; }
.grid {
  display: grid;
  grid-template-columns: 7rem repeat(3, minmax(0, 1fr));
  gap: 0.3rem 0.5rem;
  align-items: center;
}
.lbl {
  color: #aaa;
  font-size: 0.8rem;
}
.axis {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}
.ax {
  color: #777;
  font-size: 0.75rem;
  width: 0.7rem;
}
.axis input {
  width: 100%;
  min-width: 0;
  background: #1a1a1a;
  border: 1px solid #333;
  color: #ddd;
  padding: 0.15rem 0.3rem;
  border-radius: 3px;
  font-size: 0.85rem;
}
.hint {
  color: #777;
  font-size: 0.75rem;
  margin: 0;
}
.actions {
  display: flex;
  gap: 0.5rem;
}
.apply, .reset {
  border: 1px solid #444;
  background: #222;
  color: #ddd;
  padding: 0.2rem 0.7rem;
  border-radius: 3px;
  cursor: pointer;
  font-size: 0.8rem;
}
.apply:disabled {
  opacity: 0.4;
  cursor: default;
}
.moves {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}
.move-row {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  font-size: 0.8rem;
  color: #bbb;
}
.move-row .name {
  color: #ddd;
  font-weight: 600;
}
.comp {
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  color: #9c9;
}
.link {
  background: none;
  border: none;
  color: #8ab;
  cursor: pointer;
  font-size: 0.75rem;
  padding: 0;
}
</style>
