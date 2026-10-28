<script setup lang="ts">
import { computed } from 'vue'
import type { ComposedClothing } from '../stores/compose'
import { isProp, useComposeStore } from '../stores/compose'
import MeshList from './MeshList.vue'
import BlendShapeList from './BlendShapeList.vue'
import GeometryEditPanel from './GeometryEditPanel.vue'
import BoneTransformPanel from './BoneTransformPanel.vue'

const props = defineProps<{ clothing: ComposedClothing }>()
const compose = useComposeStore()

const meshCount = computed(() =>
  props.clothing.inspect
    ? props.clothing.inspect.bones.filter((b) => b.type_class === 'Mesh').length
    : 0,
)
const channelCount = computed(() =>
  props.clothing.inspect?.blend_shape_channels?.length ?? 0,
)

// A rigid garment (no bones) can hang from one of the TARGET's bones so it
// follows her; by default it hangs from the target's Armature (as in Unity),
// which follows no bone.
const targetBones = computed(() =>
  (compose.target.inspect?.bones ?? [])
    .filter((b) => b.type_class === 'LimbNode')
    .sort((a, b) => a.name.localeCompare(b.name)),
)
// <select> values are strings in the DOM: '' = the default parent (null in the store).
const defaultParent = computed(() => props.clothing.inspect?.rigid_frame ?? 'avatar root')
const attachValue = computed<string>({
  get: () => (props.clothing.attachBoneId === null ? '' : String(props.clothing.attachBoneId)),
  set: (v) => {
    props.clothing.attachBoneId = v === '' ? null : Number(v)
  },
})

function remove() {
  compose.removeClothing(props.clothing.path)
}
</script>

<template>
  <section class="clothing-item" data-testid="clothing-item">
    <header class="head">
      <span class="path" :title="clothing.path">{{ clothing.path }}</span>
      <button class="remove" @click="remove" data-testid="clothing-remove">remove</button>
    </header>

    <div v-if="clothing.loading" class="status">Inspecting…</div>
    <div v-else-if="clothing.error" class="status error">
      Inspect failed: {{ clothing.error }}
    </div>
    <template v-else-if="clothing.inspect">
      <div class="meta">
        <template v-if="clothing.inspect.rigid">
          <strong data-testid="rigid-badge">rigid garment (no bones)</strong>
        </template>
        <template v-else-if="clothing.inspect.self_rigged">
          <strong data-testid="self-rigged-badge">prop with its own armature (no avatar bones)</strong>
        </template>
        <template v-else>
          donor: <strong>{{ clothing.inspect.donor_id ?? 'unknown' }}</strong>
          <span v-if="clothing.inspect.donor_id">
            (score {{ clothing.inspect.donor_score.toFixed(2) }})
          </span>
        </template>
        — {{ meshCount }} meshes, {{ channelCount }} blendshapes
      </div>
      <div v-if="isProp(clothing.inspect)" class="attach">
        <label>
          attach to bone
          <select v-model="attachValue" data-testid="attach-bone" class="attach-select">
            <option value="">{{ defaultParent }} (follows no bone)</option>
            <option v-for="b in targetBones" :key="b.model_id" :value="String(b.model_id)">{{ b.name }}</option>
          </select>
        </label>
        <p class="hint">
          Hangs from this bone of the target so it follows her; it stays where it
          is placed (same as dragging it onto the bone in Unity's Hierarchy).
        </p>
      </div>
      <div class="tree-wrap">
        <MeshList :source="clothing" armature-label="Armature" />
      </div>
      <div v-if="channelCount > 0" class="bs-wrap">
        <BlendShapeList :source="clothing" heading="Blendshapes" />
      </div>
      <GeometryEditPanel v-if="meshCount > 0" :clothing="clothing" />
      <BoneTransformPanel :source="clothing" />
    </template>
  </section>
</template>

<style scoped>
.clothing-item {
  border: 1px solid #2a2a2a;
  border-radius: 6px;
  padding: 0.85rem 1rem;
}
.head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}
.path {
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.85rem;
  color: #bbb;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.remove {
  background: none;
  border: 1px solid #444;
  color: #aaa;
  padding: 0.15rem 0.55rem;
  border-radius: 3px;
  cursor: pointer;
  font-size: 0.75rem;
}
.remove:hover {
  border-color: #888;
  color: #ddd;
}
.status {
  color: #888;
  font-style: italic;
  padding: 0.5rem 0;
}
.status.error {
  color: #d88;
}
.meta {
  color: #999;
  font-size: 0.85rem;
  padding: 0.25rem 0 0.5rem 0;
  border-bottom: 1px solid #1a1a1a;
  margin-bottom: 0.5rem;
}
.meta strong {
  color: #ddd;
  font-weight: 600;
}
.attach {
  padding: 0.25rem 0 0.5rem 0;
  font-size: 0.85rem;
  color: #bbb;
}
.attach-select {
  margin-left: 0.5rem;
  background: #1a1a1a;
  color: #ddd;
  border: 1px solid #444;
  padding: 0.15rem 0.35rem;
}
.attach .hint {
  margin: 0.25rem 0 0 0;
  color: #777;
  font-size: 0.75rem;
}
.tree-wrap {
  max-height: 400px;
  overflow-y: auto;
}
.bs-wrap {
  margin-top: 0.75rem;
  padding-top: 0.5rem;
  border-top: 1px solid #1a1a1a;
}
</style>
