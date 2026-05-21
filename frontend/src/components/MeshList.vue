<script setup lang="ts">
import { computed } from 'vue'
import type { BoneTreeSource, InspectedBone } from '../stores/compose'
import { useComposeStore } from '../stores/compose'

const props = defineProps<{
  source: BoneTreeSource
  // The "armature" entry — display label for the rig as a single row. Doesn't
  // get a checkbox (you can't drop the rig). Just a header for context, the
  // way Blender shows the armature above its sibling meshes.
  armatureLabel?: string
}>()

const compose = useComposeStore()

// Flat list of Mesh-type Models. This is the user's mental model: in
// Blender's outliner, the armature sits alongside meshes (Body, Hair, Cloth,
// ...) — not a deep bone tree. Bones live in the backend.
const meshes = computed<InspectedBone[]>(() => {
  if (!props.source.inspect) return []
  return props.source.inspect.bones
    .filter((b) => b.type_class === 'Mesh')
    .sort((a, b) => a.name.localeCompare(b.name))
})

function isDropped(id: number): boolean {
  return props.source.droppedRoots.has(id)
}

function onToggle(b: InspectedBone, e: Event) {
  const checked = (e.target as HTMLInputElement).checked
  compose.toggleBone(props.source, b.model_id, checked)
}
</script>

<template>
  <ul class="mesh-list" data-testid="mesh-list">
    <li v-if="armatureLabel" class="armature-row" data-testid="armature-row">
      <span class="icon">⛂</span>
      <span class="name">{{ armatureLabel }}</span>
      <span class="meta">(armature)</span>
    </li>
    <li v-for="m in meshes" :key="m.model_id" class="mesh-row"
        data-testid="mesh-row" :data-mesh-id="m.model_id"
        :data-mesh-name="m.name"
        :data-state="isDropped(m.model_id) ? 'dropped' : 'kept'">
      <input type="checkbox" :checked="!isDropped(m.model_id)"
             @change="onToggle(m, $event)" />
      <span class="icon">▦</span>
      <span class="name">{{ m.name }}</span>
    </li>
    <li v-if="meshes.length === 0 && !armatureLabel" class="empty">
      No meshes found.
    </li>
  </ul>
</template>

<style scoped>
.mesh-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.armature-row, .mesh-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.4rem;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.9rem;
}
.armature-row {
  color: #ccc;
  border-bottom: 1px solid #222;
  margin-bottom: 0.25rem;
}
.armature-row .meta {
  color: #666;
  font-size: 0.75rem;
}
.mesh-row {
  color: #ddd;
  padding-left: 1.5rem;
}
.icon {
  color: #888;
  width: 1rem;
  text-align: center;
}
.mesh-row[data-state="dropped"] .name,
.mesh-row[data-state="dropped"] .icon {
  color: #555;
  text-decoration: line-through;
}
.empty {
  color: #666;
  font-style: italic;
  padding: 0.5rem 0;
}
</style>
