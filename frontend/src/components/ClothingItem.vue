<script setup lang="ts">
import { computed } from 'vue'
import type { ComposedClothing } from '../stores/compose'
import { useComposeStore } from '../stores/compose'
import MeshList from './MeshList.vue'

const props = defineProps<{ clothing: ComposedClothing }>()
const compose = useComposeStore()

const meshCount = computed(() =>
  props.clothing.inspect
    ? props.clothing.inspect.bones.filter((b) => b.type_class === 'Mesh').length
    : 0,
)

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
        donor: <strong>{{ clothing.inspect.donor_id ?? 'unknown' }}</strong>
        <span v-if="clothing.inspect.donor_id">
          (score {{ clothing.inspect.donor_score.toFixed(2) }})
        </span>
        — {{ meshCount }} meshes
      </div>
      <div class="tree-wrap">
        <MeshList :source="clothing" armature-label="Armature" />
      </div>
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
.tree-wrap {
  max-height: 400px;
  overflow-y: auto;
}
</style>
