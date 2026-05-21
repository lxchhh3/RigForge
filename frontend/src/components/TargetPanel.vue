<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useComposeStore } from '../stores/compose'
import MeshList from './MeshList.vue'

const compose = useComposeStore()
const { target } = storeToRefs(compose)

const meshCount = computed(() =>
  target.value.inspect
    ? target.value.inspect.bones.filter((b) => b.type_class === 'Mesh').length
    : 0,
)
</script>

<template>
  <section v-if="target.avatarId" class="target-panel" data-testid="target-panel">
    <header class="head">
      <span class="title">Target avatar bones</span>
      <span class="hint">
        Uncheck Maya's bundled-clothing chains here so the new clothing doesn't
        sit on top of them.
      </span>
    </header>

    <div v-if="target.loading" class="status">Inspecting target…</div>
    <div v-else-if="target.error" class="status error">
      Target inspect failed: {{ target.error }}
    </div>
    <template v-else-if="target.inspect">
      <div class="meta">
        <strong>{{ target.avatarId }}</strong> — {{ meshCount }} meshes
      </div>
      <div class="tree-wrap">
        <MeshList :source="target" armature-label="Armature" />
      </div>
    </template>
  </section>
</template>

<style scoped>
.target-panel {
  border: 1px solid #2a2a2a;
  border-radius: 6px;
  padding: 0.85rem 1rem;
  margin: 0.5rem 0 1rem 0;
}
.head {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  margin-bottom: 0.5rem;
}
.title {
  color: #ddd;
  font-weight: 600;
  font-size: 0.95rem;
}
.hint {
  color: #888;
  font-size: 0.8rem;
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
  max-height: 320px;
  overflow-y: auto;
}
</style>
