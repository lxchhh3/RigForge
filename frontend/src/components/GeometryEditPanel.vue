<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ComposedClothing, TasteEditResult } from '../stores/compose'
import { useComposeStore } from '../stores/compose'

const props = defineProps<{ clothing: ComposedClothing }>()
const compose = useComposeStore()

const open = ref(false)
const cutPct = ref(0.15)
const loading = ref(false)
const error = ref<string | null>(null)
const result = ref<TasteEditResult | null>(null)

// Mesh names (the user picks the footwear mesh). Default to one that looks
// like footwear, else the first mesh.
const meshNames = computed<string[]>(() =>
  (props.clothing.inspect?.bones ?? [])
    .filter((b) => b.type_class === 'Mesh')
    .map((b) => b.name),
)
const FOOTWEAR = /heel|shoe|sole|boot|foot|sandal|loafer/i
const selectedMesh = ref<string>('')
function defaultMesh(): string {
  const m = meshNames.value
  return m.find((n) => FOOTWEAR.test(n)) ?? m[0] ?? ''
}

function toggle() {
  open.value = !open.value
  if (open.value && !selectedMesh.value) selectedMesh.value = defaultMesh()
}

async function preview() {
  if (!selectedMesh.value) return
  loading.value = true
  error.value = null
  result.value = null
  try {
    result.value = await compose.tasteEdit(props.clothing, selectedMesh.value, cutPct.value)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

function apply() {
  if (!result.value || !result.value.weights_ok) return
  compose.applyEditedPath(props.clothing, result.value.edited_path)
  result.value = null
  open.value = false
}
</script>

<template>
  <div class="geom-edit" data-testid="taste-edit">
    <button class="toggle" data-testid="taste-edit-toggle" @click="toggle">
      {{ open ? '▾' : '▸' }} Cut sole height
      <span v-if="clothing.edited" class="edited-tag" data-testid="taste-edit-applied">edited ✓</span>
    </button>

    <div v-if="open" class="body">
      <div class="controls">
        <label>
          mesh
          <select v-model="selectedMesh" data-testid="taste-edit-mesh">
            <option v-for="n in meshNames" :key="n" :value="n">{{ n }}</option>
          </select>
        </label>
        <label>
          cut
          <input
            type="range" min="0.02" max="0.5" step="0.01"
            v-model.number="cutPct" data-testid="taste-edit-cutpct"
          />
          <span class="pct">{{ Math.round(cutPct * 100) }}%</span>
        </label>
        <button
          class="preview-btn" data-testid="taste-edit-preview"
          :disabled="loading || !selectedMesh" @click="preview"
        >
          {{ loading ? 'Cutting…' : 'Preview cut' }}
        </button>
      </div>

      <p class="hint">
        Removes the bottom {{ Math.round(cutPct * 100) }}% of the chosen mesh — a
        real cut (no texture squashing). The clothing in isolation; assemble is
        unaffected until you apply.
      </p>

      <div v-if="error" class="status error">Cut failed: {{ error }}</div>

      <div v-if="result" class="result" data-testid="taste-edit-result">
        <div class="imgs">
          <figure>
            <img v-if="result.preview_before" :src="result.preview_before"
                 data-testid="taste-edit-before" alt="before" />
            <figcaption>before</figcaption>
          </figure>
          <figure>
            <img v-if="result.preview_after" :src="result.preview_after"
                 data-testid="taste-edit-after" alt="after" />
            <figcaption>after</figcaption>
          </figure>
        </div>
        <div class="report">
          <span>{{ result.mesh }}: {{ result.verts_before }} → {{ result.verts_after }} verts
            (−{{ result.verts_removed }})</span>
          <span
            class="weights" data-testid="taste-edit-weights"
            :class="result.weights_ok ? 'ok' : 'bad'"
          >{{ result.weights_ok ? 'weights intact ✓' : 'weights broke ✗' }}</span>
        </div>
        <button
          class="apply-btn" data-testid="taste-edit-apply"
          :disabled="!result.weights_ok" @click="apply"
        >Use edited clothing</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.geom-edit {
  margin-top: 0.6rem;
  padding-top: 0.5rem;
  border-top: 1px solid #1a1a1a;
}
.toggle {
  background: none;
  border: none;
  color: #bbb;
  cursor: pointer;
  font-size: 0.85rem;
  padding: 0.1rem 0;
}
.toggle:hover { color: #ddd; }
.edited-tag {
  color: #7ec77e;
  font-size: 0.75rem;
  margin-left: 0.5rem;
}
.body { padding: 0.4rem 0 0.2rem; }
.controls {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  flex-wrap: wrap;
  font-size: 0.8rem;
  color: #aaa;
}
.controls label { display: inline-flex; align-items: center; gap: 0.35rem; }
.controls select {
  background: #1c1c1c; color: #ddd; border: 1px solid #444;
  border-radius: 3px; padding: 0.1rem 0.3rem;
}
.pct { color: #ddd; min-width: 2.5em; }
.preview-btn, .apply-btn {
  background: #243; border: 1px solid #3a5; color: #cfe;
  padding: 0.2rem 0.7rem; border-radius: 3px; cursor: pointer; font-size: 0.8rem;
}
.preview-btn:disabled, .apply-btn:disabled {
  opacity: 0.5; cursor: default;
}
.apply-btn { margin-top: 0.5rem; }
.hint { color: #777; font-size: 0.75rem; margin: 0.4rem 0 0.2rem; }
.status.error { color: #d88; font-size: 0.8rem; }
.result { margin-top: 0.5rem; }
.imgs { display: flex; gap: 0.75rem; }
.imgs figure { margin: 0; text-align: center; }
.imgs img {
  width: 130px; height: auto; border: 1px solid #333; background: #222;
}
.imgs figcaption { color: #888; font-size: 0.72rem; }
.report {
  display: flex; gap: 1rem; align-items: center;
  font-size: 0.8rem; color: #aaa; margin-top: 0.4rem;
}
.weights.ok { color: #7ec77e; }
.weights.bad { color: #d88; }
</style>
