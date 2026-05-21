<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useAvatarsStore } from '../stores/avatars'
import { useComposeStore } from '../stores/compose'
import ClothingItem from '../components/ClothingItem.vue'
import TargetPanel from '../components/TargetPanel.vue'

const avatarsStore = useAvatarsStore()
const compose = useComposeStore()
const { avatars, loading: avatarsLoading, error: avatarsError } = storeToRefs(avatarsStore)
const { clothings, assembling, results, assembleError, targetId } = storeToRefs(compose)

const newClothingPath = ref('')

onMounted(async () => {
  await avatarsStore.loadAvatars()
  // Auto-pick first avatar if user hasn't chosen
  if (!compose.targetId && avatars.value.length > 0) {
    compose.targetId = avatars.value[0].id
  }
})

watch(avatars, (av) => {
  if (!compose.targetId && av.length > 0) compose.targetId = av[0].id
})

// Whenever the picked target avatar changes, fetch its bone tree so the
// modder can pre-filter its bundled clothing.
watch(targetId, (id) => {
  if (id) compose.loadTargetInspect(id)
}, { immediate: true })

function addClothing() {
  const p = newClothingPath.value.trim()
  if (!p) return
  compose.addClothing(p)
  newClothingPath.value = ''
}

async function runAssemble() {
  await compose.assembleAll()
}
</script>

<template>
  <main class="compose">
    <header>
      <h1>RigForge</h1>
      <p class="tagline">
        Pick a target avatar, add the clothings you want to rig onto it, uncheck parts you don't want, then assemble.
      </p>
    </header>

    <section class="row target-row">
      <label for="target">Target avatar:</label>
      <select id="target" v-model="targetId" data-testid="avatar-picker"
              :disabled="avatarsLoading || avatars.length === 0">
        <option v-for="a in avatars" :key="a.id" :value="a.id">
          {{ a.display_name }} ({{ a.id }})
        </option>
      </select>
      <span v-if="avatarsLoading" class="hint">loading…</span>
      <span v-else-if="avatarsError" class="hint error">{{ avatarsError }}</span>
    </section>

    <TargetPanel />

    <section class="add-clothing">
      <label for="clothing-path">Add clothing FBX:</label>
      <input id="clothing-path" v-model="newClothingPath" type="text"
             placeholder="C:/path/to/clothing.fbx"
             data-testid="clothing-path-input"
             @keyup.enter="addClothing" />
      <button @click="addClothing" data-testid="clothing-add-button">add</button>
    </section>

    <section class="clothings">
      <div v-if="clothings.length === 0" class="empty">
        No clothings added yet.
      </div>
      <ClothingItem v-for="c in clothings" :key="c.path" :clothing="c" />
    </section>

    <section class="assemble-row">
      <button class="primary" :disabled="assembling || clothings.length === 0 || !targetId"
              @click="runAssemble" data-testid="assemble-button">
        {{ assembling ? 'Assembling…' : 'Assemble' }}
      </button>
      <span v-if="assembleError" class="hint error">{{ assembleError }}</span>
    </section>

    <section v-if="results.length > 0" class="results" data-testid="assemble-results">
      <h2>Results</h2>
      <ul>
        <li v-for="r in results" :key="r.id" data-testid="assemble-result">
          <span class="id">{{ r.id }}</span>
          <span class="arrow">→</span>
          <span class="out">{{ r.output_fbx }}</span>
        </li>
      </ul>
    </section>
  </main>
</template>

<style scoped>
.compose {
  max-width: 980px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
}
header { margin-bottom: 1.25rem; }
h1 { margin: 0 0 0.25rem 0; font-size: 2rem; }
.tagline { margin: 0; color: #888; font-size: 0.95rem; }

.row, .add-clothing, .assemble-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.6rem 0;
}
.target-row label, .add-clothing label {
  color: #aaa;
  font-size: 0.9rem;
}
select, input[type="text"] {
  background: #181818;
  color: #ddd;
  border: 1px solid #333;
  padding: 0.35rem 0.5rem;
  border-radius: 4px;
  font-size: 0.9rem;
}
input[type="text"] {
  flex: 1;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}
button {
  background: #2a2a2a;
  color: #ddd;
  border: 1px solid #444;
  padding: 0.35rem 0.85rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
}
button:hover:not(:disabled) {
  border-color: #888;
}
button.primary {
  background: #2a4565;
  border-color: #3a5a85;
  font-weight: 600;
}
button.primary:hover:not(:disabled) {
  background: #355075;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.hint { font-size: 0.8rem; color: #888; }
.hint.error { color: #d88; }
.clothings {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin: 1rem 0;
}
.empty {
  color: #666;
  font-style: italic;
  padding: 1rem 0;
}
.results {
  margin-top: 1.5rem;
  border-top: 1px solid #2a2a2a;
  padding-top: 1rem;
}
.results h2 { font-size: 1.1rem; margin: 0 0 0.5rem 0; }
.results ul { list-style: none; margin: 0; padding: 0; }
.results li {
  display: flex;
  gap: 0.6rem;
  padding: 0.25rem 0;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.85rem;
}
.results .id { color: #ddd; min-width: 12rem; }
.results .arrow { color: #555; }
.results .out { color: #8ab; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
