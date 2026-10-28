<script setup lang="ts">
import { ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useComposeStore } from '../stores/compose'

// Load a saved Unity scene (.unity) into the page: the backend reads what
// kevin set up in Unity (outfits and props on the avatar, hidden meshes,
// blendshape values, moved bones) and the store REPLACES the compose with it.
// The report lists what the read did and what it could not carry.
const compose = useComposeStore()
const { unityScene, unitySceneLoading, unitySceneError } = storeToRefs(compose)

const scenePath = ref(unityScene.value?.path ?? '')

async function load() {
  if (!scenePath.value.trim()) return
  const ok = await compose.loadUnityScene(scenePath.value)
  if (ok && unityScene.value) scenePath.value = unityScene.value.path
}
</script>

<template>
  <section class="unity-scene" data-testid="unity-scene">
    <div class="row">
      <label for="unity-scene-path">Load Unity scene:</label>
      <input id="unity-scene-path" v-model="scenePath" type="text"
             placeholder="D:/unity_proj/Project/Assets/Scene.unity"
             data-testid="unity-scene-input"
             @keyup.enter="load" />
      <button :disabled="unitySceneLoading || !scenePath.trim()" @click="load"
              data-testid="unity-scene-load"
              title="Replaces the target and clothings on this page with the scene's setup">
        {{ unitySceneLoading ? 'Reading…' : 'load' }}
      </button>
    </div>
    <p class="hint">
      Save the scene in Unity first — this reads the saved file. Loading replaces
      what is on this page.
    </p>
    <div v-if="unitySceneError" class="error" data-testid="unity-scene-error">
      Could not read the scene: {{ unitySceneError }}
    </div>
    <div v-if="unityScene" class="report" data-testid="unity-scene-report">
      <div class="report-head">
        Read from <span class="path">{{ unityScene.path }}</span>
      </div>
      <ul class="notes">
        <li v-for="(n, i) in unityScene.notes" :key="`n${i}`" data-testid="unity-scene-note">{{ n }}</li>
      </ul>
      <template v-if="unityScene.skipped.length">
        <div class="skipped-head">Not loaded:</div>
        <ul class="skipped">
          <li v-for="(s, i) in unityScene.skipped" :key="`s${i}`" data-testid="unity-scene-skipped"
              :title="s.path">
            <strong>{{ s.name }}</strong> — {{ s.reason }}
          </li>
        </ul>
      </template>
    </div>
  </section>
</template>

<style scoped>
.unity-scene {
  border: 1px solid #2a3a4a;
  background: #121820;
  border-radius: 6px;
  padding: 0.6rem 0.8rem;
  margin: 0.4rem 0 0.8rem 0;
}
.row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
label { color: #aaa; font-size: 0.9rem; white-space: nowrap; }
input[type="text"] {
  flex: 1;
  background: #181818;
  color: #ddd;
  border: 1px solid #333;
  padding: 0.35rem 0.5rem;
  border-radius: 4px;
  font-size: 0.9rem;
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
button:disabled { opacity: 0.5; cursor: not-allowed; }
.hint { margin: 0.35rem 0 0 0; color: #777; font-size: 0.78rem; }
.error { margin-top: 0.5rem; color: #d88; font-size: 0.85rem; }
.report { margin-top: 0.6rem; font-size: 0.82rem; color: #bbb; }
.report-head { color: #999; margin-bottom: 0.25rem; }
.path { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; color: #8ab; }
.notes, .skipped { margin: 0.2rem 0 0.2rem 1.1rem; padding: 0; }
.notes li, .skipped li { margin: 0.12rem 0; }
.skipped-head { color: #d9a85b; margin-top: 0.4rem; }
.skipped li { color: #d9b98b; }
</style>
