<script setup lang="ts">
import { onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useAssembliesStore } from '../stores/assemblies'
import AssemblyList from '../components/AssemblyList.vue'

const store = useAssembliesStore()
const { recent, loading, error } = storeToRefs(store)

onMounted(() => {
  store.loadRecent()
})
</script>

<template>
  <main class="home">
    <header>
      <h1>RigForge</h1>
      <p class="tagline">Booth-asset assembly for MHWilds clothing mods.</p>
    </header>
    <div v-if="loading" data-testid="loading" class="loading">Loading…</div>
    <div v-else-if="error" data-testid="error" class="error">
      Couldn't reach the API: {{ error }}
    </div>
    <AssemblyList v-else :runs="recent" />
  </main>
</template>

<style scoped>
.home {
  max-width: 880px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
}

header {
  margin-bottom: 1.5rem;
}

h1 {
  margin: 0 0 0.25rem 0;
  font-size: 2rem;
}

.tagline {
  margin: 0;
  color: #888;
}

.loading,
.error {
  padding: 1rem 1.25rem;
  border: 1px solid #2a2a2a;
  border-radius: 6px;
  color: #aaa;
}

.error {
  border-color: #5a2a2a;
  color: #d88;
}
</style>
