<script setup lang="ts">
import type { AssemblyRun } from '../stores/assemblies'

defineProps<{
  runs: AssemblyRun[]
}>()
</script>

<template>
  <section class="assembly-list" data-testid="recent-assemblies">
    <h2>Recent Assembly Runs</h2>
    <div v-if="runs.length === 0" class="empty" data-testid="empty-state">
      No recent runs yet.
    </div>
    <ul v-else>
      <li v-for="run in runs" :key="run.id" data-testid="assembly-item">
        <span class="id">{{ run.id }}</span>
        <span class="target">{{ run.input.target_id }}</span>
        <span class="donor">from {{ run.input.donor_id }}</span>
        <span class="kept">kept {{ run.decisions_summary.kept }}</span>
        <span class="dropped">dropped {{ run.decisions_summary.dropped }}</span>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.assembly-list {
  border: 1px solid #2a2a2a;
  border-radius: 6px;
  padding: 1rem 1.25rem;
}

.assembly-list h2 {
  margin: 0 0 0.75rem 0;
  font-size: 1.1rem;
}

.empty {
  color: #888;
  font-style: italic;
}

ul {
  list-style: none;
  margin: 0;
  padding: 0;
}

li {
  display: flex;
  gap: 0.75rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid #1f1f1f;
}

li:last-child {
  border-bottom: none;
}
</style>
