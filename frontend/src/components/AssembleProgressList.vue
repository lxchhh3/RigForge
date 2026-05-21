<script setup lang="ts">
import { computed } from 'vue'
import { storeToRefs } from 'pinia'
import { ASSEMBLE_PHASES, useComposeStore, type AssembleProgress } from '../stores/compose'

const compose = useComposeStore()
const { progress } = storeToRefs(compose)

// Each phase's display label. Kept short — the long notes from the backend
// fill in the per-step detail.
const PHASE_LABELS: Record<string, string> = {
  phase_a: 'Identify',
  phase_b: 'Classify',
  phase_c: 'Merge',
  write: 'Write',
}

function phaseIndex(p: AssembleProgress): number {
  if (p.phase == null) return -1
  return ASSEMBLE_PHASES.indexOf(p.phase)
}

function fillPercent(p: AssembleProgress): number {
  if (p.status === 'succeeded') return 100
  if (p.status === 'failed') return 0
  const idx = phaseIndex(p)
  if (idx < 0) return 4 // small visible sliver while we wait for the first event
  // 4 phases → 25% per phase. We render the bar at the start of each phase
  // (not at the end) so the user sees "we're in phase B" rather than "B done".
  return Math.min(100, (idx + 1) * 25)
}

function basename(path: string): string {
  // Strip Windows + POSIX directory components for a compact display.
  const cleaned = path.replace(/\\/g, '/')
  const slash = cleaned.lastIndexOf('/')
  return slash >= 0 ? cleaned.slice(slash + 1) : cleaned
}

const items = computed(() => progress.value)
</script>

<template>
  <section v-if="items.length > 0" class="progress-panel"
           data-testid="assemble-progress">
    <h2>Progress</h2>
    <ul>
      <li v-for="p in items" :key="p.clothingPath"
          class="progress-row" :data-status="p.status"
          :data-phase="p.phase ?? 'pending'"
          data-testid="assemble-progress-row">
        <div class="head">
          <span class="path" :title="p.clothingPath">{{ basename(p.clothingPath) }}</span>
          <span class="status" data-testid="assemble-progress-status">
            <template v-if="p.status === 'running'">
              {{ p.phase ? PHASE_LABELS[p.phase] : 'starting' }}…
            </template>
            <template v-else-if="p.status === 'succeeded'">done</template>
            <template v-else>failed</template>
          </span>
        </div>
        <div class="bar-wrap">
          <div class="bar" :style="{ width: fillPercent(p) + '%' }"
               data-testid="assemble-progress-bar"></div>
        </div>
        <div class="phase-row">
          <span v-for="(phase, i) in ASSEMBLE_PHASES" :key="phase"
                class="phase-pip" :class="{
                  done: phaseIndex(p) > i || p.status === 'succeeded',
                  active: phaseIndex(p) === i && p.status === 'running',
                }">
            {{ PHASE_LABELS[phase] }}
          </span>
        </div>
        <div class="note" data-testid="assemble-progress-note">
          <template v-if="p.status === 'failed' && p.error">{{ p.error }}</template>
          <template v-else>{{ p.note }}</template>
        </div>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.progress-panel {
  margin: 1rem 0;
  border: 1px solid #2a2a2a;
  border-radius: 6px;
  padding: 0.85rem 1rem;
}
.progress-panel h2 {
  font-size: 1rem;
  margin: 0 0 0.5rem 0;
  color: #ddd;
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.progress-row {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}
.head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
}
.path {
  color: #bbb;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.85rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.status {
  color: #8ab;
  font-size: 0.8rem;
  text-transform: lowercase;
}
.progress-row[data-status="succeeded"] .status { color: #8a8; }
.progress-row[data-status="failed"] .status { color: #d88; }
.bar-wrap {
  height: 4px;
  background: #1a1a1a;
  border-radius: 2px;
  overflow: hidden;
}
.bar {
  height: 100%;
  background: #6a8;
  transition: width 0.3s ease;
}
.progress-row[data-status="failed"] .bar { background: #b66; }
.phase-row {
  display: flex;
  gap: 0.4rem;
  font-size: 0.7rem;
  color: #555;
}
.phase-pip {
  padding: 0 0.3rem;
}
.phase-pip.done { color: #6a8; }
.phase-pip.active { color: #cd8; font-weight: 600; }
.note {
  color: #888;
  font-size: 0.8rem;
  font-style: italic;
}
.progress-row[data-status="failed"] .note { color: #d88; font-style: normal; }
</style>
