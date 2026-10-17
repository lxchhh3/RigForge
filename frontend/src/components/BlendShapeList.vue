<script setup lang="ts">
import { computed, ref } from 'vue'
import type { BoneTreeSource, InspectedChannel } from '../stores/compose'
import { useComposeStore } from '../stores/compose'

const props = defineProps<{
  source: BoneTreeSource
  // Display label rendered as the section header (e.g. "Maya blendshapes").
  // Optional — the parent panel typically supplies its own h-tag, in which
  // case this is left blank.
  heading?: string
}>()

const compose = useComposeStore()

// Filter box. Maya/Moe ship 200–400 channels, mostly with the same Korean
// default prefix ("평면."), so a substring filter is the practical way for
// the modder to find a specific morph. Kept local to the component — no
// store state needed; reset is the user toggling the filter input.
const filter = ref('')

const channels = computed<InspectedChannel[]>(() => {
  if (!props.source.inspect) return []
  const all = props.source.inspect.blend_shape_channels ?? []
  const needle = filter.value.trim().toLowerCase()
  if (!needle) return all
  return all.filter((c) => c.name.toLowerCase().includes(needle))
})

function isDropped(id: number): boolean {
  return props.source.droppedChannels.has(id)
}

function onToggle(c: InspectedChannel, e: Event) {
  const checked = (e.target as HTMLInputElement).checked
  compose.toggleChannel(props.source, c.channel_id, checked)
}

// The slider value: prefer the user override; fall back to the on-disk
// DeformPercent so the slider opens at the file's actual baseline.
function valueFor(c: InspectedChannel): number {
  const override = props.source.channelOverrides.get(c.channel_id)
  return override !== undefined ? override : c.deform_percent
}

function hasOverride(c: InspectedChannel): boolean {
  return props.source.channelOverrides.has(c.channel_id)
}

// Two decimals is the precision the modder asked for (2026-09-08): the slider
// alone was integer-only. Both the slider and the typed field go through
// this, so a value is always 0..100 with at most two decimals.
function normalize(raw: number): number {
  const clamped = Math.max(0, Math.min(100, raw))
  return Math.round(clamped * 100) / 100
}

// "75" not "75.00", "12.34" not "12.340000001": what the file will carry.
function display(v: number): string {
  return String(normalize(v))
}

function apply(c: InspectedChannel, raw: number) {
  if (!Number.isFinite(raw)) return // empty / half-typed field ("12.")
  const value = normalize(raw)
  if (value === c.deform_percent) {
    // Snapping back to the on-disk value clears the override entirely — keeps
    // the assemble payload free of no-op entries.
    compose.clearChannelOverride(props.source, c.channel_id)
  } else {
    compose.setChannelOverride(props.source, c.channel_id, value)
  }
}

function onSlide(c: InspectedChannel, e: Event) {
  apply(c, (e.target as HTMLInputElement).valueAsNumber)
}

function onType(c: InspectedChannel, e: Event) {
  apply(c, (e.target as HTMLInputElement).valueAsNumber)
}

// On blur/Enter, rewrite the field with the normalized value so "12.346" or
// "250" does not linger in the box after it was rounded/clamped.
function onTypeCommit(c: InspectedChannel, e: Event) {
  const el = e.target as HTMLInputElement
  apply(c, el.valueAsNumber)
  el.value = display(valueFor(c))
}
</script>

<template>
  <section class="bs-section" data-testid="blendshape-list">
    <header v-if="heading" class="bs-heading">{{ heading }}</header>
    <input
      v-if="source.inspect && (source.inspect.blend_shape_channels?.length ?? 0) > 0"
      v-model="filter"
      type="search"
      placeholder="Filter morphs…"
      class="bs-filter"
      data-testid="blendshape-filter"
    />
    <ul class="bs-list">
      <li v-for="c in channels" :key="c.channel_id" class="bs-row"
          data-testid="blendshape-row" :data-channel-id="c.channel_id"
          :data-channel-name="c.name" :data-owner-mesh="c.owner_mesh ?? ''"
          :data-state="isDropped(c.channel_id) ? 'dropped' : 'kept'"
          :data-has-override="hasOverride(c) ? '1' : '0'">
        <input type="checkbox" :checked="!isDropped(c.channel_id)"
               @change="onToggle(c, $event)" />
        <span class="icon">◈</span>
        <span v-if="c.owner_mesh" class="mesh" data-testid="blendshape-owner-mesh"
              :title="`mesh: ${c.owner_mesh}`">{{ c.owner_mesh }}</span>
        <span class="name">{{ c.name }}</span>
        <span v-if="c.owner_name" class="meta">{{ c.owner_name }}</span>
        <input
          type="range" min="0" max="100" step="0.01"
          :disabled="isDropped(c.channel_id)"
          :value="valueFor(c)"
          @input="onSlide(c, $event)"
          class="slider"
          data-testid="blendshape-slider"
        />
        <input
          type="number" min="0" max="100" step="0.01"
          :disabled="isDropped(c.channel_id)"
          :value="display(valueFor(c))"
          @input="onType(c, $event)"
          @change="onTypeCommit(c, $event)"
          class="number"
          :class="{ overridden: hasOverride(c) }"
          title="Type a value (0–100, two decimals)"
          data-testid="blendshape-number"
        />
        <span class="value" :class="{ overridden: hasOverride(c) }"
              data-testid="blendshape-value">{{ display(valueFor(c)) }}</span>
      </li>
      <li v-if="channels.length === 0" class="empty">
        <template v-if="filter">No matches for "{{ filter }}".</template>
        <template v-else>No blendshape channels.</template>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.bs-section {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.bs-heading {
  color: #ccc;
  font-size: 0.85rem;
  font-weight: 600;
  border-bottom: 1px solid #222;
  padding-bottom: 0.25rem;
}
.bs-filter {
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  color: #ddd;
  padding: 0.25rem 0.4rem;
  font-family: inherit;
  font-size: 0.85rem;
}
.bs-list {
  list-style: none;
  margin: 0;
  padding: 0;
  max-height: 12rem;
  overflow-y: auto;
}
.bs-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.15rem 0.4rem 0.15rem 0.75rem;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.85rem;
  color: #ddd;
}
.bs-row .icon {
  color: #888;
  width: 1rem;
  text-align: center;
}
.bs-row .mesh {
  color: #8ab;
  background: #1c2530;
  border: 1px solid #2a3a48;
  border-radius: 3px;
  padding: 0 0.35rem;
  font-size: 0.7rem;
  max-width: 8rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bs-row .meta {
  color: #666;
  font-size: 0.7rem;
}
.bs-row .slider {
  margin-left: auto;
  width: 7rem;
  accent-color: #6a8;
}
.bs-row .slider:disabled {
  opacity: 0.3;
}
.bs-row .number {
  width: 4.2rem;
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  color: #ddd;
  padding: 0.1rem 0.3rem;
  font-family: inherit;
  font-size: 0.75rem;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.bs-row .number:disabled {
  opacity: 0.3;
}
.bs-row .number.overridden {
  color: #cd8;
  border-color: #564;
}
.bs-row .value {
  color: #888;
  font-size: 0.75rem;
  width: 2.5rem;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.bs-row .value.overridden {
  color: #cd8;
  font-weight: 600;
}
.bs-row[data-state="dropped"] .name,
.bs-row[data-state="dropped"] .icon {
  color: #555;
  text-decoration: line-through;
}
.empty {
  color: #666;
  font-style: italic;
  padding: 0.4rem 0;
  font-size: 0.85rem;
}
</style>
