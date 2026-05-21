import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface DonorCandidate {
  id: string
  score: number
}

export interface AssemblyInput {
  donor_id: string
  target_id: string
  donor_score: number
  candidates: DonorCandidate[]
}

export interface AssemblyCache {
  key: string
  hit: boolean
}

export interface EditPlanSummary {
  drops: number[]
  renames: Record<string, string>
  n_drops: number
  n_renames: number
}

export interface ValidatorWarning {
  rule: string
  message: string
  bone_ids: number[]
}

export interface DecisionsSummary {
  kept: number
  dropped: number
  llm_model_id: string | null
}

export interface AssemblyRun {
  id: string
  schema_version: number
  generated_at: string
  input: AssemblyInput
  output_fbx: string
  cache: AssemblyCache
  edit_plan: EditPlanSummary
  warnings: ValidatorWarning[]
  decisions_summary: DecisionsSummary
  notes: string[]
}

const API_BASE = (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/$/, '')

export const useAssembliesStore = defineStore('assemblies', () => {
  const recent = ref<AssemblyRun[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function loadRecent(): Promise<AssemblyRun[]> {
    loading.value = true
    error.value = null
    try {
      const res = await fetch(`${API_BASE}/api/assemblies`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = (await res.json()) as AssemblyRun[]
      recent.value = data
      return data
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      recent.value = []
      return []
    } finally {
      loading.value = false
    }
  }

  return { recent, loading, error, loadRecent }
})
