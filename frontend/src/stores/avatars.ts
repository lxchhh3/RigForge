import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Avatar {
  id: string
  display_name: string
  canonical_roles: string[]
}

const API_BASE = (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/$/, '')

export const useAvatarsStore = defineStore('avatars', () => {
  const avatars = ref<Avatar[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function loadAvatars(): Promise<Avatar[]> {
    loading.value = true
    error.value = null
    try {
      const res = await fetch(`${API_BASE}/api/avatars`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = (await res.json()) as Avatar[]
      avatars.value = data
      return data
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return []
    } finally {
      loading.value = false
    }
  }

  return { avatars, loading, error, loadAvatars }
})
