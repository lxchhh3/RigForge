import { createRouter, createWebHistory } from 'vue-router'
import Compose from '../views/Compose.vue'
import History from '../views/History.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'compose', component: Compose },
    { path: '/history', name: 'history', component: History },
  ],
})
