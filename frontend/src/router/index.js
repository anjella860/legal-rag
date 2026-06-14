import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const routes = [
  { path: '/login', component: () => import('@/views/LoginView.vue') },
  { path: '/',      component: () => import('@/views/MainView.vue'), meta: { requiresAuth: true } },
]

const router = createRouter({ history: createWebHistory(), routes })
router.beforeEach(to => {
  const auth = useAuthStore()
  if (to.meta.requiresAuth && !auth.isAuth) return '/login'
})
export default router
