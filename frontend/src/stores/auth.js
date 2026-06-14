import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

const http = axios.create({ baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8001' })

export const useAuthStore = defineStore('auth', () => {
  const token    = ref(localStorage.getItem('access_token') || '')
  const username = ref(localStorage.getItem('username') || '')
  const isAuth   = computed(() => !!token.value)

  async function signup(payload) {
    const { data } = await http.post('/api/v1/auth/signup', payload)
    _save(data.data || data)
  }
  async function login(payload) {
    const { data } = await http.post('/api/v1/auth/login', payload)
    _save(data.data || data)
  }
  function logout() {
    token.value = ''; username.value = ''
    localStorage.removeItem('access_token')
    localStorage.removeItem('username')
  }
  function _save(d) {
    const at = d.access_token || d.accessToken
    const un = d.username
    if (at) { token.value = at; localStorage.setItem('access_token', at) }
    if (un) { username.value = un; localStorage.setItem('username', un) }
  }
  return { token, username, isAuth, signup, login, logout }
})
