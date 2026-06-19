<template>
  <div class="min-h-screen" style="background-color: #f0f4f8;">
    <!-- 헤더 -->
    <header class="bg-white border-b border-gray-200 px-6 py-3 flex justify-between items-center">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg flex items-center justify-center" style="background-color: #1e3a5f;">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3"/>
          </svg>
        </div>
        <span class="font-semibold text-gray-800">법령 RAG 검색 시스템</span>
      </div>
      <div class="flex items-center gap-3">
        <span v-if="auth.isAuth" class="text-sm text-gray-500">{{ auth.username }} 님</span>
        <button v-if="auth.isAuth" @click="logout"
          class="text-sm px-3 py-1.5 border border-gray-200 rounded-lg hover:bg-gray-50 text-gray-600 transition">
          로그아웃
        </button>
        <router-link v-else to="/login"
          class="text-sm px-3 py-1.5 rounded-lg text-white transition"
          style="background-color: #1e3a5f;">
          로그인
        </router-link>
      </div>
    </header>

    <div class="max-w-4xl mx-auto p-6 space-y-4">

      <!-- 법령 선택 카테고리 탭 -->
      <div class="bg-white rounded-2xl border border-gray-200 overflow-hidden">
        <!-- 카테고리 탭 -->
        <div class="flex border-b border-gray-200 overflow-x-auto">
          <button v-for="cat in Object.keys(availableLaws)" :key="cat"
            @click="activeCategory = cat"
            :class="['px-4 py-3 text-sm font-medium whitespace-nowrap transition border-b-2',
                     activeCategory === cat
                       ? 'border-b-2 text-white'
                       : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50']"
            :style="activeCategory === cat ? 'background-color: #1e3a5f; border-color: #1e3a5f;' : ''">
            {{ cat }}
          </button>
        </div>

        <!-- 법령 버튼 -->
        <div class="p-4">
          <div class="flex items-center gap-2 mb-3">
            <span class="text-xs text-gray-400">미선택 시 전체 검색 · 복수 선택 가능</span>
            <button v-if="selectedLaws.length" @click="selectedLaws = []"
              class="text-xs text-red-400 hover:text-red-600 ml-auto">
              선택 초기화
            </button>
          </div>
          <div class="flex flex-wrap gap-2">
            <button v-for="law in availableLaws[activeCategory]" :key="law"
              @click="toggleLaw(law)"
              :class="['px-3 py-1.5 rounded-full text-sm border transition',
                       selectedLaws.includes(law)
                         ? 'text-white border-transparent'
                         : 'text-gray-600 border-gray-200 hover:border-gray-400 bg-white']"
              :style="selectedLaws.includes(law) ? 'background-color: #1e3a5f;' : ''">
              {{ law }}
            </button>
          </div>
          <!-- 선택된 법령 표시 -->
          <div v-if="selectedLaws.length" class="mt-3 pt-3 border-t border-gray-100">
            <span class="text-xs text-gray-400 mr-2">선택됨:</span>
            <span v-for="law in selectedLaws" :key="law"
              class="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full mr-1"
              style="background-color: #e8eef5; color: #1e3a5f;">
              {{ law }}
              <button @click="toggleLaw(law)" class="hover:opacity-70">×</button>
            </span>
          </div>
        </div>
      </div>

      <!-- 질문 입력 -->
      <div class="bg-white rounded-2xl border border-gray-200 p-5">
        <p class="text-sm font-semibold text-gray-700 mb-3">질문하기</p>
        <div class="flex gap-2">
          <input v-model="question" @keydown.enter="ask"
            placeholder="법령에 관해 궁금한 점을 질문해주세요..."
            class="flex-1 border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:border-transparent"
            style="--tw-ring-color: #1e3a5f;"/>
          <button @click="ask" :disabled="loading || !question.trim()"
            class="px-6 py-2.5 rounded-xl text-sm font-medium text-white transition disabled:opacity-40"
            style="background-color: #1e3a5f;">
            {{ loading ? '검색 중...' : '검색' }}
          </button>
        </div>
        <div v-if="loading" class="mt-3 flex items-center gap-2 text-sm text-gray-400">
          <svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
          </svg>
          AI가 법령을 검색하고 있습니다...
        </div>
      </div>

      <!-- 답변 -->
      <div v-if="result" class="bg-white rounded-2xl border border-gray-200 p-5 space-y-4">
        <p class="text-sm font-semibold text-gray-700">답변</p>
        <div class="rounded-xl p-4 text-sm text-gray-700 whitespace-pre-wrap leading-relaxed"
          style="background-color: #f8fafc; border: 1px solid #e8eef5;">
          {{ result.answer }}
        </div>
        <div v-if="result.sources?.length">
          <p class="text-xs font-semibold text-gray-500 mb-2">참고 조문</p>
          <div v-for="(s, i) in result.sources" :key="i"
            class="border border-gray-100 rounded-xl p-3 mb-2">
            <div class="flex gap-2 mb-2">
              <span class="text-xs px-2 py-0.5 rounded-full font-medium"
                style="background-color: #e8eef5; color: #1e3a5f;">{{ s.law_name }}</span>
              <span class="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
                {{ s.article_no }} {{ s.article_title }}
              </span>
            </div>
            <p class="text-xs text-gray-500 leading-relaxed">{{ s.content?.slice(0, 200) }}...</p>
          </div>
        </div>
        <p v-if="!auth.isAuth" class="text-xs text-gray-400">
          💡 <router-link to="/login" class="hover:underline" style="color: #1e3a5f;">로그인</router-link>하면 질의 히스토리가 저장됩니다.
        </p>
      </div>

      <!-- 히스토리 -->
      <div v-if="auth.isAuth && history.length" class="bg-white rounded-2xl border border-gray-200 p-5">
        <p class="text-sm font-semibold text-gray-700 mb-3">질의 히스토리</p>
        <div v-for="item in history" :key="item.id" class="border-b border-gray-100 last:border-0 py-3 cursor-pointer hover:bg-gray-50 rounded-lg px-2 transition"
          @click="question = item.question">
          <p class="text-sm text-gray-700">Q: {{ item.question }}</p>
          <p class="text-xs text-gray-400 mt-0.5">{{ item.created_at?.slice(0,10) }}</p>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import axios from 'axios'

const http = axios.create({ baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8001' })
http.interceptors.request.use(c => {
  const t = localStorage.getItem('access_token')
  if (t) c.headers.Authorization = `Bearer ${t}`
  return c
})

const auth           = useAuthStore()
const router         = useRouter()
const availableLaws  = ref({})
const activeCategory = ref('')
const selectedLaws   = ref([])
const question       = ref('')
const loading        = ref(false)
const result         = ref(null)
const history        = ref([])

onMounted(async () => {
  const res = await http.get('/api/v1/laws')
  const laws = res.data.data || res.data
  const allLaws = Object.values(laws).flat()
  availableLaws.value = { "전체": allLaws, ...laws }
  activeCategory.value = Object.keys(availableLaws.value)[0] || ''
  if (auth.isAuth) loadHistory()
})

function toggleLaw(law) {
  const idx = selectedLaws.value.indexOf(law)
  if (idx >= 0) selectedLaws.value.splice(idx, 1)
  else selectedLaws.value.push(law)
}

async function ask() {
  if (!question.value.trim()) return
  loading.value = true
  result.value = null
  try {
    const res = await http.post('/api/v1/qa/ask', {
      question: question.value,
      law_names: selectedLaws.value.length ? selectedLaws.value : []
    })
    result.value = res.data.data || res.data
    if (auth.isAuth) loadHistory()
  } catch(e) {
    alert(e.response?.data?.detail || '오류가 발생했습니다.')
  } finally {
    loading.value = false
  }
}

async function loadHistory() {
  const res = await http.get('/api/v1/qa/history')
  history.value = res.data.data?.items || []
}

function logout() {
  auth.logout()
  router.push('/login')
}
</script>
