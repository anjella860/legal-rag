<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-900 to-gray-900 flex items-center justify-center">
    <div class="w-full max-w-md bg-white rounded-2xl shadow-xl p-8 space-y-6">
      <div class="text-center">
        <h1 class="text-2xl font-bold text-gray-800">{{ appTitle }}</h1>
        <p class="text-sm text-gray-500 mt-1">AI 풀스택 포트폴리오</p>
      </div>
      <div class="flex rounded-lg overflow-hidden border border-gray-200">
        <button v-for="t in ['login','signup']" :key="t" @click="tab=t"
          :class="['flex-1 py-2 text-sm font-medium transition',
                   tab===t ? 'bg-blue-600 text-white' : 'text-gray-500 hover:bg-gray-50']">
          {{ t==='login' ? '로그인' : '회원가입' }}
        </button>
      </div>
      <form @submit.prevent="submit" class="space-y-4">
        <input v-if="tab==='signup'" v-model="form.username" placeholder="사용자명 (영문/숫자/밑줄)"
          class="w-full border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"/>
        <input v-model="form.identifier" :placeholder="tab==='login' ? '이메일 또는 사용자명' : '이메일'"
          class="w-full border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"/>
        <div class="relative">
          <input v-model="form.password" :type="showPw ? 'text' : 'password'" placeholder="비밀번호"
            class="w-full border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"/>
          <button type="button" @click="showPw=!showPw"
            class="absolute right-3 top-2.5 text-gray-400 hover:text-gray-600 text-xs">
            {{ showPw ? '숨기기' : '보기' }}
          </button>
        </div>
        <p v-if="error" class="text-red-500 text-sm bg-red-50 px-3 py-2 rounded-lg">{{ error }}</p>
        <button type="submit" :disabled="loading"
          class="w-full py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-semibold
                 disabled:opacity-50 transition text-sm">
          {{ loading ? '처리 중...' : (tab==='login' ? '로그인' : '회원가입') }}
        </button>
      </form>
    </div>
  </div>
</template>
<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
const props   = defineProps({ appTitle: { type: String, default: 'AI 서비스' } })
const tab     = ref('login')
const showPw  = ref(false)
const loading = ref(false)
const error   = ref('')
const form    = reactive({ username:'', identifier:'', password:'' })
const router  = useRouter()
const auth    = useAuthStore()
async function submit() {
  error.value = ''; loading.value = true
  try {
    if (tab.value === 'login') await auth.login({ identifier: form.identifier, password: form.password })
    else await auth.signup({ username: form.username, email: form.identifier, password: form.password })
    router.push('/')
  } catch(e) {
    error.value = e.response?.data?.detail || '오류가 발생했습니다. 다시 시도해 주세요.'
  } finally { loading.value = false }
}
</script>
