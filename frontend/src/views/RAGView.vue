<template>
  <div class="min-h-screen bg-gray-50">
    <div class="max-w-6xl mx-auto p-6 grid grid-cols-3 gap-6">
      <!-- 문서 목록 -->
      <aside class="col-span-1 bg-white rounded-2xl shadow p-5">
        <h2 class="font-bold text-gray-800 mb-3">내 문서</h2>
        <label class="block w-full cursor-pointer mb-3">
          <span class="block w-full py-2 bg-blue-600 text-white text-center rounded-xl text-sm hover:bg-blue-700">
            + 문서 업로드
          </span>
          <input type="file" accept=".pdf,.txt,.docx" class="hidden" @change="uploadFile"/>
        </label>
        <div v-if="uploading" class="text-xs text-blue-500 mb-2 animate-pulse">처리 중...</div>
        <div v-for="doc in documents" :key="doc.id"
          @click="toggleDoc(doc.id)"
          :class="['cursor-pointer px-3 py-2 rounded-lg text-sm mb-1 flex justify-between',
                   selectedDocs.includes(doc.id) ? 'bg-blue-100 text-blue-800' : 'hover:bg-gray-100']">
          <span class="truncate">{{ doc.filename }}</span>
          <span :class="['text-xs px-1.5 py-0.5 rounded',
                         doc.status==='READY' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700']">
            {{ doc.status === 'READY' ? '준비' : '처리중' }}
          </span>
        </div>
      </aside>
      <!-- Q&A 영역 -->
      <section class="col-span-2 bg-white rounded-2xl shadow p-5 flex flex-col">
        <h2 class="font-bold text-gray-800 mb-4">문서 기반 질의응답</h2>
        <div class="flex-1 overflow-y-auto space-y-4 mb-4 max-h-96">
          <div v-for="qa in history" :key="qa.id" class="space-y-2">
            <div class="bg-blue-50 rounded-xl px-4 py-2 text-sm text-blue-800">
              <strong>Q:</strong> {{ qa.question }}
            </div>
            <div class="bg-gray-50 rounded-xl px-4 py-3 text-sm text-gray-800">
              <strong>A:</strong> {{ qa.answer }}
              <div v-if="qa.sources?.length" class="mt-2 space-y-1">
                <p class="text-xs text-gray-500 font-semibold">출처:</p>
                <div v-for="(s,i) in qa.sources" :key="i"
                  class="text-xs bg-white border rounded px-2 py-1 text-gray-600">
                  📄 {{ s.content?.slice(0,80) }}...
                </div>
              </div>
            </div>
          </div>
          <div v-if="loading" class="text-sm text-gray-400 animate-pulse">AI가 답변을 생성 중입니다...</div>
        </div>
        <div class="flex gap-2">
          <input v-model="question" @keydown.enter="ask" placeholder="문서에 대해 질문하세요..."
            class="flex-1 border rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"/>
          <button @click="ask" :disabled="loading || !question.trim()"
            class="px-5 py-2 bg-blue-600 text-white rounded-xl text-sm disabled:opacity-40 hover:bg-blue-700">
            질문
          </button>
        </div>
      </section>
    </div>
  </div>
</template>
<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
const http = axios.create({ baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8001' })
http.interceptors.request.use(c => { const t = localStorage.getItem('access_token'); if(t) c.headers.Authorization=`Bearer ${t}`; return c })
const documents   = ref([])
const selectedDocs= ref([])
const history     = ref([])
const question    = ref('')
const loading     = ref(false)
const uploading   = ref(false)
onMounted(async () => { documents.value = await http.get('/api/v1/documents').then(r=>r.data) })
function toggleDoc(id) {
  const idx = selectedDocs.value.indexOf(id)
  if(idx>=0) selectedDocs.value.splice(idx,1)
  else selectedDocs.value.push(id)
}
async function uploadFile(e) {
  const file = e.target.files[0]; if(!file) return
  uploading.value = true
  const fd = new FormData(); fd.append('file', file)
  try {
    const res = await http.post('/api/v1/documents/upload', fd)
    documents.value.unshift(res.data)
  } finally { uploading.value = false }
}
async function ask() {
  if(!question.value.trim()) return
  loading.value = true
  const q = question.value; question.value = ''
  try {
    const res = await http.post('/api/v1/qa/ask', { question:q, document_ids: selectedDocs.value.length ? selectedDocs.value : null })
    history.value.push({ id:Date.now(), question:q, answer:res.data.answer, sources:res.data.sources })
  } finally { loading.value = false }
}
</script>
