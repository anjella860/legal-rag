<template>
  <div class="min-h-screen bg-gray-50">
    <!-- 헤더 -->
    <header
      class="bg-white shadow-sm px-6 py-4 flex justify-between items-center"
    >
      <h1 class="text-lg font-bold text-gray-800">법령 RAG 문서 검색 시스템</h1>
      <div class="flex items-center gap-4">
        <span v-if="auth.isAuth" class="text-sm text-gray-600"
          >{{ auth.username }} 님</span
        >
        <button
          v-if="auth.isAuth"
          @click="logout"
          class="text-sm px-3 py-1.5 border rounded-lg hover:bg-gray-50 text-gray-600"
        >
          로그아웃
        </button>
        <router-link
          v-else
          to="/login"
          class="text-sm px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          로그인
        </router-link>
      </div>
    </header>

    <div class="max-w-5xl mx-auto p-6 space-y-6">
      <!-- 법령 선택 -->
      <div class="bg-white rounded-2xl shadow p-5">
        <h2 class="font-semibold text-gray-800 mb-3">
          검색할 법령 선택
          <span class="text-xs text-gray-400">(미선택 시 전체 검색)</span>
        </h2>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="law in availableLaws"
            :key="law"
            @click="toggleLaw(law)"
            :class="[
              'px-3 py-1.5 rounded-full text-sm border transition',
              selectedLaws.includes(law)
                ? 'bg-blue-600 text-white border-blue-600'
                : 'text-gray-600 border-gray-200 hover:border-blue-400',
            ]"
          >
            {{ law }}
          </button>
        </div>
      </div>

      <!-- 질문 입력 -->
      <div class="bg-white rounded-2xl shadow p-5">
        <h2 class="font-semibold text-gray-800 mb-3">질문하기</h2>
        <div class="flex gap-2">
          <input
            v-model="question"
            @keydown.enter="ask"
            placeholder="노동법령에 관해 궁금한 점을 질문해주세요..."
            class="flex-1 border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <button
            @click="ask"
            :disabled="loading || !question.trim()"
            class="px-6 py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium disabled:opacity-40 hover:bg-blue-700 transition"
          >
            {{ loading ? "검색 중..." : "검색" }}
          </button>
        </div>
        <div v-if="loading" class="mt-3 text-sm text-blue-500 animate-pulse">
          AI가 법령을 검색하고 있습니다...
        </div>
      </div>

      <!-- 답변 -->
      <div v-if="result" class="bg-white rounded-2xl shadow p-5 space-y-4">
        <h2 class="font-semibold text-gray-800">답변</h2>
        <div
          class="bg-blue-50 rounded-xl p-4 text-sm text-gray-800 whitespace-pre-wrap leading-relaxed"
        >
          {{ result.answer }}
        </div>
        <div v-if="result.sources?.length">
          <h3 class="text-sm font-semibold text-gray-600 mb-2">참고 조문</h3>
          <div
            v-for="(s, i) in result.sources"
            :key="i"
            class="border rounded-xl p-3 mb-2 text-sm"
          >
            <div class="flex gap-2 mb-1">
              <span
                class="bg-blue-100 text-blue-700 px-2 py-0.5 rounded text-xs font-medium"
                >{{ s.law_name }}</span
              >
              <span
                class="bg-gray-100 text-gray-600 px-2 py-0.5 rounded text-xs"
                >{{ s.article_no }} {{ s.article_title }}</span
              >
            </div>
            <p class="text-gray-600 text-xs leading-relaxed">
              {{ s.content?.slice(0, 200) }}...
            </p>
          </div>
        </div>
        <p v-if="!auth.isAuth" class="text-xs text-gray-400">
          💡
          <router-link to="/login" class="text-blue-500 hover:underline"
            >로그인</router-link
          >하면 질의 히스토리가 저장됩니다.
        </p>
      </div>

      <!-- 히스토리 -->
      <div
        v-if="auth.isAuth && history.length"
        class="bg-white rounded-2xl shadow p-5"
      >
        <h2 class="font-semibold text-gray-800 mb-3">질의 히스토리</h2>
        <div
          v-for="item in history"
          :key="item.id"
          class="border-b last:border-0 py-3"
        >
          <p class="text-sm font-medium text-gray-800">
            Q: {{ item.question }}
          </p>
          <p class="text-xs text-gray-500 mt-1">
            {{ item.created_at?.slice(0, 10) }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import { useAuthStore } from "@/stores/auth";
import axios from "axios";

const http = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8001",
});
http.interceptors.request.use((c) => {
  const t = localStorage.getItem("access_token");
  if (t) c.headers.Authorization = `Bearer ${t}`;
  return c;
});

const auth = useAuthStore();
const router = useRouter();
const availableLaws = ref([]);
const selectedLaws = ref([]);
const question = ref("");
const loading = ref(false);
const result = ref(null);
const history = ref([]);

onMounted(async () => {
  const res = await http.get("/api/v1/laws");
  availableLaws.value = res.data.data || res.data;
  if (auth.isAuth) loadHistory();
});

function toggleLaw(law) {
  const idx = selectedLaws.value.indexOf(law);
  if (idx >= 0) selectedLaws.value.splice(idx, 1);
  else selectedLaws.value.push(law);
}

async function ask() {
  if (!question.value.trim()) return;
  loading.value = true;
  result.value = null;
  try {
    const res = await http.post("/api/v1/qa/ask", {
      question: question.value,
      law_names: selectedLaws.value.length ? selectedLaws.value : [],
    });
    result.value = res.data.data || res.data;
    if (auth.isAuth) loadHistory();
  } catch (e) {
    alert(e.response?.data?.detail || "오류가 발생했습니다.");
  } finally {
    loading.value = false;
  }
}

async function loadHistory() {
  const res = await http.get("/api/v1/qa/history");
  history.value = res.data.data?.items || [];
}

function logout() {
  auth.logout();
  router.push("/login");
}
</script>
