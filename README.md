# Quarto AI 강의 템플릿


https://kkonoo.github.io/4_graphs/

통계 이론 설명 + R 코드 실행 + AI 학습 도우미가 통합된 Quarto 강의 사이트 템플릿입니다.

---

## 🗂 파일 구조

```
.
├── _quarto.yml                  # 사이트 전역 설정
├── index.qmd                    # 홈페이지
├── chapters/
│   └── 01_intro.qmd             # 챕터 예시
├── assets/
│   ├── custom.css               # 커스텀 스타일
│   └── ai-chat-init.html        # AI 위젯 초기화 (Worker URL 입력)
├── _extensions/
│   └── ai-chat/
│       └── ai-chat.js           # AI 채팅 위젯 코어
├── cloudflare-worker.js         # Cloudflare Worker 코드
└── .github/workflows/deploy.yml # GitHub Actions 자동 배포
```

---

## 🚀 시작하기

### 1단계: Cloudflare Worker 배포 (API 키 숨기기)

1. [Cloudflare Dashboard](https://dash.cloudflare.com) 접속
2. **Workers & Pages → Create Worker**
3. `cloudflare-worker.js` 내용 붙여넣기
4. 저장 후 **Settings → Environment Variables**
   - 변수명: `ANTHROPIC_API_KEY`
   - 값: Anthropic API 키
   - **"Encrypt" 체크** (시크릿으로 저장)
5. Worker URL 복사 (예: `https://my-worker.workers.dev`)

> Cloudflare Workers 무료 플랜: 하루 10만 요청까지 무료

### 2단계: Worker URL 입력

`assets/ai-chat-init.html` 파일에서:

```js
const WORKER_URL = "https://YOUR_WORKER.workers.dev";
//                  ↑ 여기를 복사한 URL로 교체
```

또한 `cloudflare-worker.js`의 `ALLOWED_ORIGINS`에 GitHub Pages 도메인 추가:

```js
const ALLOWED_ORIGINS = [
  "https://YOUR_USERNAME.github.io",  // ← 변경
  "http://localhost:4321",
];
```

### 3단계: 사이트 설정

`_quarto.yml`에서:
- `title`: 강의 제목
- `navbar.right.href`: GitHub 저장소 URL

### 4단계: GitHub Pages 활성화

1. GitHub 저장소 → **Settings → Pages**
2. Source: **GitHub Actions** 선택
3. `main` 브랜치에 push → 자동 빌드 & 배포

---

## ✍️ 챕터 작성법

새 챕터 `.qmd` 파일 생성 시 상단 YAML에 AI 컨텍스트 추가:

```yaml
---
title: "W5. Pre-processing of RNA-seq Data"
format:
  html:
    include-in-header:
      text: |
        <meta name="ai-title" content="W5. RNA-seq Pre-processing">
        <meta name="ai-context" content="
        이 챕터에서 다루는 핵심 개념을 2-3문단으로 요약.
        AI가 이 내용을 시스템 프롬프트로 받아 답변합니다.
        ">
---
```

그 다음 `_quarto.yml`의 `sidebar.contents`에 파일 추가:

```yaml
- section: "Part 2. DEG Analysis"
  contents:
    - chapters/05_preprocessing.qmd
```

---

## 💡 로컬 미리보기

```bash
# Quarto 설치: https://quarto.org/docs/get-started/
quarto preview
```

---

## 📦 사용된 기술

- [Quarto](https://quarto.org) — R Markdown 기반 출판 시스템
- [Cloudflare Workers](https://workers.cloudflare.com) — API 키 프록시
- [Anthropic API](https://anthropic.com) — AI 채팅 백엔드
- GitHub Pages — 무료 호스팅
