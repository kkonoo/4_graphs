/**
 * AI Chat Widget for Quarto
 * 각 페이지의 YAML front matter에서 시스템 프롬프트를 읽어옵니다.
 * window.AI_CHAT_CONFIG 에서 설정을 가져옵니다.
 */

(function () {
  const cfg = window.AI_CHAT_CONFIG || {};
  const WORKER_URL = cfg.workerUrl || "";
  const CHAPTER_CONTEXT = cfg.chapterContext || "";
  const CHAPTER_TITLE = cfg.chapterTitle || document.title;

  if (!WORKER_URL) {
    console.warn("[AI Chat] workerUrl이 설정되지 않았습니다.");
    return;
  }

  /* ── 시스템 프롬프트 ───────────────────────── */
  const SYSTEM_PROMPT = `당신은 "${CHAPTER_TITLE}" 강의자료를 도와주는 학습 도우미입니다.

아래는 현재 챕터의 핵심 내용입니다:
---
${CHAPTER_CONTEXT}
---

규칙:
- 위 챕터 내용을 중심으로 답변하세요.
- 질문이 챕터 범위를 벗어나면 간단히 언급하고 챕터 내용으로 연결해주세요.
- R 코드 예시가 도움이 될 때는 포함하세요.
- 한국어로 답변하되, 전문 용어는 영어 병기 가능합니다.
- 의학/과학 학생 대상이므로 정확성을 최우선으로 합니다.`;

  /* ── DOM 생성 ────────────────────────────── */
  const style = document.createElement("style");
  style.textContent = `
    #ai-chat-fab {
      position: fixed; bottom: 28px; right: 28px; z-index: 9999;
      width: 52px; height: 52px; border-radius: 50%;
      background: #1a6e3c; color: #fff; border: none;
      box-shadow: 0 4px 16px rgba(0,0,0,.22);
      cursor: pointer; font-size: 22px;
      display: flex; align-items: center; justify-content: center;
      transition: background .2s, transform .15s;
    }
    #ai-chat-fab:hover { background: #155730; transform: scale(1.07); }

    #ai-chat-panel {
      position: fixed; bottom: 90px; right: 28px; z-index: 9998;
      width: 360px; max-width: calc(100vw - 40px);
      height: 500px; max-height: calc(100vh - 120px);
      background: #fff; border-radius: 16px;
      box-shadow: 0 8px 40px rgba(0,0,0,.18);
      display: flex; flex-direction: column;
      font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
      overflow: hidden;
      transform: scale(.92) translateY(12px);
      opacity: 0; pointer-events: none;
      transition: transform .22s cubic-bezier(.34,1.56,.64,1), opacity .18s;
    }
    #ai-chat-panel.open {
      transform: scale(1) translateY(0);
      opacity: 1; pointer-events: all;
    }

    #ai-chat-header {
      background: #1a6e3c; color: #fff;
      padding: 14px 16px 12px;
      display: flex; align-items: center; gap: 10px;
      font-size: 14px; font-weight: 600; letter-spacing: .01em;
      flex-shrink: 0;
    }
    #ai-chat-header span { flex: 1; }
    #ai-chat-close {
      background: none; border: none; color: #fff;
      font-size: 18px; cursor: pointer; padding: 0 2px; line-height: 1;
      opacity: .8;
    }
    #ai-chat-close:hover { opacity: 1; }

    #ai-chat-subtitle {
      font-size: 11px; color: rgba(255,255,255,.72);
      font-weight: 400; display: block; margin-top: 2px;
    }

    #ai-chat-messages {
      flex: 1; overflow-y: auto; padding: 14px 14px 8px;
      display: flex; flex-direction: column; gap: 10px;
      scroll-behavior: smooth;
    }
    #ai-chat-messages::-webkit-scrollbar { width: 4px; }
    #ai-chat-messages::-webkit-scrollbar-thumb {
      background: #ddd; border-radius: 4px;
    }

    .ai-msg {
      max-width: 88%; padding: 9px 13px;
      border-radius: 14px; font-size: 13.5px; line-height: 1.55;
      white-space: pre-wrap; word-break: break-word;
    }
    .ai-msg.user {
      align-self: flex-end;
      background: #1a6e3c; color: #fff;
      border-bottom-right-radius: 4px;
    }
    .ai-msg.assistant {
      align-self: flex-start;
      background: #f3f4f6; color: #1f2937;
      border-bottom-left-radius: 4px;
    }
    .ai-msg.assistant code {
      background: #e5e7eb; padding: 1px 5px;
      border-radius: 4px; font-size: 12.5px; font-family: monospace;
    }
    .ai-msg.assistant pre {
      background: #1f2937; color: #f9fafb;
      padding: 10px 12px; border-radius: 8px; overflow-x: auto;
      font-size: 12px; margin: 6px 0;
    }
    .ai-msg.assistant pre code { background: none; padding: 0; color: inherit; }

    .ai-typing {
      align-self: flex-start;
      display: flex; gap: 5px; padding: 10px 14px;
      background: #f3f4f6; border-radius: 14px; border-bottom-left-radius: 4px;
    }
    .ai-typing span {
      width: 7px; height: 7px; background: #9ca3af;
      border-radius: 50%; animation: aiDot 1.2s infinite;
    }
    .ai-typing span:nth-child(2) { animation-delay: .2s; }
    .ai-typing span:nth-child(3) { animation-delay: .4s; }
    @keyframes aiDot {
      0%, 60%, 100% { transform: translateY(0); opacity: .4; }
      30% { transform: translateY(-5px); opacity: 1; }
    }

    #ai-chat-input-row {
      display: flex; gap: 8px; padding: 10px 12px 12px;
      border-top: 1px solid #f0f0f0; flex-shrink: 0;
    }
    #ai-chat-input {
      flex: 1; border: 1px solid #e5e7eb; border-radius: 10px;
      padding: 8px 12px; font-size: 13.5px; resize: none;
      font-family: inherit; outline: none; max-height: 100px;
      line-height: 1.45; color: #1f2937;
      transition: border-color .15s;
    }
    #ai-chat-input:focus { border-color: #1a6e3c; }
    #ai-chat-send {
      width: 36px; height: 36px; border-radius: 9px;
      background: #1a6e3c; color: #fff; border: none;
      cursor: pointer; font-size: 16px; flex-shrink: 0;
      align-self: flex-end;
      display: flex; align-items: center; justify-content: center;
      transition: background .15s;
    }
    #ai-chat-send:hover { background: #155730; }
    #ai-chat-send:disabled { background: #9ca3af; cursor: not-allowed; }
  `;
  document.head.appendChild(style);

  document.body.insertAdjacentHTML("beforeend", `
    <button id="ai-chat-fab" title="AI 학습 도우미">💬</button>
    <div id="ai-chat-panel">
      <div id="ai-chat-header">
        <div>
          <span>AI 학습 도우미</span>
          <small id="ai-chat-subtitle">${CHAPTER_TITLE}</small>
        </div>
        <button id="ai-chat-close">✕</button>
      </div>
      <div id="ai-chat-messages">
        <div class="ai-msg assistant">안녕하세요! 이 챕터에 대해 궁금한 점을 물어보세요 😊</div>
      </div>
      <div id="ai-chat-input-row">
        <textarea id="ai-chat-input" placeholder="질문을 입력하세요… (Shift+Enter: 줄바꿈)" rows="1"></textarea>
        <button id="ai-chat-send">➤</button>
      </div>
    </div>
  `);

  /* ── 상태 ─────────────────────────────────── */
  const panel = document.getElementById("ai-chat-panel");
  const fab = document.getElementById("ai-chat-fab");
  const closeBtn = document.getElementById("ai-chat-close");
  const input = document.getElementById("ai-chat-input");
  const sendBtn = document.getElementById("ai-chat-send");
  const messages = document.getElementById("ai-chat-messages");
  const history = []; // {role, content}[]

  /* ── 토글 ─────────────────────────────────── */
  fab.addEventListener("click", () => {
    panel.classList.toggle("open");
    if (panel.classList.contains("open")) input.focus();
  });
  closeBtn.addEventListener("click", () => panel.classList.remove("open"));

  /* ── 자동 높이 조정 ───────────────────────── */
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 100) + "px";
  });

  /* ── 전송 ─────────────────────────────────── */
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  sendBtn.addEventListener("click", sendMessage);

  function appendMsg(role, text) {
    const div = document.createElement("div");
    div.className = `ai-msg ${role}`;
    // 간단한 코드블록 렌더링
    div.innerHTML = text
      .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
        `<pre><code>${escHtml(code.trim())}</code></pre>`
      )
      .replace(/`([^`]+)`/g, (_, c) => `<code>${escHtml(c)}</code>`)
      .replace(/\n/g, "<br>");
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }

  function escHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function showTyping() {
    const el = document.createElement("div");
    el.className = "ai-typing";
    el.innerHTML = "<span></span><span></span><span></span>";
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || sendBtn.disabled) return;

    input.value = "";
    input.style.height = "auto";
    sendBtn.disabled = true;

    appendMsg("user", text);
    history.push({ role: "user", content: text });

    const typing = showTyping();

    try {
      const res = await fetch(WORKER_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 1024,
          system: SYSTEM_PROMPT,
          messages: history,
        }),
      });

      const data = await res.json();
      typing.remove();

      if (data.content && data.content[0]) {
        const reply = data.content[0].text;
        appendMsg("assistant", reply);
        history.push({ role: "assistant", content: reply });
      } else {
        appendMsg("assistant", "오류가 발생했습니다. 다시 시도해주세요.");
      }
    } catch (err) {
      typing.remove();
      appendMsg("assistant", "네트워크 오류가 발생했습니다.");
      console.error("[AI Chat]", err);
    }

    sendBtn.disabled = false;
    input.focus();
  }
})();
