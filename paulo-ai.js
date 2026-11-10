/* パウロ先生の AI（iPad の中だけで動く。Claude などの外の AI には問い合わせない）
   ・手書き：1文字ずつ読む手書き文字の AI（LT8/japanese-handwriting-onnx、14.5MB）を ONNX Runtime Web で動かす
   ・質問：文章の AI を WebLLM で動かす。WebGPU が使えるブラウザだけ。大きさは2つから選べる
       かしこい＝ gemma-2-2b-jpn-it（約1.4GB）／かるい＝ Qwen3 0.6B（約350MB、まちがいが多め）
   どちらも、はじめて使うときにダウンロードし、そのあとはブラウザに残したものを使う */
(function () {
'use strict';

const HAND_MODEL = 'https://huggingface.co/LT8/japanese-handwriting-onnx/resolve/main/model.fp16.onnx';
const HAND_LABELS = 'https://huggingface.co/LT8/japanese-handwriting-onnx/resolve/main/labels.json';
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
const WEBLLM_URL = 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.86/+esm';
/* 質問の AI の大きさ。国語の質問20問でくらべると、かしこい＝18問ほど正解、かるい＝半分以下だった。
   f16 は 16bit 計算ができる端末用、f32 はできない端末用（少し大きい） */
const CHAT_SIZES = {
  big:   { name: 'かしこい', dl: '約1.4GB', f16: 'gemma-2-2b-jpn-it-q4f16_1-MLC', f32: 'gemma-2-2b-jpn-it-q4f32_1-MLC' },
  small: { name: 'かるい',   dl: '約350MB', f16: 'Qwen3-0.6B-q4f16_1-MLC',        f32: 'Qwen3-0.6B-q4f32_1-MLC' },
};
const SIZE_KEY = 'paulo-ai/chat-size', READY_KEY = 'paulo-ai/chat-ready', TRY_KEY = 'paulo-ai/chat-trying', FELL_KEY = 'paulo-ai/chat-fell-back';
const SIZE = 128;            // 手書き AI に渡す絵の大きさ
const CONFIDENT = 0.7;       // これより自信が低いときは「読めなかった」とする

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function loadScript(src) {
  return new Promise((resolve, reject) => { const s = document.createElement('script'); s.src = src; s.onload = resolve; s.onerror = () => reject(new Error('script')); document.head.appendChild(s); });
}

/* =========================================================
   手書き文字の AI
   ========================================================= */
const hand = { promise: null };
function loadHandModel() {
  if (hand.promise) return hand.promise;
  hand.promise = (async () => {
    if (!window.ort) await loadScript(ORT_URL);
    const [labels, session] = await Promise.all([
      fetch(HAND_LABELS).then(r => { if (!r.ok) throw new Error('labels'); return r.json(); }),
      window.ort.InferenceSession.create(HAND_MODEL, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }),
    ]);
    return { labels, session };
  })();
  hand.promise.catch(() => { hand.promise = null; });   // 失敗したら、次にもう一度読みこむ
  return hand.promise;
}

/** マスの絵を AI に渡して、候補を5つ返す */
async function recognizeCanvas(canvas) {
  const { labels, session } = await loadHandModel();
  const small = document.createElement('canvas'); small.width = small.height = SIZE;
  const g = small.getContext('2d', { willReadFrequently: true });
  g.fillStyle = '#fff'; g.fillRect(0, 0, SIZE, SIZE); g.drawImage(canvas, 0, 0, SIZE, SIZE);
  const px = g.getImageData(0, 0, SIZE, SIZE).data; const data = new Float32Array(SIZE * SIZE);
  for (let i = 0, j = 0; i < px.length; i += 4, j++) data[j] = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
  const out = await session.run({ input: new window.ort.Tensor('float32', data, [1, 1, SIZE, SIZE]) });
  const logits = Array.from(out.logits.data); const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max)); const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v, i) => ({ p: v / sum, char: labels[i].char })).sort((a, b) => b.p - a.p).slice(0, 5);
}

/* 手書きの窓：マスに1文字ずつ書くと、文字になる */
const CELLS = 8;
function openHandPad({ onInsert }) {
  closeHandPad();
  const root = document.createElement('div'); root.className = 'pa-overlay'; root.id = 'pa-pad';
  root.innerHTML = `<div class="pa-sheet" role="dialog" aria-modal="true" aria-label="手書きで書く">
    <div class="pa-head"><h2>手書きで書く</h2><button type="button" class="pa-icon" data-pa="close" aria-label="閉じる">✕</button></div>
    <p class="pa-status" id="pa-status">手書きの AI を準備しています…（はじめてだけ約15MB）</p>
    <div class="pa-grid">${Array.from({ length: CELLS }, (_, i) => `<div class="pa-cell" data-cell="${i}"><canvas width="256" height="256" aria-label="${i + 1}文字目のマス"></canvas><div class="pa-cands"></div></div>`).join('')}</div>
    <p class="pa-result">読みとった文字：<b id="pa-text"></b></p>
    <div class="pa-foot"><button type="button" class="pa-btn" data-pa="clear">全部消す</button><button type="button" class="pa-btn pa-primary" data-pa="insert">メモに入れる</button></div>
  </div>`;
  document.body.appendChild(root);
  const picks = Array(CELLS).fill('');
  const status = root.querySelector('#pa-status');
  const showText = () => { root.querySelector('#pa-text').textContent = picks.join('') || '（まだありません）'; };
  showText();
  loadHandModel().then(() => { status.textContent = 'マスに1文字ずつ、大きく書いてください。書き終わると文字になります。'; })
    .catch(() => { status.textContent = '手書きの AI を読みこめませんでした。通信のよい所で開き直してください。'; });

  root.querySelectorAll('.pa-cell').forEach(cell => {
    const index = +cell.dataset.cell, canvas = cell.querySelector('canvas'), g = canvas.getContext('2d');
    const clear = () => { g.fillStyle = '#fff'; g.fillRect(0, 0, 256, 256); picks[index] = ''; cell.querySelector('.pa-cands').innerHTML = ''; cell.classList.remove('is-done', 'is-weak'); };
    clear();
    let drawing = false, timer = 0;
    const point = e => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) * 256 / r.width, (e.clientY - r.top) * 256 / r.height]; };
    canvas.addEventListener('pointerdown', e => {
      e.preventDefault(); clearTimeout(timer); drawing = true; try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* 古いブラウザ */ }
      const [x, y] = point(e); g.strokeStyle = '#000'; g.lineWidth = 14; g.lineCap = 'round'; g.lineJoin = 'round'; g.beginPath(); g.moveTo(x, y); g.lineTo(x + 0.1, y); g.stroke();
    });
    canvas.addEventListener('pointermove', e => { if (!drawing) return; const [x, y] = point(e); g.lineTo(x, y); g.stroke(); });
    const end = () => {
      if (!drawing) return; drawing = false;
      // 少し手を止めたら、そのマスを読みとる（漢字は何画も書くので、すぐには読まない）
      clearTimeout(timer); timer = setTimeout(async () => {
        try {
          const top = await recognizeCanvas(canvas);
          const weak = top[0].p < CONFIDENT;
          picks[index] = weak ? '' : top[0].char;
          cell.classList.toggle('is-done', !weak); cell.classList.toggle('is-weak', weak);
          cell.querySelector('.pa-cands').innerHTML = (weak ? '<span class="pa-weak">読めません</span>' : '')
            + top.slice(0, 3).map((t, i) => `<button type="button" class="pa-cand${!weak && i === 0 ? ' is-on' : ''}" data-pick="${esc(t.char)}">${esc(t.char)}</button>`).join('')
            + `<button type="button" class="pa-cand pa-erase" data-erase aria-label="このマスを消す">✕</button>`;
          showText();
        } catch (err) { status.textContent = '読みとれませんでした。もう一度書いてください。'; }
      }, 700);
    };
    canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
    cell.querySelector('.pa-cands').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.hasAttribute('data-erase')) { clear(); showText(); return; }
      picks[index] = b.dataset.pick; cell.classList.add('is-done'); cell.classList.remove('is-weak');
      cell.querySelectorAll('.pa-cand').forEach(x => x.classList.toggle('is-on', x === b)); showText();
    });
    cell.clearCell = clear;
  });

  root.addEventListener('click', e => {
    const b = e.target.closest('[data-pa]'); if (!b) { if (e.target === root) closeHandPad(); return; }
    if (b.dataset.pa === 'close') closeHandPad();
    if (b.dataset.pa === 'clear') { root.querySelectorAll('.pa-cell').forEach(c => c.clearCell()); showText(); }
    if (b.dataset.pa === 'insert') { const text = picks.join(''); if (!text) { status.textContent = 'まだ文字がありません。マスに書いてください。'; return; } onInsert(text); closeHandPad(); }
  });
}
function closeHandPad() { const old = document.getElementById('pa-pad'); if (old) old.remove(); }

/* =========================================================
   質問に答える文章の AI
   ========================================================= */
const chat = { engine: null, promise: null, size: '', model: '', busy: false };
const store = {
  get(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },
  set(key, value) { try { if (value == null) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch (e) { /* 保存できない環境 */ } },
};

/* かしこい AI は大きいので、古い iPad ではメモリが足りずにページごと止まることがある。
   止まると後始末ができないので、「いま大きい AI を動かしている」という印を先に残しておき、
   次に開いたとき印が残っていたら、かるい AI に切りかえる */
if (store.get(TRY_KEY) === 'big') { store.set(TRY_KEY, null); store.set(SIZE_KEY, 'small'); store.set(FELL_KEY, '1'); }
let tryingBig = false;
const markTrying = () => store.set(TRY_KEY, tryingBig && !document.hidden ? 'big' : null);
function trying(on) { tryingBig = on; markTrying(); }
addEventListener('pagehide', () => store.set(TRY_KEY, null));   // ふつうに閉じたときは「止まった」とは数えない
document.addEventListener('visibilitychange', markTrying);

const chatSize = () => (store.get(SIZE_KEY) === 'small' ? 'small' : 'big');
/** 大きさを変える。答えている途中は変えられない（false を返す） */
function setChatSize(size) {
  if (chat.busy) return false;
  store.set(SIZE_KEY, size === 'small' ? 'small' : 'big'); store.set(FELL_KEY, null); return true;
}
const chatReadyBefore = (size = chatSize()) => store.get(`${READY_KEY}/${size}`) === '1' || (size === 'small' && store.get(READY_KEY) === '1');
const chatInfo = () => { const size = chatSize(); return { size, name: CHAT_SIZES[size].name, dl: CHAT_SIZES[size].dl, ready: chatReadyBefore(size), fellBack: store.get(FELL_KEY) === '1' }; };

/** この端末で文章の AI が動くか（WebGPU が必要） */
async function chatSupport() {
  if (!('gpu' in navigator)) return { ok: false, reason: 'この端末のブラウザでは、質問の AI が動きません（iPad は iPadOS 26 以上の Safari が必要です）。' };
  try { const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error('none'); return { ok: true, f16: adapter.features.has('shader-f16') }; }
  catch (e) { return { ok: false, reason: 'この端末では、質問の AI に必要なグラフィックの機能が使えません。' }; }
}

function loadChat(onProgress) {
  const size = chatSize();
  if (chat.promise && chat.size === size) return chat.promise;
  const old = chat.promise; chat.size = size;
  const mine = chat.promise = (async () => {
    const support = await chatSupport(); if (!support.ok) throw Object.assign(new Error(support.reason), { friendly: true });
    if (old) { try { await (await old).unload(); } catch (e) { /* 前の AI がなければ何もしない */ } chat.engine = null; }   // 2つ同時にはメモリに置かない
    const webllm = await import(WEBLLM_URL);
    const model = support.f16 ? CHAT_SIZES[size].f16 : CHAT_SIZES[size].f32;
    const engine = await webllm.CreateMLCEngine(model, { initProgressCallback: r => onProgress && onProgress(r) });
    chat.engine = engine; chat.model = model; store.set(`${READY_KEY}/${size}`, '1');
    return engine;
  })();
  mine.catch(() => { if (chat.promise === mine) chat.promise = null; });   // 失敗したら、次にもう一度読みこむ
  return mine;
}

const RULES = `あなたは中学生に国語を教える、やさしい先生「パウロ先生」です。
下の【文章】を読んで、生徒の【質問】に答えてください。

きまり
・答えは、まず【印のある文】とその前後から探し、なければ【文章】全体から探します。
・理由を聞かれたら、文章の中で理由が書かれている所を探して「〜からです。」と答えます。
・答えは「です・ます」で、1〜2文にします。質問をくり返したり、前置きを書いたりしません。
・ことばの意味を聞かれたら、この文章の中での意味を、やさしいことばで説明します。
・この先の出来事など、文章に書かれていないことは「文章には書かれていません。」と答えます。`;

/** 質問する。passage は印のまわりの広い本文、sentence は印のある文、marked は印をつけたことば。
    onThinking は AI の準備ができて考え始めたとき、onText にはそれまでの答え全体がとどく */
async function ask({ title, author, passage, sentence, marked, question }, { onText, onProgress, onThinking } = {}) {
  if (chat.busy) throw new Error('busy');
  chat.busy = true; trying(chatSize() === 'big');
  try {
    const engine = await loadChat(onProgress);
    onThinking && onThinking();
    const small = chat.size === 'small';
    const user = `【文章】（${title}${author ? '　' + author : ''}）\n${passage}\n\n【印のある文】${sentence}\n\n【質問】（生徒が印をつけた「${marked}」について）${question}`;
    // gemma には「先生役の指示」を入れる場所がないので、質問といっしょに渡す
    const messages = /gemma/i.test(chat.model) ? [{ role: 'user', content: RULES + '\n\n' + user }] : [{ role: 'system', content: RULES }, { role: 'user', content: user }];
    const request = { messages, stream: true, temperature: 0.2, max_tokens: small ? 140 : 220 };
    if (/^Qwen3/.test(chat.model)) request.extra_body = { enable_thinking: false };   // 考えている途中の文は出さない
    const stream = await engine.chat.completions.create(request);
    let text = '';
    for await (const chunk of stream) {
      text += (chunk.choices[0] && chunk.choices[0].delta && chunk.choices[0].delta.content) || '';
      onText && onText(cleanAnswer(text, small).replace(/�+$/, ''));   // 途中で切れた文字は出さない
    }
    // 少しずつ届いた文字は、漢字の途中で切れて化けることがあるので、最後に全文を取り直す
    try { const full = await engine.getMessage(); if (full) text = full; } catch (e) {}
    return cleanAnswer(text, small);
  } finally { chat.busy = false; trying(false); }
}
/** AI が書き写した見出し（【印のある文】など）や飾りを取る。かるい AI は同じことをくり返すので、はじめの2行だけ残す */
function cleanAnswer(raw, small) {
  const s = raw.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').replace(/\*\*/g, '').replace(/【[^】\n]{1,12}】/g, '').replace(/【[^】]*$/, '');
  const lines = s.split('\n').map(line => line.replace(/^\s*(答え|パウロ先生)[：:]\s*/, '').trim()).filter(Boolean);
  return (small ? lines.slice(0, 2) : lines).join('\n');
}

/* 見た目（アプリの色の変数を使う） */
const style = document.createElement('style');
style.textContent = `
.pa-overlay{position:fixed;inset:0;z-index:200;background:rgba(20,30,24,.4);display:flex;align-items:flex-end;justify-content:center}
.pa-sheet{width:min(720px,100%);max-height:92dvh;overflow:auto;background:var(--paper,#fff);border-radius:20px 20px 0 0;padding:16px 18px calc(16px + env(safe-area-inset-bottom,0px));display:grid;gap:12px;box-shadow:0 -10px 40px rgba(0,0,0,.2)}
.pa-head{display:flex;align-items:center;justify-content:space-between}
.pa-head h2{margin:0;font:500 19px var(--f-disp,serif);letter-spacing:.08em}
.pa-icon{width:46px;height:46px;border:0;border-radius:50%;background:none;font-size:20px;color:var(--green,#1d4a37)}
.pa-status{margin:0;font-size:14px;color:var(--ink-2,#48514b)}
.pa-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.pa-cell{display:grid;gap:6px}
.pa-cell canvas{width:100%;aspect-ratio:1;border:1.5px solid var(--line-2,#d9c597);border-radius:12px;touch-action:none;background:#fff;
  background-image:linear-gradient(var(--line,#ebdfc2) 1px,transparent 1px),linear-gradient(90deg,var(--line,#ebdfc2) 1px,transparent 1px);background-size:50% 50%;background-position:center}
.pa-cell.is-done canvas{border-color:var(--green,#1d4a37)}
.pa-cell.is-weak canvas{border-color:#b53b2c}
.pa-cands{display:flex;flex-wrap:wrap;gap:4px;min-height:44px}
.pa-cand{min-width:44px;min-height:44px;border:1px solid var(--line-2,#d9c597);border-radius:10px;background:#fff;font:20px var(--f-read,serif)}
.pa-cand.is-on{border-color:var(--green,#1d4a37);background:var(--green-soft,#eef4f0);color:var(--green,#1d4a37)}
.pa-erase{font:14px var(--f-ui,sans-serif);color:var(--ink-2,#48514b)}
.pa-weak{font-size:12px;color:#b53b2c;align-self:center}
.pa-result{margin:0;font-size:15px}
.pa-result b{font:500 22px var(--f-read,serif);letter-spacing:.06em}
.pa-foot{display:flex;gap:8px;justify-content:flex-end}
.pa-btn{min-height:48px;padding:0 22px;border-radius:999px;border:1px solid var(--line-2,#d9c597);background:#fff;font-size:15px}
.pa-primary{background:var(--green,#1d4a37);border-color:var(--green,#1d4a37);color:#fff}
.pq{display:grid;gap:6px;margin-top:10px;padding-top:10px;border-top:1px dashed var(--line-2,#d9c597)}
.pq-row{display:flex;gap:6px}
.pq-row input{flex:1;min-width:0;min-height:44px;padding:6px 12px;border:1px solid var(--line,#ebdfc2);border-radius:12px;font-size:15px;background:#fff}
.pq-note{font-size:11.5px;color:var(--muted,#7a817b)}
.pq-out{white-space:pre-wrap;font-size:14px;line-height:1.9}
.pq-size{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}
@media (max-width:520px){.pa-grid{grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}}
`;
document.head.appendChild(style);

window.PauloAI = { openHandPad, closeHandPad, recognizeCanvas, loadHandModel, chatSupport, chatReadyBefore, chatInfo, chatSize, setChatSize, CHAT_SIZES, loadChat, ask };
})();
