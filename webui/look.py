"""Theme, CSS and the alert sound."""
import gradio as gr


# Browsers only allow sound after the user has interacted with the page, so the first click
# anywhere unlocks an AudioContext. sttBeep() is the low-RAM alarm; sttNotify() chimes when long
# work ends and, if the page is in the background, shows a desktop notification.
BEEP_JS = """
<script>
(() => {
  let ctx = null;
  const unlock = () => {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") ctx.resume();
  };
  document.addEventListener("pointerdown", unlock, { capture: true });
  document.addEventListener("keydown", unlock, { capture: true });
  const play = (notes) => {
    unlock();
    notes.forEach(([freq, at]) => {
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime + at);
      gain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + at + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + at + 0.22);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + at);
      osc.stop(ctx.currentTime + at + 0.25);
    });
  };
  // Low RAM: an urgent two-tone alarm.
  window.sttBeep = () => play([[880, 0], [660, 0.28], [880, 0.56], [660, 0.84]]);
  // Long work finished: a short rising chime, and a desktop notification if the page is hidden.
  window.sttAskNotify = () => {
    if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
  };
  window.sttNotify = (title, html) => {
    play([[523, 0], [659, 0.15], [784, 0.3]]);
    const tmp = document.createElement("div");
    tmp.innerHTML = html || "";
    const body = tmp.innerText.replace(/\s+/g, " ").trim();
    if ("Notification" in window && Notification.permission === "granted" && document.hidden) {
      new Notification(title, { body });
    }
  };
})();
</script>
"""

# Read along (webui.readalong): a click on a word plays the audio from there; while it plays,
# the current word is lit and kept in view. Listeners sit on the document, so they work for
# views rendered later. timeupdate doesn't bubble, hence the capture phase.
READALONG_JS = """
<script>
(() => {
  const box = (el) => el && el.closest && el.closest(".readalong");
  const editing = (b) => b.classList.contains("editing");
  const saved = new WeakMap();  // a view's text before editing, for Cancel
  const remember = (k, v) => { try { localStorage.setItem("ra-" + k, v); } catch (e) {} };
  const recall = (k) => { try { return localStorage.getItem("ra-" + k); } catch (e) { return null; } };

  // Views are re-rendered (after a save, on opening the tab): restore the reader's choices.
  const setup = (b) => {
    if (b.dataset.ready) return;
    b.dataset.ready = "1";
    const size = recall("size"), speed = recall("speed"), follow = recall("follow");
    if (size) b.style.setProperty("--ra-size", size + "rem");
    if (speed) b.querySelector('[data-act="speed"]').value = speed;
    if (follow) { b.dataset.follow = follow; b.querySelector('[data-act="follow"]').checked = follow === "1"; }
    if (sessionStorage.getItem("ra-focus") === "1") setFocus(b, true);
  };
  new MutationObserver(() => document.querySelectorAll(".readalong").forEach(setup))
    .observe(document.documentElement, { childList: true, subtree: true });

  const setFocus = (b, on) => {
    b.classList.toggle("focus", on);
    document.body.classList.toggle("ra-focused", on);
    b.querySelector(".ra-focus-btn").textContent = on ? "✕ Exit focus" : "⛶ Focus";
    sessionStorage.setItem("ra-focus", on ? "1" : "0");
    if (on) b.focus();
  };
  const resize = (b, step) => {
    const now = parseFloat(getComputedStyle(b).getPropertyValue("--ra-size")) || 1.15;
    const size = Math.min(2.2, Math.max(0.9, Math.round((now + step) * 100) / 100));
    b.style.setProperty("--ra-size", size + "rem");
    remember("size", size);
  };
  // The word at the text cursor (or the one just before it).
  const wordAtCursor = (b) => {
    const sel = getSelection();
    if (!sel.rangeCount) return null;
    let node = sel.anchorNode;
    if (node && node.nodeType === 3) node = node.parentElement;
    const word = node && node.closest && node.closest(".w");
    if (word) return word;
    return [...b.querySelectorAll(".w")].reverse().find((w) => sel.getRangeAt(0).comparePoint(w, 0) <= 0) || null;
  };
  // The edited text: paragraphs, with "Speaker N:" where a speaker badge is.
  const editedText = (b) => [...b.querySelectorAll(".ra-text p")].map((p) => {
    let out = "";
    p.childNodes.forEach((n) => {
      if (n.nodeType === 1 && n.classList.contains("ts")) return;
      if (n.nodeType === 1 && n.classList.contains("spk")) { out += "\\n\\n" + n.innerText.trim() + ": "; return; }
      out += n.textContent;
    });
    return out.trim();
  }).join("\\n\\n").trim();
  // Mark changed words and count them.
  const markChanges = (b) => {
    let n = 0;
    b.querySelectorAll(".ra-text .w").forEach((w) => {
      const changed = w.textContent.trim() !== w.dataset.o;
      w.classList.toggle("changed", changed);
      n += changed;
    });
    b.querySelector(".ra-dirty").textContent = n ? `● ${n} change${n > 1 ? "s" : ""} not saved` : "";
  };

  const startEdit = (b) => {
    const text = b.querySelector(".ra-text");
    saved.set(b, text.innerHTML);
    b.classList.add("editing");
    text.contentEditable = "true";
    text.querySelectorAll(".ts, .spk").forEach((n) => (n.contentEditable = "false"));
    text.focus();
  };
  const stopEdit = (b, keep) => {
    const text = b.querySelector(".ra-text");
    if (!keep) text.innerHTML = saved.get(b);
    text.contentEditable = "false";
    b.classList.remove("editing");
    b.querySelector(".ra-dirty").textContent = "";
  };
  const save = (b) => {
    const input = document.querySelector("#ra-edit-box textarea");
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
    setter.call(input, editedText(b));
    input.dispatchEvent(new Event("input", { bubbles: true }));
    stopEdit(b, true);
    setTimeout(() => document.querySelector("#ra-save-btn").click(), 60);
  };
  const playPause = (a) => (a.paused ? a.play() : a.pause());

  document.addEventListener("click", (e) => {
    const b = box(e.target);
    if (!b) return;
    const act = e.target.closest("[data-act]");
    const a = b.querySelector("audio");
    if (act && act.tagName === "BUTTON") {
      ({
        back: () => (a.currentTime = Math.max(0, a.currentTime - 5)),
        fwd: () => (a.currentTime = a.currentTime + 5),
        play: () => playPause(a),
        smaller: () => resize(b, -0.1), bigger: () => resize(b, 0.1),
        focus: () => setFocus(b, !b.classList.contains("focus")),
        edit: () => startEdit(b), save: () => save(b), cancel: () => stopEdit(b, false),
      })[act.dataset.act]();
      return;
    }
    const word = e.target.closest(".w, .ts");
    if (word && (!editing(b) || e.altKey)) {
      e.preventDefault();
      a.currentTime = parseFloat(word.dataset.s);
      a.play();
    }
  });
  document.addEventListener("input", (e) => { const b = box(e.target); if (b && editing(b)) markChanges(b); });
  document.addEventListener("change", (e) => {
    const b = box(e.target);
    if (!b || !e.target.dataset.act) return;
    if (e.target.dataset.act === "speed") {
      b.querySelector("audio").playbackRate = parseFloat(e.target.value);
      remember("speed", e.target.value);
    }
    if (e.target.dataset.act === "follow") {
      b.dataset.follow = e.target.checked ? "1" : "0";
      remember("follow", b.dataset.follow);
    }
  });
  document.addEventListener("play", (e) => {
    const b = box(e.target), speed = recall("speed");
    if (b && speed) e.target.playbackRate = parseFloat(speed);
  }, true);
  document.addEventListener("keydown", (e) => {
    // Keys go to the view the key was pressed in, or to the one in focus mode.
    const b = box(e.target) || document.querySelector(".readalong.focus");
    if (!b) return;
    const a = b.querySelector("audio");
    if (editing(b)) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); save(b); }
      else if (e.key === "Escape") { e.preventDefault(); stopEdit(b, false); }
      else if (e.code === "Space" && e.ctrlKey) {
        e.preventDefault();
        if (!a.paused) { a.pause(); return; }
        const word = wordAtCursor(b);
        if (word) a.currentTime = parseFloat(word.dataset.s);
        a.play();
      }
      return;
    }
    if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName)) return;  // their own keys
    const keys = {
      " ": () => playPause(a), ArrowLeft: () => (a.currentTime = Math.max(0, a.currentTime - 5)),
      ArrowRight: () => (a.currentTime = a.currentTime + 5), e: () => startEdit(b), E: () => startEdit(b),
      Escape: () => b.classList.contains("focus") && setFocus(b, false),
    };
    if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
  });
  document.addEventListener("timeupdate", (e) => {
    const b = box(e.target);
    if (!b) return;
    const words = b.querySelectorAll(".w"), t = e.target.currentTime;
    let lo = 0, hi = words.length - 1, now = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (parseFloat(words[mid].dataset.s) <= t) { now = mid; lo = mid + 1; } else hi = mid - 1;
    }
    const lit = b.querySelector(".w.now");
    if (now < 0 || lit === words[now]) return;
    if (lit) lit.classList.remove("now");
    words[now].classList.add("now");
    if (b.dataset.follow === "1" && !editing(b)) words[now].scrollIntoView({ block: "center", behavior: "smooth" });
  }, true);
})();
</script>
"""


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    # Hind Siliguri is designed for Bangla and has matching Latin letters.
    font=[gr.themes.GoogleFont("Hind Siliguri"), "Noto Sans Bengali", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_radius="14px",
    block_shadow="0 1px 3px rgb(15 23 42 / 0.06)",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="*neutral_500",
    block_label_text_color_dark="*neutral_400",
    block_title_text_color="*neutral_600",
    block_title_text_color_dark="*neutral_300",
    button_large_radius="10px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_background_fill_dark="*primary_600",
    button_primary_text_color="white",
)


CSS = """
.gradio-container { max-width: none !important; }
.main.fill_width { max-width: 1280px !important; margin: 0 auto !important; width: 100% !important; }
#header { align-items: center; border-bottom: 1px solid var(--border-color-primary); padding-bottom: 10px; }
#header h1 { font-size: 1.75rem; margin: 0; }
#header p { margin: 2px 0 0; color: var(--body-text-color-subdued); }
#ram { text-align: right; font-size: 0.95rem; }
.step-title { margin: 8px 0 0; font-size: 1.05rem; font-weight: 600; color: var(--color-accent); }
#claude-btn { background: #c96442; color: white; border: none; }
#claude-btn:hover { background: #b4573a; }
.stop-btn { background: #fef2f2 !important; color: #b91c1c !important; border: 1px solid #fecaca !important; }
.stop-btn:hover { background: #fee2e2 !important; }
.dark .stop-btn { background: #450a0a !important; color: #fecaca !important; border-color: #7f1d1d !important; }
.transcript textarea { font-size: 1.08rem !important; line-height: 1.8 !important; }
.srt textarea { font-family: var(--font-mono) !important; font-size: 0.85rem !important; line-height: 1.6 !important; }
.downloads button { min-height: 44px; }
.source { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
          border-radius: 14px; padding: 4px 12px 12px; }
.source .tabitem { padding: 10px 0 0; gap: 10px; }
.source button.lg, .source button.md { min-height: 44px; }
.pbar { position: relative; height: 36px; border-radius: 10px; overflow: hidden;
        background: var(--neutral-200); margin-bottom: 4px; }
.dark .pbar { background: var(--neutral-800); }
.pbar .fill { position: absolute; top: 0; bottom: 0; left: 0; background: var(--primary-500);
              opacity: 0.35; transition: width 0.4s ease; }
.pbar.busy .fill { opacity: 0.2; background-size: 36px 36px; animation: pbar-move 1s linear infinite;
                   background-image: repeating-linear-gradient(45deg, var(--primary-500) 0 9px,
                                                               transparent 9px 18px); }
.pbar.done .fill { opacity: 0.25; }
.pbar span { position: relative; line-height: 36px; padding: 0 14px; font-weight: 600; }
@keyframes pbar-move { from { background-position: 0 0; } to { background-position: 36px 0; } }
#status p { margin: 0; padding: 10px 14px; border-radius: 10px;
            background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
#status p:empty { display: none; }
#alert { display: none; }
.readalong { --ra-size: 1.15rem; outline: none; }
.readalong audio { width: 100%; margin-bottom: 4px; }
.ra-hint { font-size: 0.85rem; color: var(--body-text-color-subdued); margin-bottom: 6px; }
.ra-text { height: clamp(420px, calc(100vh - 330px), 1000px); overflow-y: auto; padding: 10px 16px;
           font-size: var(--ra-size); line-height: 2; border: 1px solid var(--border-color-primary);
           border-radius: 12px; background: var(--block-background-fill); }
.ra-text p { margin: 0 0 14px; }
/* Focus mode: the view takes the whole window */
.readalong.focus { position: fixed; inset: 0; z-index: 1000; display: flex; flex-direction: column;
                   padding: 14px max(16px, 8vw) 18px; background: var(--body-background-fill); }
.readalong.focus .ra-text { flex: 1; height: auto; padding: 16px 28px; }
body.ra-focused { overflow: hidden; }
.ra-text .w { cursor: pointer; border-radius: 4px; padding: 0 1px; }
.ra-text .w:hover { background: var(--color-accent-soft); }
.ra-text .w.now { background: var(--primary-200); }
.ra-text .w.changed { background: #fef3c7; box-shadow: inset 0 -2px 0 #f59e0b; }
.dark .ra-text .w.changed { background: #422006; box-shadow: inset 0 -2px 0 #d97706; }
.ra-dirty { color: #b45309; font-size: 0.85rem; font-weight: 600; margin-right: 6px; }
.dark .ra-text .w.now { background: var(--primary-800); }
.ra-text .ts { cursor: pointer; font-family: var(--font-mono); font-size: 0.78rem; margin-right: 8px;
               color: var(--body-text-color-subdued); }
.ra-text .spk { display: inline-block; font-size: 0.8rem; font-weight: 600; margin: 0 6px 0 2px;
                padding: 0 8px; border-radius: 999px; background: var(--color-accent-soft);
                color: var(--color-accent); }
.ra-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 2px 0 6px; }
.ra-bar button, .ra-bar select { border: 1px solid var(--border-color-primary); border-radius: 8px;
                                 background: var(--block-background-fill); padding: 4px 10px;
                                 font-size: 0.9rem; cursor: pointer; color: var(--body-text-color); }
.ra-bar button:hover { background: var(--color-accent-soft); }
.ra-bar label { display: flex; align-items: center; gap: 4px; font-size: 0.9rem; }
.ra-bar .ra-spacer { flex: 1; }
.ra-bar .ra-primary { background: var(--primary-600); color: white; border-color: var(--primary-600); }
.readalong .edit-only { display: none; }
.readalong.editing .edit-only { display: inline; }
.readalong.editing button.edit-only { display: inline-block; }
.readalong.editing .read-only { display: none; }
.readalong.editing .ra-text { outline: 2px solid var(--primary-400); background: var(--background-fill-primary);
                              cursor: text; }
.readalong.editing .ra-text .w { cursor: text; }
.ra-hidden { display: none !important; }
.ra-empty { padding: 24px; text-align: center; color: var(--body-text-color-subdued); }
#batch-table tbody tr { cursor: pointer; }
#batch-table, #batch-table * { font-family: var(--font) !important; }
#batch-table tbody tr:hover td { background: var(--color-accent-soft); }
.hint p { margin: 0; color: var(--body-text-color-subdued); }
.pbar a { color: var(--color-accent); font-weight: 600; }
"""


def step(title: str) -> gr.HTML:
    return gr.HTML(f'<div class="step-title">{title}</div>', padding=False)
