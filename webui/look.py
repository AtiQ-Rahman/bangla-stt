"""Theme, CSS and the alert sound."""
import gradio as gr


# Browsers only allow sound after the user has interacted with the page, so the first click
# anywhere unlocks an AudioContext; sttBeep() then plays a two-tone alert through it.
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
  window.sttBeep = () => {
    unlock();
    [[880, 0], [660, 0.28], [880, 0.56], [660, 0.84]].forEach(([freq, at]) => {
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
#batch-table tbody tr { cursor: pointer; }
#batch-table, #batch-table * { font-family: var(--font) !important; }
#batch-table tbody tr:hover td { background: var(--color-accent-soft); }
.hint p { margin: 0; color: var(--body-text-color-subdued); }
.pbar a { color: var(--color-accent); font-weight: 600; }
"""


def step(title: str) -> gr.HTML:
    return gr.HTML(f'<div class="step-title">{title}</div>', padding=False)
