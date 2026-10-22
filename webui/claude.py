"""Send to Claude: copy a prompt with the text and open claude.ai."""
import json


# {lang} / {other} are filled in by the browser from the chosen language.
CLAUDE_TASKS = {
    "Fix spelling & punctuation": (
        "Below is a {lang} transcript produced by a speech-recognition model. It may have spelling "
        "mistakes, wrongly split or joined words, and missing punctuation. Rewrite it in correct "
        "standard {lang} with proper punctuation. Do not add, remove or summarise anything."),
    "Summarise (bullet points)": "Summarise this {lang} transcript in {lang} as short bullet points.",
    "Translate (Bangla ↔ English)": (
        "Translate this {lang} transcript into natural {other}. It came from speech recognition, "
        "so silently fix obvious recognition mistakes."),
    "Custom (write below)": "",
}


# Runs in the browser: copy prompt + transcript, then open claude.ai with it prefilled
# when it fits in a URL (otherwise the user pastes it).
SEND_TO_CLAUDE_JS = """
async (transcript, fixed, which, task, custom, language) => {
    const tasks = %s;
    transcript = transcript || ""; fixed = fixed || ""; custom = custom || "";  // empty boxes are null
    const text = which.startsWith("AI") ? fixed : transcript;
    if (!text.trim() || (task.startsWith("Custom") && !custom.trim())) {
        return [text, fixed, which, task, custom, language, false];  // the server explains
    }
    const other = language === "English" ? "Bangla" : "English";
    const instruction = task.startsWith("Custom") ? custom
        : tasks[task].replaceAll("{lang}", language).replaceAll("{other}", other);
    const full = (instruction ? instruction + "\\n\\n" : "") + "Transcript:\\n" + text;
    // Open the tab first, while the click still counts as a user action (popup blockers).
    const q = encodeURIComponent(full);
    window.open("https://claude.ai/new" + (q.length < 14000 ? "?q=" + q : ""), "_blank");
    let copied = false;
    try { await navigator.clipboard.writeText(full); copied = true; } catch (e) {
        const area = document.createElement("textarea");  // fallback for blocked clipboard API
        area.value = full; document.body.appendChild(area); area.select();
        copied = document.execCommand("copy"); area.remove();
    }
    return [text, fixed, which, task, custom, language, copied];
}
""" % json.dumps(CLAUDE_TASKS)


def claude_sent(text, _fixed, which, task, custom, language, copied):
    # Shown under the button rather than raised: the button's error popup was easy to miss.
    if not (text or "").strip():
        return ("⚠️ There is no AI-fixed text yet: use the **AI-fixed** tab first."
                if which.startswith("AI") else "⚠️ Transcribe something first.")
    if task.startswith("Custom") and not (custom or "").strip():
        return "⚠️ Write what Claude should do in the **Your instruction** box."
    if copied:
        return ("📋 **Copied for Claude.** In the Claude tab that just opened, click the message box, "
                "press **Ctrl+V**, then Enter. (Short texts are filled in already.)")
    return ("⚠️ Your browser blocked copying. Use the copy button on the text box, then paste it "
            "into Claude with Ctrl+V.")
