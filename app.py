"""Gradio web UI: paste a YouTube link or upload audio/video, get a Bangla or English transcript.

The handlers live in the webui package; this file lays out the page and wires the events."""
import argparse
import json
import logging
import shutil
import signal
import subprocess
import sys
import threading
import traceback

import gradio as gr

from stt import history
from stt.transcribe import DEFAULT_MODEL, LANGUAGE_MODELS
from webui.batch import add_files, add_links, clear_queue, remove_items, run_batch
from webui.chaptering import claude_prompt, make_chapters, saved_chapters
from webui.claude import claude_sent, CLAUDE_TASKS, SEND_TO_CLAUDE_JS
from webui.common import bar_html, BROWSER, idle_unloader, ram_status, ram_tick
from webui.controls import (busy_buttons, clear_results, enable_if, idle_buttons, pick_language,
                            stopped, text_choices)
from webui.exporting import export_files, export_translation, export_video
from webui.fixing import refresh_llms, resume_correct, run_correct
from webui.look import BEEP_JS, CSS, READALONG_JS, step, THEME
from webui.opening import (follow_tick, history_choices, history_delete, history_open,
                           history_preview, history_search, open_queue_item, stop_following)
from webui.phone import USER as PHONE_USER, certificate, lan_ip, new_password
from webui.readalong import EMPTY as READALONG_EMPTY, readalong, save_readalong
from webui import watch
from webui.settings import (apply_defaults, default_llm, load as load_settings, reset_defaults,
                            save_defaults)
from webui.sources import (audio_changed, detect_language, get_audio_file, get_video_file,
                           get_youtube, get_youtube_video, selected_part, show_length)
from webui.transcribing import (edited_subtitles, edited_text, learn_edits, learned_words, resume, run,
                                set_learned_words)

threading.Thread(target=idle_unloader, daemon=True).start()


with gr.Blocks(title="Bangla STT", delete_cache=(3600, 3600), fill_width=True) as demo:
    title = gr.State("")
    ram_state = gr.State("ok")
    busy = gr.State("")  # what this page is doing: "", "transcribe", "fix", "export" or "batch"
    partial = gr.State(None)  # a stopped transcription that Resume can continue
    fix_partial = gr.State(None)  # a stopped AI fix that Resume fix can continue
    segs = gr.State(None)  # the timed pieces of the current transcript, for subtitles
    entry = gr.State(None)  # the History folder of the current transcript
    video_src = gr.State(None)  # the video the audio came from, for subtitled video export
    source = gr.State(None)  # the YouTube link the audio came from, kept with the run
    follow = gr.State(None)  # History folder of a running transcription this page shows live
    follow_timer = gr.Timer(1.5, active=False)
    batch_queue = gr.State([])  # the Batch tab's items: {id, kind, src, name, status, language}

    with gr.Row(elem_id="header"):
        gr.Markdown("# 🎙️ Bangla & English Speech-to-Text\n"
                    "Runs entirely on this PC — no audio is sent anywhere.")
        ram = gr.Markdown(ram_status(), elem_id="ram")
        alert = gr.HTML("", elem_id="alert", padding=False)

    with gr.Tabs() as pages:
        with gr.Tab("🎙️ Transcribe", id="main"):
            with gr.Row():
                # ---- Left: input ------------------------------------------------------------
                with gr.Column(scale=2):
                    step("① Get the audio")
                    with gr.Tabs(elem_classes="source"):
                        with gr.Tab("▶ YouTube"):
                            youtube_url = gr.Textbox(show_label=False, container=False,
                                                     placeholder="Paste a YouTube link here…")
                            with gr.Row(equal_height=True):
                                get_yt = gr.Button("⬇ Audio only", variant="primary")
                                get_yt_video = gr.Button("🎬 Video + audio")
                            save_video = gr.DownloadButton("💾 Save the video", visible=False)
                        with gr.Tab("💻 From your computer"):
                            with gr.Row(equal_height=True):
                                video_btn = gr.UploadButton("🎬 Video file", file_types=["video"],
                                                            type="filepath")
                                audio_btn = gr.UploadButton("🎵 Audio file", file_types=["audio"],
                                                            type="filepath")
                    audio = gr.Audio(label="Audio — listen, ✂ trim, or 🎤 record", type="filepath",
                                     sources=["microphone"], buttons=["download"])
                    with gr.Row():
                        start = gr.Textbox(label="From (optional)", placeholder="0:00")
                        end = gr.Textbox(label="To (optional)", placeholder="end")
                    clip = gr.Audio(label="Selected part", type="filepath", interactive=False,
                                    visible=False, buttons=["download"])
                    language = gr.Radio(["Bangla", "English"], value="Bangla", label="Language of the speech")
                    with gr.Accordion("⚙️ Settings", open=False):
                        model_name = gr.Radio(LANGUAGE_MODELS["Bangla"], value=DEFAULT_MODEL, label="Model")
                        quantize = gr.Checkbox(value=True, label="Fast mode (int8): ~1.7x faster, less RAM, "
                                                                 "same accuracy")
                        use_lm = gr.Checkbox(value=True, label="Bangla spell-fixing language model "
                                                               "(3.5 GB download on first use)")
                        digits = gr.Checkbox(value=False, label="Write numbers in Bangla digits (২০, not 20)")
                        speakers = gr.Checkbox(
                            value=False, label="Label who is speaking (Speaker 1, 2 …) — takes about a "
                                               "quarter longer; for interviews and talk shows")
                        hotwords = gr.Textbox(
                            label="Names or special words in the audio (optional, comma separated)",
                            placeholder="e.g. রাজশাহী, তারেক রহমান, প্রথম আলো",
                            info="Helps the right spelling of names win when the audio sounds close to them.")
                        use_learned = gr.Checkbox(
                            value=True, label="Also use the words I corrected before (learned from my edits)")
                        learned_box = gr.Textbox(
                            label="Words learned from my edits (this language)", lines=2,
                            placeholder="Empty so far: fix a word in the transcript or Read along and it "
                                        "is remembered here",
                            info="Comma separated. Remove a wrong one, or add your own; saved when you "
                                 "click outside the box.")
                        use_browser = gr.Checkbox(
                            value=bool(BROWSER), visible=bool(BROWSER),
                            label=f"If YouTube says “confirm you're not a bot”, use my "
                                  f"{(BROWSER or ('',))[0].title()} login (cookies stay on this PC)")
                        cookies_file = gr.File(
                            label="YouTube cookies.txt (only if YouTube still blocks the download)",
                            file_types=[".txt"], type="filepath")
                        with gr.Row():
                            save_defaults_btn = gr.Button("💾 Save these as my defaults", size="sm")
                            reset_defaults_btn = gr.Button("↺ Built-in defaults", size="sm")
                        defaults_msg = gr.Markdown(
                            "Saves the language, models and options here, plus the Claude task, the "
                            "local AI model and the Batch language, for every time the app opens.",
                            elem_classes="hint")
                    with gr.Row():
                        btn = gr.Button("▶ Transcribe", variant="primary", size="lg", scale=3,
                                        interactive=False)  # until there is audio
                        resume_btn = gr.Button("⏯ Resume", size="lg", scale=2, visible=False)
                        stop = gr.Button("■ Stop", size="lg", scale=1, min_width=90,
                                         elem_classes="stop-btn", interactive=False)
                    status = gr.Markdown("", elem_id="status")

                # ---- Right: result ----------------------------------------------------------
                with gr.Column(scale=3):
                    step("② Result")
                    with gr.Row(equal_height=True):
                        pbar = gr.HTML("", padding=False, visible=False, scale=1)  # while transcribing
                        stop_top = gr.Button("■ Stop", visible=False, scale=0, min_width=100,
                                             elem_classes="stop-btn")
                        resume_top = gr.Button("⏯ Resume", visible=False, scale=0, min_width=120,
                                               variant="primary")
                    with gr.Tabs():
                        with gr.Tab("Transcript"):
                            transcript = gr.Textbox(
                                show_label=False, lines=16, max_lines=16, interactive=True,
                                buttons=["copy"], elem_classes="transcript",
                                placeholder="The transcript appears here. You can edit it; "
                                            "the downloads follow your edits.")
                        with gr.Tab("▶ Read along") as readalong_tab:
                            readalong_view = gr.HTML(READALONG_EMPTY, padding=False)
                            # Filled and pressed by the page script when an edit is saved.
                            ra_edit = gr.Textbox(elem_id="ra-edit-box", elem_classes="ra-hidden",
                                                 show_label=False, container=False)
                            ra_save = gr.Button("save", elem_id="ra-save-btn", elem_classes="ra-hidden")
                        with gr.Tab("Subtitles (.srt)"):
                            srt_view = gr.Textbox(show_label=False, lines=16, max_lines=16,
                                                  buttons=["copy"], elem_classes="srt", autoscroll=False)
                        with gr.Tab("AI-fixed (local)"):
                            with gr.Row(equal_height=True):
                                llm = gr.Dropdown([], label="Ollama model", allow_custom_value=True, scale=3)
                                refresh = gr.Button("↻", scale=0, min_width=44)
                                fix_btn = gr.Button("✨ Fix transcript", variant="primary", scale=2,
                                                    interactive=False)  # until there is a transcript
                                fix_resume_btn = gr.Button("⏯ Resume fix", scale=2, visible=False)
                            fix_bar = gr.HTML("", padding=False, visible=False)
                            corrected = gr.Textbox(
                                show_label=False, lines=12, max_lines=12, interactive=True,
                                buttons=["copy"], elem_classes="transcript",
                                placeholder="Press 'Fix transcript' to add punctuation and fix spelling "
                                            "with a free AI on this PC (Ollama). Slower and less accurate "
                                            "than Claude; check the result.")
                            with gr.Row(elem_classes="downloads"):
                                fixed_txt_btn = gr.DownloadButton("⬇ Fixed text (.txt)", interactive=False)
                                fixed_srt_btn = gr.DownloadButton("⬇ Fixed subtitles (.srt)",
                                                                  interactive=False)
                        with gr.Tab("📑 Chapters") as chapters_tab:
                            gr.Markdown("A chapter list with times — ready for a YouTube description "
                                        "— and a short summary of each part. Made by the local AI "
                                        "(the Ollama model in *AI-fixed*); best for long recordings. "
                                        "Check names: the small local AI can get them wrong.",
                                        elem_classes="hint")
                            chapters_btn = gr.Button("📑 Make chapters", variant="primary",
                                                     interactive=False)  # until there is a transcript
                            chapters_bar = gr.HTML("", padding=False, visible=False)
                            chapters_out = gr.Textbox(
                                show_label=False, lines=12, max_lines=12, interactive=True,
                                buttons=["copy"], elem_classes="transcript",
                                placeholder="00:00 Introduction\n03:40 …")
                            chapters_file = gr.DownloadButton("⬇ Chapters (.txt)", interactive=False)
                            with gr.Accordion("Or ask Claude (better chapters, needs internet)", open=False):
                                claude_chapters_btn = gr.Button("📋 Make a prompt for Claude", size="sm")
                                claude_chapters = gr.Textbox(
                                    show_label=False, lines=4, max_lines=4, buttons=["copy"], visible=False,
                                    info="Copy it (button at the top right) and paste it at claude.ai/new")
                        with gr.Tab("⬇ Export"):
                            export_which = gr.Radio(["Transcript"], value="Transcript",
                                                    label="Export which text")
                            export_btn = gr.Button("📄 Make .vtt subtitles and a Word (.docx) file")
                            export_out = gr.File(label="Files", file_count="multiple", visible=False)
                            gr.Markdown("**Video with subtitles** — for audio that came from a video "
                                        "(🎬 Video + audio, or 🎬 Video file).")
                            video_btn_burn = gr.Button("🎬 Make the video with subtitles")
                            export_bar = gr.HTML("", padding=False, visible=False)
                            save_subtitled = gr.DownloadButton("💾 Save video with subtitles", visible=False)
                            gr.Markdown("**Subtitles in the other language** — Bangla ↔ English with the "
                                        "local AI (Ollama; see the AI-fixed tab), timed like the originals.")
                            translate_btn = gr.Button("🌐 Translate the subtitles")
                            translate_bar = gr.HTML("", padding=False, visible=False)
                            translate_out = gr.File(label="Translated subtitles", file_count="multiple",
                                                    visible=False)
                    with gr.Row(elem_classes="downloads"):
                        txt_btn = gr.DownloadButton("⬇ Transcript (.txt)", interactive=False)
                        srt_btn = gr.DownloadButton("⬇ Subtitles (.srt)", interactive=False)
                        clear = gr.Button("✕ Clear results", min_width=120)

                    step("③ Send to Claude")
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            claude_text = gr.Radio(["Transcript"], value="Transcript", label="Text to send")
                            claude_task = gr.Dropdown(list(CLAUDE_TASKS), value=next(iter(CLAUDE_TASKS)),
                                                      label="Ask Claude to…")
                        claude_custom = gr.Textbox(label="Your instruction (only for Custom)", lines=1,
                                                   placeholder="e.g. Turn this into a news article")
                    claude_btn = gr.Button("Send to Claude ↗", size="lg", elem_id="claude-btn")
                    claude_status = gr.Markdown("", elem_id="claude-status")

        with gr.Tab("📚 Batch", id="batch"):
            gr.Markdown("Queue up YouTube links and audio/video files, then start — they are "
                        "transcribed one after another and saved in **🕘 History**. Uses the "
                        "Settings of the Transcribe tab.")
            with gr.Row():
                with gr.Column(scale=2):
                    step("① Add to the queue")
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            batch_link = gr.Textbox(show_label=False, container=False, scale=4,
                                                    placeholder="Paste YouTube link(s), then Add or Enter")
                            batch_add = gr.Button("➕ Add", variant="primary", scale=1, min_width=90)
                        batch_upload = gr.UploadButton("🎬🎵 Add audio / video files (pick many at once)",
                                                       file_types=["audio", "video"],
                                                       file_count="multiple", type="filepath")
                    batch_lang = gr.Radio(["Auto", "Bangla", "English"], value="Auto",
                                          label="Language (Auto detects each one)")
                    with gr.Accordion("📂 Watch a folder", open=False):
                        watch_folder = gr.Textbox(label="Folder", info="Audio and video files put here "
                                                  "are transcribed automatically (not its subfolders).")
                        watch_on = gr.Checkbox(False, label="Watch it (while this page is open)")
                        watch_msg = gr.Markdown("Not watching.", elem_classes="hint")
                        watch_timer = gr.Timer(10, active=False)
                        watch_start = gr.Textbox(visible=False)  # a new value presses ▶ Start
                    with gr.Row():
                        batch_btn = gr.Button("▶ Start", variant="primary", size="lg", scale=3,
                                              elem_id="batch-start")
                        batch_stop = gr.Button("■ Stop", size="lg", scale=1, elem_classes="stop-btn",
                                               interactive=False)
                with gr.Column(scale=3):
                    step("② Queue")
                    batch_bar = gr.HTML("", padding=False, visible=False)
                    gr.Markdown("Click a row to open it in the 🎙️ Transcribe dashboard — transcript, "
                                "subtitles, AI fix, export, Claude, Resume.", elem_classes="hint")
                    batch_table = gr.Dataframe(
                        headers=["#", "Item", "Language", "Status", ""], value=[],
                        datatype=["number", "str", "str", "str", "str"], interactive=False, wrap=True,
                        column_widths=["48px", "45%", "90px", "25%", "90px"], elem_id="batch-table")
                    with gr.Row(equal_height=True):
                        batch_pick = gr.Dropdown([], multiselect=True, label="Remove from the queue",
                                                 scale=4)
                        batch_remove = gr.Button("🗑 Remove", scale=1, min_width=100)
                        batch_clear = gr.Button("✕ Empty queue", scale=1, min_width=150)

        with gr.Tab("🕘 History", id="history"):
            gr.Markdown("Every transcription is kept here (in the `history` folder of the app), even "
                        "after the app is closed. Open one to read, download or **Resume** it.")
            history_query = gr.Textbox(show_label=False, container=False,
                                       placeholder="🔍 Search words in all transcripts…")
            history_found = gr.Markdown("")
            with gr.Row(equal_height=True):
                history_pick = gr.Dropdown(history.entries(), label="Saved transcriptions", scale=4)
                history_refresh = gr.Button("↻", scale=0, min_width=44)
            history_info = gr.Markdown("")
            with gr.Row():
                history_open_btn = gr.Button("📂 Open", variant="primary")
                history_delete_btn = gr.Button("🗑 Delete", elem_classes="stop-btn")

    # ---- events ---------------------------------------------------------------------------
    # Buttons switched together by busy_buttons()/idle_buttons(), then the page's busy state.
    controls = [btn, stop, stop_top, fix_btn, clear, get_yt, get_yt_video, resume_btn, resume_top,
                batch_btn, batch_stop, batch_add, batch_upload, batch_remove, batch_clear,
                fix_resume_btn, video_btn_burn, translate_btn, chapters_btn, busy]
    run_outputs = [transcript, txt_btn, srt_btn, srt_view, status, clip, pbar, partial, segs, entry]
    idle_inputs = [audio, transcript, partial, fix_partial]

    transcribe_event = btn.click(busy_buttons("transcribe"), None, controls).then(
        run, [audio, title, model_name, quantize, use_lm, hotwords, start, end, digits, speakers,
              use_learned, video_src, source],
        run_outputs,
        show_progress="hidden")  # our own progress bar above the transcript shows it
    resume_event = resume_btn.click(busy_buttons("transcribe"), None, controls).then(
        resume, partial, run_outputs, show_progress="hidden")
    resume_top_event = resume_top.click(busy_buttons("transcribe"), None, controls).then(
        resume, partial, run_outputs, show_progress="hidden")
    fix_inputs = [transcript, llm, language, segs, title, entry, fix_partial]
    fix_outputs = [corrected, fixed_txt_btn, fixed_srt_btn, fix_bar, fix_partial]
    correct_event = fix_btn.click(busy_buttons("fix"), None, controls).then(
        run_correct, fix_inputs, fix_outputs, show_progress="hidden")
    fix_resume_event = fix_resume_btn.click(busy_buttons("fix"), None, controls).then(
        resume_correct, fix_inputs, fix_outputs, show_progress="hidden")
    translate_event = translate_btn.click(busy_buttons("translate"), None, controls).then(
        export_translation, [export_which, transcript, corrected, segs, title, language, llm],
        [translate_bar, translate_out], show_progress="hidden")
    export_event = video_btn_burn.click(busy_buttons("export"), None, controls).then(
        export_video, [export_which, transcript, corrected, segs, title, video_src],
        [export_bar, save_subtitled], show_progress="hidden")
    chapters_event = chapters_btn.click(busy_buttons("chapters"), None, controls).then(
        make_chapters, [transcript, segs, llm, language, title, entry],
        [chapters_out, chapters_file, chapters_bar], show_progress="hidden")
    chapters_tab.select(saved_chapters, entry, chapters_out)
    claude_chapters_btn.click(claude_prompt, [transcript, segs, language], claude_chapters)
    batch_event = batch_btn.click(busy_buttons("batch"), None, controls).then(
        run_batch, [batch_queue, batch_lang, quantize, use_lm, digits, speakers, use_learned,
                    cookies_file, use_browser],
        [batch_queue, batch_table, batch_bar], show_progress="hidden")
    for trigger in (batch_add.click, batch_link.submit):
        trigger(add_links, [batch_link, batch_queue], [batch_link, batch_queue, batch_table, batch_pick])
    batch_upload.upload(add_files, [batch_upload, batch_queue], [batch_queue, batch_table, batch_pick])
    batch_remove.click(remove_items, [batch_pick, batch_queue], [batch_queue, batch_table, batch_pick])
    batch_clear.click(clear_queue, None, [batch_queue, batch_table, batch_pick, batch_bar])
    demo.load(lambda: load_settings()["watch_folder"], None, watch_folder)
    watch_on.change(watch.toggle, [watch_on, watch_folder], [watch_timer, watch_msg, watch_on])
    watch_timer.tick(watch.tick, [watch_on, watch_folder, batch_queue, busy],
                     [batch_queue, batch_table, batch_pick, watch_start], show_progress="hidden")
    watch_start.change(None, None, None, js="() => { const b = document.querySelector('#batch-start'); "
                                            "if (b && !b.disabled) b.click(); }")
    running = [transcribe_event, resume_event, resume_top_event, correct_event, fix_resume_event,
               export_event, translate_event, batch_event, chapters_event]
    # When long work ends well (not stopped, no error): chime, and notify if the page is hidden.
    # Starting it is a click, the moment a browser allows asking for notification permission.
    finished = [(transcribe_event, status, "Transcription finished"),
                (resume_event, status, "Transcription finished"),
                (resume_top_event, status, "Transcription finished"),
                (correct_event, fix_bar, "AI fix finished"), (fix_resume_event, fix_bar, "AI fix finished"),
                (export_event, export_bar, "Video with subtitles ready"), (batch_event, batch_bar, "Batch finished"),
                (translate_event, translate_bar, "Translated subtitles ready"),
                (chapters_event, chapters_bar, "Chapters ready")]
    for start_button in (btn, resume_btn, resume_top, fix_btn, fix_resume_btn, video_btn_burn,
                         translate_btn, batch_btn, chapters_btn):
        start_button.click(None, js="() => window.sttAskNotify && window.sttAskNotify()")
    # Read along: render when its tab is opened, and after a run or an open brings a new result.
    readalong_inputs = [transcript, segs, entry]
    readalong_tab.select(readalong, readalong_inputs, readalong_view)
    ra_save.click(save_readalong, [ra_edit, segs, title, entry],
                  [transcript, txt_btn, srt_btn, srt_view, readalong_view]).then(
        learn_edits, [transcript, segs, entry, language], learned_box)
    transcript.blur(readalong, readalong_inputs, readalong_view)  # edits there show in Read along too
    for event in (transcribe_event, resume_event, resume_top_event):
        event.success(readalong, readalong_inputs, readalong_view)
    for event, message, title_text in finished:
        event.success(None, message, None,
                      js=f"(m) => window.sttNotify && window.sttNotify({json.dumps(title_text)}, m)")
    for event in running:  # .then runs after errors too; Stop restores the buttons itself
        event.then(idle_buttons, idle_inputs, controls)
    stopping = bar_html("⏹ Stopping — finishing the current step…", done=True)
    for button in (stop, stop_top, batch_stop):
        # Instant feedback from the browser: while a model loads the server can take a few
        # seconds to answer.
        button.click(None, None, [status, pbar], js=f"() => ['⏹ Stopping…', {json.dumps(stopping)}]")
        # queue=False: answer at once instead of waiting behind the busy queue.
        button.click(stopped, idle_inputs + [busy], [status, pbar, fix_bar] + controls,
                     cancels=running, queue=False)
    clear.click(clear_results, None,
                [start, end, transcript, txt_btn, srt_btn, srt_view, corrected, fixed_txt_btn,
                 fixed_srt_btn, status, clip, pbar, stop_top, claude_status, partial, resume_btn,
                 resume_top, segs, entry, fix_bar, fix_partial, fix_resume_btn, chapters_out],
                cancels=running)

    # Getting audio
    got_audio = [audio, title, status, video_src, source]
    get_yt.click(get_youtube, [youtube_url, cookies_file, use_browser], got_audio, show_progress_on=[audio])
    youtube_url.submit(get_youtube, [youtube_url, cookies_file, use_browser], got_audio,
                       show_progress_on=[audio])
    get_yt_video.click(get_youtube_video, [youtube_url, cookies_file, use_browser],
                       [audio, title, status, save_video, video_src, source], show_progress_on=[audio])
    video_btn.upload(get_video_file, video_btn, got_audio, show_progress_on=[audio])
    audio_btn.upload(get_audio_file, audio_btn, got_audio, show_progress_on=[audio])
    audio.stop_recording(lambda: ("recording", None, None), None, [title, video_src, source])
    # Starting something new on this page ends following a background transcription.
    for trigger in (btn.click, resume_btn.click, resume_top.click, clear.click, get_yt.click,
                    get_yt_video.click, youtube_url.submit, video_btn.upload, audio_btn.upload,
                    audio.stop_recording):
        trigger(stop_following, None, [follow, follow_timer])
    # New or trimmed audio: length hint, From..To preview, language, buttons, stale Resume.
    audio.change(show_length, audio, end).then(selected_part, [audio, start, end], clip).then(
        detect_language, [audio, language, transcript], [language, status])
    audio.change(enable_if, [audio, busy], btn)
    audio.change(audio_changed, [audio, partial], [partial, resume_btn, resume_top])
    demo.load(enable_if, [audio, busy], btn)  # for audio that is already there when the page opens
    for trigger in (start.blur, start.submit, end.blur, end.submit):
        trigger(selected_part, [audio, start, end], clip)
    language.change(pick_language, language, [model_name, quantize, use_lm, digits, hotwords])

    # Edits: the downloads follow the text boxes
    transcript.change(enable_if, [transcript, busy], fix_btn)
    transcript.change(enable_if, [transcript, busy], chapters_btn)
    transcript.input(lambda t, ti, en: edited_text(t, ti, en, "transcript.txt"),
                     [transcript, title, entry], txt_btn, trigger_mode="always_last")
    transcript.blur(lambda t, s, ti: edited_subtitles(t, s, ti, ".srt"), [transcript, segs, title],
                    [srt_btn, srt_view])
    transcript.blur(learn_edits, [transcript, segs, entry, language], learned_box)  # see stt.dictionary
    language.change(learned_words, language, learned_box)
    demo.load(learned_words, language, learned_box)
    learned_box.blur(set_learned_words, [learned_box, language], learned_box)
    corrected.input(lambda t, ti, en: edited_text(t, ti, en, "corrected.txt"),
                    [corrected, title, entry], fixed_txt_btn, trigger_mode="always_last")
    corrected.blur(lambda t, s, ti: edited_subtitles(t, s, ti, " (fixed).srt")[0],
                   [corrected, segs, title], fixed_srt_btn)

    corrected.change(text_choices, corrected, [claude_text, export_which])

    # Export
    export_btn.click(export_files, [export_which, transcript, corrected, segs, title], export_out)

    # History
    history_refresh.click(history_choices, history_query, history_pick)
    pages.select(history_choices, history_query, history_pick)  # fresh list whenever a page is opened
    history_query.input(history_search, history_query, [history_pick, history_found],
                        trigger_mode="always_last")
    history_pick.change(history_preview, history_pick, history_info)
    open_outputs = [pages, audio, title, language, transcript, txt_btn, srt_btn, srt_view, corrected,
                    status, partial, segs, entry, resume_btn, resume_top, video_src, youtube_url,
                    source, pbar, follow, follow_timer]  # in OPEN_FIELDS order
    history_open_btn.click(history_open, history_pick, open_outputs).then(
        readalong, readalong_inputs, readalong_view)
    batch_table.select(open_queue_item, batch_queue, open_outputs).then(
        readalong, readalong_inputs, readalong_view)
    follow_timer.tick(follow_tick, follow, open_outputs, show_progress="hidden")
    history_delete_btn.click(history_delete, history_pick, [history_pick, history_info])

    # Claude, LLM list, RAM
    copied = gr.Checkbox(False, visible=False)  # filled by the browser (a State can't be)
    claude_btn.click(claude_sent, [transcript, corrected, claude_text, claude_task, claude_custom,
                                   language, copied], claude_status, js=SEND_TO_CLAUDE_JS)
    refresh.click(refresh_llms, None, llm)
    demo.load(default_llm, None, llm)
    # Defaults: applied when the page opens; saved / reset from Settings.
    default_outputs = [language, model_name, quantize, use_lm, digits, speakers, use_learned, use_browser,
                       claude_task, batch_lang]
    demo.load(apply_defaults, None, default_outputs)
    save_defaults_btn.click(save_defaults, [language, model_name, quantize, use_lm, digits, speakers,
                                            use_learned, use_browser, claude_task, llm, batch_lang],
                            defaults_msg)
    reset_defaults_btn.click(reset_defaults, None, default_outputs + [defaults_msg])
    gr.Timer(5).tick(ram_tick, ram_state, [ram, ram_state, alert], show_progress="hidden").then(
        None, alert, None, js="(signal) => { if (signal) window.sttBeep && window.sttBeep(); }")


def open_browser(url: str):
    """Open the app in the default browser without its startup chatter in this terminal
    (a browser that is already running prints e.g. "zygote … Broken pipe" while handing over)."""
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    else:
        import webbrowser
        webbrowser.open(url)


class _HideCancelRace(logging.Filter):
    """Gradio 6.30 may get two cancel requests for one event and fail the second with a
    KeyError in cancel_event (the event is already cancelled). Harmless; don't print it."""

    def filter(self, record):
        exc = record.exc_info and record.exc_info[1]
        return not (isinstance(exc, KeyError) and "cancel_event" in "".join(
            traceback.format_tb(record.exc_info[2])))


def main():
    parser = argparse.ArgumentParser(description="Bangla & English speech-to-text in the browser.")
    parser.add_argument("--phone", action="store_true",
                        help="also open it to phones on the same Wi-Fi (HTTPS, with a password)")
    parser.add_argument("--password", help="password for --phone (default: a new random one)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.getLogger("uvicorn.error").addFilter(_HideCancelRace())
    # A plain `kill` should end the app like Ctrl+C does: exit handlers then stop the worker.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    history.HISTORY_DIR.mkdir(exist_ok=True)
    options = {}
    if args.phone:
        ip = lan_ip()
        cert, key = certificate(ip)
        password = args.password or new_password()
        options = {"server_name": "0.0.0.0", "ssl_certfile": str(cert), "ssl_keyfile": str(key),
                   "ssl_verify": False, "auth": (PHONE_USER, password),
                   "auth_message": "Bangla STT — enter the password shown in the PC's terminal."}
    # No footer links: the API docs, settings and run history are developer tools; History
    # above keeps the results. allowed_paths: History audio is played and downloaded from there.
    demo.queue().launch(theme=THEME, css=CSS, head=BEEP_JS + READALONG_JS, footer_links=[], run_history=False,
                        allowed_paths=[str(history.HISTORY_DIR)], prevent_thread_lock=True, quiet=True,
                        server_port=args.port, **options)
    print(f"Bangla STT is running at {demo.local_url}  —  press Ctrl+C here to stop it.", flush=True)
    if args.phone:
        print(f"\nOn your phone (same Wi-Fi) open  https://{ip}:{args.port}\n"
              f"  user: {PHONE_USER}   password: {password}\n"
              "  The phone warns that the connection isn't private (the certificate is made by this\n"
              "  PC): tap Advanced → Proceed. If it can't connect, the PC's firewall may block the port.",
              flush=True)
    open_browser(demo.local_url)
    demo.block_thread()


if __name__ == "__main__":
    main()
