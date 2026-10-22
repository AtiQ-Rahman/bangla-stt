"""The Batch tab's queue: add links/files, run them one after another."""
import shutil
import time
from pathlib import Path

import gradio as gr

from stt import history
from stt.audio import cut_clip, download_youtube
from stt.langid import detect
from stt.transcribe import LANGUAGE_MODELS
from webui.common import bar, check_ram, cookie_args, time_left, VIDEO_EXT, WORK_DIR
from webui.transcribing import _work, make_job


_queue_ids = iter(range(1, 10**9))


def queue_rows(queue: list[dict]) -> list[list]:
    """The batch queue as table rows; the last column says a row can be opened."""
    return [[n, f"{'🔗' if item['kind'] == 'link' else '🎞️'} {item['name']}", item.get("language", ""),
             item["status"], "📂 Open"] for n, item in enumerate(queue, 1)]


def queue_views(queue):
    """The table, plus the item picker for Remove (only items not yet done make sense there)."""
    choices = [(f"{n}. {item['name'][:60]}", item["id"]) for n, item in enumerate(queue, 1)]
    return queue_rows(queue), gr.Dropdown(choices=choices, value=[])


def add_links(text, queue):
    """Add every link in the box (one or many, any separators) and empty the box."""
    links = [w for w in (text or "").replace(",", " ").split() if w.startswith(("http://", "https://"))]
    if not links:
        raise gr.Error("Paste a YouTube link (starting with https://) first.")
    queue = queue + [{"id": next(_queue_ids), "kind": "link", "src": u, "name": u, "status": "waiting"}
                     for u in links]
    return ("", queue) + queue_views(queue)


def add_files(files, queue):
    queue = queue + [{"id": next(_queue_ids), "kind": "file", "src": f, "name": Path(f).name,
                      "status": "waiting"} for f in files or []]
    return (queue,) + queue_views(queue)


def remove_items(ids, queue):
    queue = [item for item in queue if item["id"] not in set(ids or [])]
    return (queue,) + queue_views(queue)


def clear_queue():
    return ([],) + queue_views([]) + (gr.HTML(visible=False),)


def run_batch(queue, lang_choice, quantize, use_lm, digits, cookies_file, use_browser):
    """Transcribe the queued items that aren't done yet, one after another; each goes to History."""
    queue = [dict(item) for item in queue]  # this run's copy; yielded back to the page
    todo = [item for item in queue if item["status"] != "✅ done"]
    if not todo:
        raise gr.Error("Nothing to do: add YouTube links or files to the queue first.")
    work = WORK_DIR / "batch"
    first = None
    for i, item in enumerate(todo):
        def show(status, frac=0.0):
            item["status"] = status
            overall = (i + frac) / len(todo)
            return (queue, queue_rows(queue),
                    bar(f"⏳ Item {i + 1} of {len(todo)}{time_left(first, overall)}", overall))

        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            yield show("getting the audio…")
            if item["kind"] == "link":
                path, title = download_youtube(item["src"], work / "download",
                                               *cookie_args(cookies_file, use_browser))
            else:
                path, title = Path(item["src"]), Path(item["src"]).stem
            item["name"] = title
            audio = cut_clip(path, work / "audio.mp3")
            language = lang_choice if lang_choice != "Auto" else detect(audio)[0]
            item["language"] = language
            model = LANGUAGE_MODELS[language][0]
            job = make_job(str(audio), title, model, quantize, use_lm, "", "", "", digits)
            if item["kind"] == "file" and Path(item["src"]).suffix.lower() in VIDEO_EXT:
                job["video"] = item["src"]
            if item["kind"] == "link":
                job["source"] = item["src"]
            check_ram(model, quantize, job["use_lm"])
            entry = history.create(job)
            item["entry"] = str(entry)
            yield show("loading the model…")
            for _, frac, _, _, _ in _work(job, entry, [], 0.0):
                first = first or (time.monotonic(), (i + frac) / len(todo))
                yield show(f"transcribing {frac:.0%}", frac)
            item["status"] = "✅ done"
        except Exception as e:  # one bad item shouldn't stop the rest
            item["status"] = f"❌ {e}"[:120]
        yield queue, queue_rows(queue), bar(f"⏳ Item {i + 1} of {len(todo)}", (i + 1) / len(todo))
    shutil.rmtree(work, ignore_errors=True)
    ok = sum(item["status"] == "✅ done" for item in todo)
    yield (queue, queue_rows(queue),
           bar(f"✅ Finished: {ok} of {len(todo)} done · open them in 🕘 History", 1, done=True))
