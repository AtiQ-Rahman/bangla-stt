"""What the app is doing in the background that the user would otherwise not see (internet
downloads, models loading, the LLM at work); the page header shows it."""
from contextlib import contextmanager

_now: dict[object, str] = {}

# Set in the worker process: forward(key, text) sends each change to the app (whose Worker shows
# it), because the header reads only the app's own _now.
forward = None


def show(key, text: str | None) -> None:
    """Show text under key, replacing what key showed before; None removes it."""
    if forward:
        forward(id(key), text)
    elif text:
        _now[key] = text
    else:
        _now.pop(key, None)


@contextmanager
def doing(text: str):
    """Show text while the with-block runs; yields a function that changes the text."""
    key = object()
    show(key, text)
    try:
        yield lambda new: show(key, new)
    finally:
        show(key, None)


def current() -> list[str]:
    """What is going on right now, oldest first, without repeats."""
    return list(dict.fromkeys(list(_now.values())))
