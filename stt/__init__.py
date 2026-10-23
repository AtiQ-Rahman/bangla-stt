"""Speech-to-text for Bangla and English."""
from huggingface_hub.utils import logging as hub_logging

# The Hub sends notices such as "set a HF_TOKEN for higher rate limits" with each request;
# they are not actionable for this app, so only errors are shown.
hub_logging.set_verbosity_error()


def cached_first(load):
    """Call load(local_files_only=True) so cached models load without asking the Hugging Face
    Hub (no network wait, no Hub notices); only if something is missing, go online."""
    try:
        return load(local_files_only=True)
    except Exception:  # each library raises its own "not in cache" error
        from stt.activity import doing

        with doing("🌐 Downloading a model from the internet (first time only)…"):
            return load(local_files_only=False)
