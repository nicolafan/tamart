"""tamart — Token Activation Map explainability for Multimodal LLMs.

Importing this package configures ``HF_HOME`` (and any other vars in the
repo-root ``.env``) before Hugging Face libraries are loaded, so models and
caches land under ``<repo>/data/hf`` by default.

``.env`` is authoritative for repo-scoped configuration: it overrides any
shell-level ``HF_HOME``. Relative paths in ``.env`` are resolved against the
repo root, and ``~`` is expanded, so the location is correct no matter where
scripts are run from.
"""
import os
from pathlib import Path

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_REPO_ROOT / ".env", override=True)

_hf_home = os.path.expanduser(os.environ.get("HF_HOME", "data/hf"))
if not os.path.isabs(_hf_home):
    _hf_home = str(_REPO_ROOT / _hf_home)
os.environ["HF_HOME"] = _hf_home
