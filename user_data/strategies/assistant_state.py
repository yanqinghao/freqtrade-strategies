"""Atomic manual-plan storage shared by the strategy and Telegram thread."""
import fcntl
import json
import os
import tempfile
from pathlib import Path


def read_state(path):
    path = Path(path)
    if not path.exists():
        return {}
    state = json.loads(path.read_text())
    if not isinstance(state, dict):
        raise ValueError(f'Invalid assistant state: {path}')
    return state


def update_state(path, change):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_state(path)
        change(state)
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as target:
                json.dump(state, target, ensure_ascii=False, indent=2, allow_nan=False)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return state
