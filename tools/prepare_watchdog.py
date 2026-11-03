"""Prepare an ignored Telegram-only configuration; never print credentials."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
from assistant_state import read_state, update_state


def prepare_watchdog(source, target):
    telegram = read_state(source).get('telegram', {})
    if (
        not telegram.get('enabled')
        or not isinstance(telegram.get('token'), str)
        or not telegram['token']
        or telegram.get('chat_id') is None
    ):
        raise ValueError('Enabled Telegram token/chat_id required')
    minimal = {
        key: telegram[key]
        for key in ('token', 'chat_id', 'topic_id')
        if telegram.get(key) is not None
    }

    def replace(document):
        document.clear()
        document.update(minimal)

    update_state(target, replace)
    os.chmod(target, 0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='deploy/config.json')
    parser.add_argument('--output', default='deploy/watchdog.json')
    parser.add_argument('--runtime-root', default='user_data')
    args = parser.parse_args()
    try:
        prepare_watchdog(args.config, args.output)
        for name in ('health', 'watchdog'):
            (Path(args.runtime_root) / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    except Exception as exc:
        print(f'Watchdog preparation failed ({type(exc).__name__})', file=sys.stderr)
        return 2
    print('TG-only watchdog configuration prepared; credentials were not printed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
