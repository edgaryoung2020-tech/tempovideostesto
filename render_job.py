"""GitHub Actions bridge. Bot token comes only from Actions Secrets."""
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from engine import render


def download_telegram(file_id, token, destination):
    if not token: raise ValueError('Set TELEGRAM_BOT_TOKEN in GitHub Actions secrets')
    # URLs containing credentials must never be logged or included in payloads.
    try:
        req=Request('https://api.telegram.org/bot'+token+'/getFile',data=urlencode({'file_id':file_id}).encode())
        with urlopen(req,timeout=60) as r: info=json.load(r)
        if not info.get('ok'): raise ValueError('Telegram rejected file')
        path=info['result']['file_path']
        if not isinstance(path,str) or '..' in path or path.startswith('/'): raise ValueError('Invalid file path')
        with urlopen('https://api.telegram.org/file/bot'+token+'/'+path,timeout=120) as r:
            with open(destination,'wb') as f:
                total=0
                while True:
                    block=r.read(1024*1024)
                    if not block: break
                    total+=len(block)
                    if total>20*1024*1024: raise ValueError('Test input limit: 20 MiB')
                    f.write(block)
    except Exception:
        raise RuntimeError('Cannot download Telegram file. Check bot secret, file ownership and 20MB limit.') from None


def main():
    event=json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())['inputs']
    plan=json.loads(event['plan_json'])
    root=Path('job'); root.mkdir(exist_ok=True)
    download_telegram(event['file_id'],os.getenv('TELEGRAM_BOT_TOKEN',''),root/'source.mp4')
    render(root/'source.mp4',root/'result.mp4',plan)
    # Never put source, transcript or plan into a public Actions artifact.
    if (root/'result.mp4').stat().st_size>49*1024*1024:
        raise ValueError('Result exceeds test Telegram upload limit; use a shorter input')

if __name__=='__main__':
    try: main()
    except Exception as e:
        print('Render job failed: '+str(e),file=sys.stderr)
        sys.exit(1)
