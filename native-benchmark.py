"""Bounded native acceptance test; only mutates the bundled Jev Mac Lab."""
import getpass
import io
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

SKILL = Path.home() / '.agents/skills/jev-mac-use'
sys.path.insert(0, str(SKILL / 'scripts'))
from jev import JevClient, Halt
from mac_use import Bridge, run
from benchmark import percentile


def main():
    key = getpass.getpass('TypeSafe API key (hidden): ')
    subprocess.run(['open', str(Path.home() / 'Library/Caches/jev-mac-use/JevMacLab.app')], check=True)
    bridge = Bridge('local.jev.maclab')
    client = None
    report = {'environment': {'system': platform.system(), 'machine': platform.machine()},
              'measurement': 'Native demo task wall time including final AX verification; excludes warmup and reset',
              'runs': []}
    output = Path(__file__).with_name('native-benchmark-macos.json')
    try:
        bridge.request({'cmd': 'snapshot'})
        client = JevClient(key)
        _, report['warmup'] = client.evaluate('Connection warmup.', {'ready': {
            'type': 'noul', 'instructions': 'Does the state contain the word warmup?'}})
        task = json.loads((SKILL / 'references/demo-task.json').read_text())
        modes = ['flat', 'fanout'] * 5
        random.Random(20260919).shuffle(modes)
        for index, mode in enumerate(modes):
            snap = bridge.request({'cmd': 'snapshot'})
            reset = [n for n in snap['nodes'] if n['identifier'] == 'jev.reset']
            if snap['truncated'] or len(reset) != 1:
                raise Halt('Demo reset target is unavailable or ambiguous')
            bridge.request({'cmd': 'act', 'snapshot': snap['snapshot'], 'target': reset[0]['id'], 'op': 'press'})
            bridge.request({'cmd': 'settle', 'timeout_ms': 600})
            snap = bridge.request({'cmd': 'snapshot'})
            values = {n['identifier']: n.get('value') for n in snap['nodes'] if n['identifier']}
            if values.get('jev.query') != '' or values.get('jev.status') != 'Ready':
                raise Halt('Demo reset was not verified')
            trace = io.StringIO()
            started = time.perf_counter()
            row = {'index': index, 'mode': mode}
            try:
                with redirect_stdout(io.StringIO()):
                    result = run(task, client, bridge, mode, 6, 20, True, trace)
                row.update(result)
            except Halt as exc:
                row.update(status='stopped', error=str(exc))
            row['task_ms'] = round((time.perf_counter() - started) * 1000, 3)
            row['trace'] = [json.loads(line) for line in trace.getvalue().splitlines()]
            report['runs'].append(row)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
            if row['status'] != 'verified':
                break  # Reobserve and investigate; do not replay failed mutations.
        report['summary'] = {}
        for mode in ['flat', 'fanout']:
            rows = [r for r in report['runs'] if r['mode'] == mode]
            times = [r['task_ms'] for r in rows if r['status'] == 'verified']
            report['summary'][mode] = {'n': len(rows), 'verified': len(times),
                'task_p50_ms': statistics.median(times) if times else None,
                'task_p95_ms': percentile(times, .95)}
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps({'warmup': report['warmup'], 'summary': report['summary']}, indent=2))
    except Halt as exc:
        report['error'] = str(exc)
        output.write_text(json.dumps(report, indent=2))
        raise SystemExit(str(exc))
    finally:
        bridge.close()
        if client:
            client.close()


if __name__ == '__main__':
    main()
