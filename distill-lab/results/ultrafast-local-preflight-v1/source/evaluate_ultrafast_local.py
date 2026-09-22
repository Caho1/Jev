"""Measure the real ultrafast request/response and executor against local DOM fixtures."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from unittest.mock import patch

from ultrafast_cases import cases, closed_cases
from ultrafast_laya import LAB, UltrafastLaya, make_server

UPSTREAM = LAB / 'reference/jev-ultrafast'
sys.path.insert(0, str(UPSTREAM))
from jev_ultrafast import agent as agent_module
from jev_ultrafast import browser as browser_module
from jev_ultrafast.model import action_space, choose, validate_choice
from playwright.sync_api import sync_playwright


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(context):
    records = []
    for case in cases():
        page = context.new_page()
        page.set_content(case['html'])
        if case['scroll'] is not None:
            page.evaluate('(y) => window.scrollTo(0,y)', case['scroll'])
        state = page.evaluate(browser_module.READ_STATE)
        state['fingerprint'] = browser_module.fingerprint(state)
        _, targets, controls = action_space(state['actions'])
        op, target = case['operation'], None
        if case['target']:
            node = page.evaluate('(id) => window.__jevFast.ids.get(document.getElementById(id))', case['target'])
            matching = [(index, a) for index, a in targets[op].items()
                        if a['node'] == node and (case['value'] is None or a.get('value') == case['value'])]
            assert len(matching) == 1, (case['id'], matching)
            target, action = matching[0]
            choice = action['id']
        else:
            choice = controls[op]['id'] if op in controls else op
        records.append({k: case[k] for k in ('id', 'family', 'language', 'goal', 'history')} |
                       {'page': state, 'gold': {'operation': op, 'target': target, 'choice': choice}})
        page.close()
    return records


def summarize(records):
    n = len(records)
    target = [r for r in records if r['gold']['target'] is not None]
    nontrivial = [r for r in target if r.get('target_candidates', 0) > 1]
    latency = sorted(r['elapsed_ms'] for r in records)
    return {'n': n, 'valid': sum(r['ok'] for r in records),
            'operation_correct': sum(r.get('operation_correct', False) for r in records),
            'operation_accuracy': sum(r.get('operation_correct', False) for r in records)/n,
            'step_correct': sum(r.get('step_correct', False) for r in records),
            'step_accuracy': sum(r.get('step_correct', False) for r in records)/n,
            'oracle_operation_target_n': len(target),
            'oracle_operation_target_correct': sum(r.get('target_correct', False) for r in target),
            'oracle_operation_target_accuracy': sum(r.get('target_correct', False) for r in target)/len(target) if target else None,
            'nontrivial_target_n': len(nontrivial),
            'nontrivial_target_correct': sum(r.get('target_correct', False) for r in nontrivial),
            'request_p50_ms': statistics.median(latency),
            'request_p95_ms': latency[max(0, int(0.95*n+0.999)-1)],
            'forward_passes': sum(r.get('decision', {}).get('usage', {}).get('forward_passes', 0) for r in records),
            'singleton_heads': sum(len(r.get('decision', {}).get('usage', {}).get('singleton_heads', [])) for r in records)}


def replay(rows, output):
    results = []
    with (output/'replay.jsonl').open('x', buffering=1) as stream:
        for row in rows:
            record = {k: row[k] for k in ('id', 'language', 'gold')}
            started = time.perf_counter()
            try:
                decision = choose(row['page'], row['goal'], row['history'])
                operation = row['gold']['operation']
                head = operation.lower()+'_target'
                if row['gold']['target']:
                    candidates = decision['request']['questions'][head]['criteria']
                    answer = validate_choice(decision['raw_answers'][head], candidates)
                    record['target_correct'] = answer['choice'] == row['gold']['target']
                    record['target_candidates'] = len(candidates)
                record.update(ok=True, decision=decision,
                              operation_correct=decision['operation'] == operation,
                              step_correct=decision['choice'] == row['gold']['choice'])
            except Exception as exc:
                record.update(ok=False, error_type=type(exc).__name__, error=str(exc))
            record['elapsed_ms'] = (time.perf_counter()-started)*1000
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            results.append(record)
            if len(results) % 8 == 0:
                print(json.dumps({'phase': 'replay', 'model': output.name, 'done': len(results),
                                  'total': len(rows), 'step_correct': sum(r.get('step_correct', False) for r in results)}), flush=True)
    result = summarize(results)
    result['by_language'] = {lang: summarize([r for r in results if r['language'] == lang]) for lang in ['en', 'zh']}
    result['by_operation'] = {op: summarize([r for r in results if r['gold']['operation'] == op])
                              for op in sorted({r['gold']['operation'] for r in results})}
    return result


def closed_loop(context, output):
    results = []
    for case in closed_cases():
        class OwnedBrowser(browser_module.Browser):
            def __init__(self, _url):
                self.page = context.new_page()
                self.page.set_content(case['html'])
                self.session = context.new_cdp_session(self.page)
                self.target = None

            def close(self):
                self.page.close()

        record = {'id': case['id'], 'language': case['language'], 'goal': case['goal'], 'budget_decisions': 10}
        started = time.perf_counter()
        def cdp_bridge(method, session_id=None, **params):
            return session_id.send(method, params)
        with patch.object(agent_module, 'Browser', OwnedBrowser), patch.object(browser_module, 'cdp', cdp_bridge):
            with agent_module.Agent('about:blank', case['goal']) as agent:
                try:
                    for _ in range(10):
                        state = agent.command('tick')
                        if state['status'] in {'done', 'blocked'}:
                            break
                except Exception as exc:
                    record.update(error_type=type(exc).__name__, error=str(exc))
                record['trace'] = agent.snapshot()
                record['verification'] = agent.browser.page.evaluate(case['verify'])
                record['final_dom_text'] = agent.browser.page.inner_text('body')
                record['success'] = bool(record['verification']) and agent.state['status'] == 'done'
                record['outcome_reached'] = bool(record['verification'])
                record['elapsed_ms'] = (time.perf_counter()-started)*1000
        write_json(output/f"closed-{case['id']}.json", record)
        results.append(record)
        print(json.dumps({'phase': 'closed_loop', 'model': output.name, 'case': case['id'],
                          'status': record['trace']['status'], 'verified': record['verification'],
                          'success': record['success']}), flush=True)
    return {'n': len(results), 'success': sum(r['success'] for r in results),
            'success_rate': sum(r['success'] for r in results)/len(results),
            'outcome_reached': sum(r['outcome_reached'] for r in results),
            'by_language': {lang: {'n': 4, 'success': sum(r['success'] for r in results if r['language'] == lang)}
                            for lang in ['en', 'zh']},
            'cases': [{k: r[k] for k in ('id', 'success', 'outcome_reached', 'elapsed_ms')} for r in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['laya_base', 'laya_computer'], default=['laya_base', 'laya_computer'])
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name('ultrafast_cases.py'), Path(__file__).with_name('ultrafast_laya.py'),
               Path(__file__).with_name('typed_laya.py'), Path(__file__).with_name('evaluate_computer_replay.py'),
               UPSTREAM/'jev_ultrafast/model.py', UPSTREAM/'jev_ultrafast/snapshot.js', UPSTREAM/'jev_ultrafast/agent.py',
               UPSTREAM/'jev_ultrafast/browser.py']
    protocol = {'upstream_commit': '1231850a0bf1a0c0341fe408ef1668dbbfdfac46', 'device': args.device,
                'models': args.models, 'code_sha256': {str(p.relative_to(LAB)): digest(p) for p in sources},
                'dataset': '24 authored DOM scenes × English/Chinese goals = 48 frozen diagnostic steps',
                'closed_loop': '4 authored local tasks × English/Chinese goals; 10 decisions max; no text-generation tasks',
                'limits': ['Small synthetic diagnostic; not public benchmark or production accuracy',
                           'Language pairs are correlated; page labels are English in both languages',
                           'Target accuracy uses gold operation, step accuracy requires both operation and target',
                           'Frozen correct states for replay; actual model history for closed-loop',
                           'No calibration, holdout, paid APIs or optimizer updates'],
                'browser': 'isolated headless Chrome via Playwright CDP; upstream snapshot, guards, execution, Agent loop unchanged',
                'temperature': 1.0, 'confidence_calibrated': False}
    (args.output/'source').mkdir()
    for p in sources:
        (args.output/'source'/p.name).write_bytes(p.read_bytes())
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='chrome', headless=True)
        context = browser.new_context(viewport={'width': 1120, 'height': 780}, locale='en-US')
        context.route('**/*', lambda route: route.abort())
        protocol['browser_version'] = browser.version
        rows = collect(context)
        frozen = args.output/'cases.jsonl'
        frozen.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        protocol['cases_sha256'] = digest(frozen)
        write_json(args.output/'protocol.json', protocol)
        if args.collect_only:
            print(json.dumps({'cases': len(rows), 'closed_cases': len(closed_cases())}))
            browser.close()
            return
        summaries = {}
        for variant in args.models:
            output = args.output/variant
            output.mkdir()
            started = time.perf_counter()
            backend = UltrafastLaya(variant, args.device, LAB/'results/computer-use-grounding-cuda-v1',
                                    LAB/'checkpoints/banking77-lora-v1/base')
            write_json(output/'identity.json', {**backend.identity, 'load_seconds': time.perf_counter()-started})
            server = make_server(backend, 0)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with patch.dict(os.environ, {'TYPESAFE_BASE_URL': f'http://127.0.0.1:{server.server_port}/v1',
                                             'TYPESAFE_MODEL': variant}):
                    # One untimed request per model; recorded separately, never discarded from provenance.
                    write_json(output/'warmup.json', choose(rows[0]['page'], rows[0]['goal'], []))
                    summaries[variant] = {'replay': replay(rows, output), 'closed_loop': closed_loop(context, output)}
                    write_json(output/'results.json', summaries[variant])
                    write_json(args.output/'results.json', summaries)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()
                backend.close()
        browser.close()
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
