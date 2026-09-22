# WebChain Seed/Demo SFT

WebChain GUI trace SFT data in Seed/Demo format.

## Overview

- Source traces: 31,675
- SFT rows after sliding-window conversion: 39,332
- Assistant/image turns: 263,995
- Parts: 48
- SFT parquet shards: 71
- Evaluation subset: 150 traces, 193 SFT window rows
- Coordinate space: integer 0-1000 normalized points

## Files

- `parts/part_*/sft/*.parquet`: SFT rows.
- `parts/part_*/metadata/actions.parquet`: part-level action metadata and Seed function mapping.
- `parts/part_*/metadata/windows.parquet`: part-level sliding-window metadata.
- `metadata/traces.parquet` and `metadata/traces.csv`: trace-level index.
- `metadata/actions.parquet` and `metadata/actions.csv`: action-level index.
- `metadata/windows.parquet` and `metadata/windows.csv`: window-level index.
- `metadata/action_mapping.tsv`: original action labels mapped to Seed function names.
- `metadata/intent_taxonomy.tsv`: Simple IR / Multi-Constraint / Conditional taxonomy.
- `test_suite_150/`: evaluation subset with manifest, metadata, and matching SFT rows.
- `tools/verify_seed_delivery.py`: full package and test-suite verifier.
- `tools/sample_test_suite.py`: compact reader for test-suite SFT rows.
- `tools/check_structure_urls.py`: DOM/AX URL sample checker.
- `rl_reward_spec.md`: reward design notes for RL or trajectory scoring.
- `validation_summary.json` and `verify_summary.json`: package validation summaries.
- `MANIFEST.sha256` and `dataset_index.json`: file integrity and size index.

## SFT Schema

Every SFT parquet shard uses SNAPPY compression and these columns:

| column | type | description |
|---|---|---|
| `uid` | string | Window UID in the form `{trace_uid}::w{index}`. |
| `images` | list<binary> | Viewport PNG bytes for the window. |
| `inputs` | string | JSON-encoded multi-turn conversation items. |

`inputs` contains `[CLS]`, role tokens, text/image items, and `[EOS]` tokens. Tool image items carry `image_index`, `width`, `height`, and `has_loss=0`. Assistant text and assistant `[EOS]` use `has_loss=1`; non-assistant content uses `has_loss=0`.

## Assistant Format

Assistant turns use Seed/Demo XML:

```xml
<gui_think>...</gui_think>
<seed:tool_call><function name="click"><parameter name="point" string="true"><point>x y</point></parameter></function></seed:tool_call>
```

Point coordinates are normalized with:

```text
round(pixel / image_dimension * 1000)
```

## Action Mapping

The target function names are in the Seed GUI action space:

| original action | Seed function | strategy |
|---|---|---|
| `click` | `click` | direct click point |
| `type` | `type` | text content |
| `select` | `type` | selected text content |
| `press_enter` | `hotkey` | `enter` |
| `paste` | `hotkey` | `ctrl v` |
| `copy` | `hotkey` | `ctrl c` |
| `double_click` | `left_double` | normalized point |
| `right_click` | `right_single` | normalized point |
| `drag` | `drag` | start and end points |
| `hover` | `click` | normalized point with mapping recorded |

The exact counts are in `metadata/action_mapping.tsv`.

## Metadata

`metadata/traces.csv` provides `uid`, cleaned `query`, `raw_user_query`, `intent_type`, `primary_host`, `web_type`, `first_href`, source IDs, action/image counts, window IDs, DOM/AX URL lists, and full-screenshot IDs.

`metadata/actions.csv` provides source trace and step IDs, original action type, Seed function, Seed parameters, normalized coordinates, bbox, selector, DOM path, href, DOM/AX URLs, full-screenshot IDs, CoT provenance, and Seed answer derivation fields.

`metadata/windows.csv` provides each output window UID, trace UID, window index, source turn span, step count, image count, estimated image tokens, source shard, output shard, action types, and source step indices.

## Test Suite

`test_suite_150/` contains 50 short, 50 medium, and 50 long traces by included reasoning step count. The folder includes:

- `manifest.jsonl`
- `trace_ids.tsv`
- `summary.json`
- `sft/test_suite_150.parquet`
- `metadata/traces.parquet` and `metadata/traces.csv`
- `metadata/actions.parquet` and `metadata/actions.csv`
- `metadata/windows.parquet` and `metadata/windows.csv`
- `verify_summary.json`

Sample rows:

```bash
python tools/sample_test_suite.py --root . --limit 5
```

## Verification

Run the full verifier from the package root:

```bash
python tools/verify_seed_delivery.py --root . --workers 48 --output verify_summary.json
```

Run the same verifier on the evaluation subset:

```bash
python tools/verify_seed_delivery.py --root . --test-suite-only --output test_suite_150/verify_summary.json
```

Check the file manifest:

```bash
sha256sum -c MANIFEST.sha256 --quiet
```

Sample-check DOM/AX URLs:

```bash
python tools/check_structure_urls.py --root . --limit 64 --workers 16 --output metadata/structure_url_sample_check.json
```

The verifier checks SFT schema, image metadata, loss masks, Seed XML, allowed function names, point coordinate range, action metadata agreement, and sliding-window image-token budget.

## Reward Notes

See `rl_reward_spec.md` for format, action schema, coordinate, text, sequence, and environment reward components.
